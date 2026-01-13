"""
Exchange manager for handling exchange connections and API calls.
Refactored: Separated into AuthenticatedExchangeManager and PublicExchangeManager.
"""

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Dict, Optional

import ccxt

logger = logging.getLogger(__name__)

# Import normalize_symbol from utils (core module, should always be available)
from modules.common.utils.domain import normalize_symbol

try:
    from config import (
        DEFAULT_CONTRACT_TYPE,
        DEFAULT_EXCHANGE_STRING,
        DEFAULT_REQUEST_PAUSE,
    )
    from config.config_api import (
        get_binance_api_key,
        get_binance_api_secret,
    )
except ImportError:
    DEFAULT_EXCHANGE_STRING = "binance,kraken,kucoin,gate,okx,bybit,mexc,huobi"
    DEFAULT_REQUEST_PAUSE = 0.2
    DEFAULT_CONTRACT_TYPE = "future"
    DEFAULT_EXCHANGES = [
        "binance",
        "kraken",
        "kucoin",
        "gate",
        "okx",
        "bybit",
        "mexc",
        "huobi",
    ]

    # Fallback getter functions if config_api is not available
    def get_binance_api_key():
        return None

    def get_binance_api_secret():
        return None


class ExchangeWrapper:
    """
    Wrapper class to track exchange instance and reference count.
    Ensures thread-safe reference counting for exchange cleanup.
    """

    def __init__(self, exchange: ccxt.Exchange):
        """
        Initialize ExchangeWrapper with an exchange instance.

        Args:
            exchange: The ccxt.Exchange instance to wrap
        """
        self.exchange = exchange
        self._refcount = 0
        self._refcount_lock = threading.Lock()

    def increment_refcount(self) -> int:
        """
        Increment reference count atomically.

        Returns:
            New reference count after increment
        """
        with self._refcount_lock:
            self._refcount += 1
            return self._refcount

    def decrement_refcount(self) -> int:
        """
        Decrement reference count atomically.

        Returns:
            New reference count after decrement
        """
        with self._refcount_lock:
            if self._refcount > 0:
                self._refcount -= 1
            return self._refcount

    def get_refcount(self) -> int:
        """
        Get current reference count.

        Returns:
            Current reference count
        """
        with self._refcount_lock:
            return self._refcount

    def is_in_use(self) -> bool:
        """
        Check if exchange is currently in use.

        Returns:
            True if refcount > 0, False otherwise
        """
        with self._refcount_lock:
            return self._refcount > 0


class AuthenticatedExchangeManager:
    """Manages authenticated exchange connections (requires API credentials)."""

    def __init__(
        self,
        api_key=None,
        api_secret=None,
        testnet=False,
        request_pause=None,
        contract_type=None,
    ):
        """
        Initialize AuthenticatedExchangeManager.

        Args:
            api_key: Default API key (for Binance backward compatibility)
            api_secret: Default API secret (for Binance backward compatibility)
            testnet: Use testnet if True
            request_pause: Pause between requests in seconds
            contract_type: Contract type ('spot', 'margin', 'future'). Defaults to DEFAULT_CONTRACT_TYPE
        """
        # Store default credentials for Binance (backward compatibility)
        # Use thread-safe getter functions for API keys
        self.default_api_key = api_key or os.getenv("BINANCE_API_KEY") or get_binance_api_key()
        self.default_api_secret = api_secret or os.getenv("BINANCE_API_SECRET") or get_binance_api_secret()
        self.testnet = testnet
        self.contract_type = contract_type or os.getenv("DEFAULT_CONTRACT_TYPE", DEFAULT_CONTRACT_TYPE)

        # Cache for authenticated exchanges (key: exchange_id, value: ExchangeWrapper)
        self._authenticated_exchanges: Dict[str, ExchangeWrapper] = {}

        # Store credentials per exchange (key: exchange_id)
        self._exchange_credentials: Dict[str, Dict[str, str]] = {}

        # Store timestamps for exchange creation (key: cache_key, value: creation timestamp)
        self._exchange_timestamps: Dict[str, float] = {}

        self.request_pause = float(request_pause or os.getenv("BINANCE_REQUEST_SLEEP", DEFAULT_REQUEST_PAUSE))
        self._request_lock = threading.Lock()
        self._last_request_ts = 0.0

    def connect_to_exchange_with_credentials(
        self,
        exchange_id: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated exchange instance (REQUIRES credentials).

        Supports multiple exchanges: binance, okx, kucoin, bybit, gate, mexc, huobi, etc.

        Args:
            exchange_id: Exchange name (e.g., 'binance', 'okx', 'kucoin', 'bybit')
            api_key: API key for this exchange (optional, uses default if not provided)
            api_secret: API secret for this exchange (optional, uses default if not provided)
            testnet: Use testnet if True (optional, uses instance default if not provided)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated exchange instance

        Raises:
            ValueError: If exchange not supported or credentials not provided
        """
        exchange_id = exchange_id.strip().lower()
        testnet = testnet if testnet is not None else self.testnet
        contract_type = contract_type or self.contract_type

        # Check cache and increment refcount if found
        cache_key = f"{exchange_id}_{testnet}_{contract_type}"
        with self._request_lock:
            if cache_key in self._authenticated_exchanges:
                wrapper = self._authenticated_exchanges[cache_key]
                wrapper.increment_refcount()
                return wrapper.exchange

        # Exchange not in cache, need to create new one
        # Note: Lock is released here to allow other threads to proceed
        # We'll re-acquire lock when storing the new exchange

        # Get credentials (per-exchange or default)
        if exchange_id == "binance":
            # Use default credentials for Binance (backward compatibility)
            cred_key = api_key or self.default_api_key
            cred_secret = api_secret or self.default_api_secret
        else:
            # For other exchanges, try per-exchange credentials or default
            exchange_creds = self._exchange_credentials.get(exchange_id, {})
            cred_key = api_key or exchange_creds.get("api_key") or self.default_api_key
            cred_secret = api_secret or exchange_creds.get("api_secret") or self.default_api_secret

        if not cred_key or not cred_secret:
            raise ValueError(
                f"API Key và API Secret là bắt buộc cho {exchange_id}!\n"
                f"Cung cấp qua một trong các cách sau:\n"
                f"  1. Tham số khi gọi connect_to_exchange_with_credentials()\n"
                f"  2. Sử dụng set_exchange_credentials() để set credentials cho exchange\n"
                f"  3. Biến môi trường: {exchange_id.upper()}_API_KEY và {exchange_id.upper()}_API_SECRET\n"
                f"  4. File config: modules/config_api.py"
            )

        # Check if exchange is supported
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt.")

        exchange_class = getattr(ccxt, exchange_id)

        # Build options with configurable contract type
        options = {
            "defaultType": contract_type,
            "options": {
                "defaultType": contract_type,
            },
        }

        # Build params
        params = {
            "apiKey": cred_key,
            "secret": cred_secret,
            "options": options,
            "enableRateLimit": True,
        }

        # Add testnet/sandbox for supported exchanges
        if testnet:
            if exchange_id == "binance":
                params["sandbox"] = True
            elif exchange_id in ["okx", "kucoin", "bybit"]:
                params["sandbox"] = True
            # Some exchanges use 'test' instead of 'sandbox'
            elif exchange_id in ["gate"]:
                params["options"]["sandboxMode"] = True

        try:
            exchange_instance = exchange_class(params)
            # Wrap exchange and set initial refcount to 1 (since we're returning it)
            wrapper = ExchangeWrapper(exchange_instance)
            wrapper.increment_refcount()  # Set to 1 for the caller
            with self._request_lock:
                # Double-check: another thread might have created it while we were creating
                if cache_key in self._authenticated_exchanges:
                    # Another thread created it first, use that one and close ours
                    existing_wrapper = self._authenticated_exchanges[cache_key]
                    existing_wrapper.increment_refcount()
                    # Try to close the exchange we just created to avoid resource leak
                    if hasattr(exchange_instance, "close"):
                        try:
                            exchange_instance.close()
                        except Exception:
                            pass  # Ignore errors during cleanup
                    return existing_wrapper.exchange
                else:
                    # Store our newly created exchange
                    self._authenticated_exchanges[cache_key] = wrapper
                    # Store creation timestamp for age-based cleanup
                    self._exchange_timestamps[cache_key] = time.time()
            return wrapper.exchange

        except Exception as exc:
            raise ValueError(f"Cannot initialize authenticated {exchange_id} exchange: {exc}")

    def set_exchange_credentials(self, exchange_id: str, api_key: str, api_secret: str):
        """
        Set credentials for a specific exchange.

        Args:
            exchange_id: Exchange name (e.g., 'okx', 'kucoin', 'bybit')
            api_key: API key for this exchange
            api_secret: API secret for this exchange
        """
        exchange_id = exchange_id.strip().lower()
        self._exchange_credentials[exchange_id] = {
            "api_key": api_key,
            "api_secret": api_secret,
        }
        # Clear cached exchange if exists to force reconnection with new credentials
        # Only remove if not in use (refcount = 0)
        with self._request_lock:
            keys_to_remove = []
            for k, wrapper in list(self._authenticated_exchanges.items()):
                if k.startswith(f"{exchange_id}_"):
                    if not wrapper.is_in_use():
                        keys_to_remove.append(k)
                    else:
                        logger.warning(f"Cannot clear exchange {k} - still in use (refcount={wrapper.get_refcount()})")
            for key in keys_to_remove:
                wrapper = self._authenticated_exchanges.pop(key)
                # Remove timestamp if exists
                if key in self._exchange_timestamps:
                    del self._exchange_timestamps[key]
                # Try to close exchange if it has a close method
                if hasattr(wrapper.exchange, "close"):
                    try:
                        wrapper.exchange.close()
                    except Exception as e:
                        logger.warning(f"Error closing exchange {key}: {e}")

    def update_default_credentials(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Update default credentials used for authenticated exchanges and clear caches.

        Args:
            api_key: New default API key (if None, keep existing)
            api_secret: New default API secret (if None, keep existing)
        """
        updated = False
        if api_key is not None:
            self.default_api_key = api_key
            updated = True
        if api_secret is not None:
            self.default_api_secret = api_secret
            updated = True
        if updated:
            # Only clear exchanges that are not in use
            with self._request_lock:
                keys_to_remove = []
                for k, wrapper in list(self._authenticated_exchanges.items()):
                    if not wrapper.is_in_use():
                        keys_to_remove.append(k)
                    else:
                        logger.warning(f"Cannot clear exchange {k} - still in use (refcount={wrapper.get_refcount()})")
                for key in keys_to_remove:
                    wrapper = self._authenticated_exchanges.pop(key)
                    # Remove timestamp if exists
                    if key in self._exchange_timestamps:
                        del self._exchange_timestamps[key]
                    if hasattr(wrapper.exchange, "close"):
                        try:
                            wrapper.exchange.close()
                        except Exception as e:
                            logger.warning(f"Error closing exchange {key}: {e}")

    def cleanup_unused_exchanges(self, max_age_hours: Optional[float] = None):
        """
        Cleanup cached authenticated exchange connections that are not in use.

        This frees memory and closes unused connections by removing only exchanges
        with reference count of 0. Exchanges that are currently in use (refcount > 0)
        will not be cleaned up to prevent TOCTOU race conditions.

        Args:
            max_age_hours: Optional maximum age in hours. If None, cleans up all unused exchanges.
                          If provided, only unused exchanges older than max_age_hours will be closed and removed.

        Stored credentials are NOT cleared by this method.

        Note: This method is thread-safe and will not remove exchanges that are
        currently being used by other threads.
        """
        with self._request_lock:
            current_time = time.time()
            keys_to_remove = []

            for cache_key, wrapper in list(self._authenticated_exchanges.items()):
                # Only consider unused exchanges
                if not wrapper.is_in_use():
                    # If max_age_hours is provided, check age
                    if max_age_hours is not None:
                        creation_time = self._exchange_timestamps.get(cache_key)
                        # If no timestamp exists, treat as old (close it to be safe)
                        if creation_time is None:
                            keys_to_remove.append(cache_key)
                        else:
                            age_seconds = current_time - creation_time
                            max_age_seconds = max_age_hours * 3600
                            if age_seconds > max_age_seconds:
                                keys_to_remove.append(cache_key)
                    else:
                        # No age filter, remove all unused
                        keys_to_remove.append(cache_key)

            cleared_count = 0
            for cache_key in keys_to_remove:
                wrapper = self._authenticated_exchanges.pop(cache_key)
                # Remove timestamp if exists
                if cache_key in self._exchange_timestamps:
                    del self._exchange_timestamps[cache_key]
                cleared_count += 1
                # Close exchange before removing from cache
                if hasattr(wrapper.exchange, "close"):
                    try:
                        wrapper.exchange.close()
                    except Exception as e:
                        logger.warning(f"Error closing exchange {cache_key}: {e}")

            if cleared_count > 0:
                if max_age_hours is not None:
                    logger.info(
                        f"Cleaned up {cleared_count} unused authenticated "
                        f"exchange connections (older than {max_age_hours} hours)"
                    )
                else:
                    logger.info(f"Cleaned up {cleared_count} unused authenticated exchange connections")
            else:
                if max_age_hours is not None:
                    logger.debug(f"No unused exchanges found older than {max_age_hours} hours to clean up")
                else:
                    logger.debug("No unused exchanges to clean up")

    def close_exchange(self, exchange_id: str, testnet: bool = False, contract_type: str = None):
        """
        Close and remove a specific exchange connection.

        Only closes exchanges that are not in use (refcount = 0). If the exchange
        is currently in use, a warning is logged and the exchange is not closed.

        Args:
            exchange_id: Exchange identifier
            testnet: Testnet flag
            contract_type: Contract type
        """
        contract_type = contract_type or self.contract_type
        cache_key = f"{exchange_id}_{testnet}_{contract_type}"

        with self._request_lock:
            if cache_key in self._authenticated_exchanges:
                wrapper = self._authenticated_exchanges[cache_key]
                if wrapper.is_in_use():
                    logger.warning(
                        f"Cannot close exchange {cache_key} - still in use (refcount={wrapper.get_refcount()})"
                    )
                    return

                wrapper = self._authenticated_exchanges.pop(cache_key)
                # Remove timestamp if exists
                if cache_key in self._exchange_timestamps:
                    del self._exchange_timestamps[cache_key]
                # Try to close exchange if it has a close method
                if hasattr(wrapper.exchange, "close"):
                    try:
                        wrapper.exchange.close()
                    except Exception as e:
                        logger.warning(f"Error closing exchange {exchange_id}: {e}")
                logger.debug(f"Closed exchange connection: {cache_key}")

    def release_exchange(self, exchange_id: str, testnet: bool = False, contract_type: str = None):
        """
        Release a reference to an exchange, decrementing its reference count.

        Call this method when you're done using an exchange that was retrieved
        via connect_to_exchange_with_credentials(). This allows cleanup to proceed
        when all references are released.

        Args:
            exchange_id: Exchange identifier
            testnet: Testnet flag
            contract_type: Contract type

        Note:
            This method is safe to call even if the exchange has already been
            removed from the cache (e.g., by cleanup). It will simply do nothing
            in that case.
        """
        contract_type = contract_type or self.contract_type
        cache_key = f"{exchange_id}_{testnet}_{contract_type}"

        with self._request_lock:
            if cache_key in self._authenticated_exchanges:
                wrapper = self._authenticated_exchanges[cache_key]
                new_refcount = wrapper.decrement_refcount()
                logger.debug(f"Released reference to {cache_key}, refcount now: {new_refcount}")
            else:
                # Exchange already removed from cache, nothing to release
                logger.debug(f"Attempted to release reference to {cache_key}, but exchange not in cache")

    @contextmanager
    def exchange_context(
        self,
        exchange_id: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ):
        """
        Context manager for safely using an exchange with automatic reference management.

        Usage:
            with manager.exchange_context('binance') as exchange:
                # Use exchange here
                ticker = exchange.fetch_ticker('BTC/USDT')
            # Exchange reference is automatically released on exit

        Args:
            exchange_id: Exchange identifier
            api_key: Optional API key
            api_secret: Optional API secret
            testnet: Optional testnet flag
            contract_type: Optional contract type

        Yields:
            ccxt.Exchange: The exchange instance

        Note:
            The reference count is automatically decremented when exiting the context,
            even if an exception occurs.
        """
        exchange = None
        try:
            exchange = self.connect_to_exchange_with_credentials(
                exchange_id, api_key, api_secret, testnet, contract_type
            )
            yield exchange
        finally:
            if exchange is not None:
                # Determine parameters to release
                testnet_val = testnet if testnet is not None else self.testnet
                contract_type_val = contract_type or self.contract_type
                try:
                    self.release_exchange(exchange_id, testnet_val, contract_type_val)
                except Exception as e:
                    # Log but don't raise - we're in cleanup
                    logger.warning(f"Error releasing exchange {exchange_id}: {e}")

    def connect_to_binance_with_credentials(self) -> ccxt.Exchange:
        """
        Connect to authenticated Binance exchange instance (REQUIRES credentials).

        Use this for:
        - fetch_ticker() - Get current prices
        - load_markets() - List available symbols
        - fetch_positions() - Get account positions
        - Any authenticated API calls

        Returns:
            ccxt.Exchange: Authenticated Binance exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials("binance")

    def connect_to_kraken_with_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated Kraken exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('kraken').

        Args:
            api_key: API key for Kraken (optional, uses set credentials or default)
            api_secret: API secret for Kraken (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated Kraken exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials("kraken", api_key, api_secret, testnet, contract_type)

    def connect_to_kucoin_with_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated KuCoin exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('kucoin').

        Args:
            api_key: API key for KuCoin (optional, uses set credentials or default)
            api_secret: API secret for KuCoin (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated KuCoin exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials("kucoin", api_key, api_secret, testnet, contract_type)

    def connect_to_gate_with_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated Gate.io exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('gate').

        Args:
            api_key: API key for Gate.io (optional, uses set credentials or default)
            api_secret: API secret for Gate.io (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated Gate.io exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials("gate", api_key, api_secret, testnet, contract_type)

    def connect_to_okx_with_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated OKX exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('okx').

        Args:
            api_key: API key for OKX (optional, uses set credentials or default)
            api_secret: API secret for OKX (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated OKX exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials("okx", api_key, api_secret, testnet, contract_type)

    def connect_to_bybit_with_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated Bybit exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('bybit').

        Args:
            api_key: API key for Bybit (optional, uses set credentials or default)
            api_secret: API secret for Bybit (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated Bybit exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials("bybit", api_key, api_secret, testnet, contract_type)

    def connect_to_mexc_with_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated MEXC exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('mexc').

        Args:
            api_key: API key for MEXC (optional, uses set credentials or default)
            api_secret: API secret for MEXC (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated MEXC exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials("mexc", api_key, api_secret, testnet, contract_type)

    def connect_to_huobi_with_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated Huobi exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('huobi').

        Args:
            api_key: API key for Huobi (optional, uses set credentials or default)
            api_secret: API secret for Huobi (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated Huobi exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials("huobi", api_key, api_secret, testnet, contract_type)

    def throttled_call(self, func, *args, **kwargs):
        """
        Ensures a minimum delay between REST calls to respect rate limits.

        Args:
            func: Function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func(*args, **kwargs)
        """
        with self._request_lock:
            wait = self.request_pause - (time.time() - self._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            result = func(*args, **kwargs)
            self._last_request_ts = time.time()
            return result


class PublicExchangeManager:
    """Manages public exchange connections (no credentials required)."""

    def __init__(self, request_pause=None):
        self.request_pause = float(request_pause or os.getenv("BINANCE_REQUEST_SLEEP", DEFAULT_REQUEST_PAUSE))
        self._request_lock = threading.Lock()
        self._last_request_ts = 0.0
        self._public_exchanges: Dict[str, ccxt.Exchange] = {}

        # Store timestamps for exchange creation (key: exchange_id, value: creation timestamp)
        self._exchange_timestamps: Dict[str, float] = {}
        fallback_string = os.getenv("OHLCV_FALLBACKS", DEFAULT_EXCHANGE_STRING)
        self._exchange_priority_for_fallback = fallback_string.split(",")

    def connect_to_exchange_with_no_credentials(self, exchange_id: str) -> ccxt.Exchange:
        """
        Connect to public exchange instance (NO credentials required).

        Use this for:
        - fetch_ohlcv() - Get historical OHLCV data (public data)
        - fetch_ticker() - Get current prices (public data, if supported)
        - Any public API calls that don't require authentication

        Args:
            exchange_id: Exchange name (e.g., 'binance', 'kraken', 'kucoin')

        Returns:
            ccxt.Exchange: Public exchange instance (no authentication)

        Raises:
            ValueError: If exchange is not supported by ccxt or cannot be initialized
        """
        exchange_id = exchange_id.strip().lower()

        # First check cache with lock to prevent race conditions
        with self._request_lock:
            if exchange_id in self._public_exchanges:
                return self._public_exchanges[exchange_id]

        # Exchange not in cache, need to create new one
        # Note: Lock is released here to allow other threads to proceed
        # We'll re-acquire lock when storing the new exchange

        # Check if exchange is supported
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt.")

        exchange_class = getattr(ccxt, exchange_id)
        contract_type = os.getenv("DEFAULT_CONTRACT_TYPE", DEFAULT_CONTRACT_TYPE)
        params = {
            "enableRateLimit": True,
            "options": {
                "defaultType": contract_type,
            },
        }

        try:
            exchange_instance = exchange_class(params)
            with self._request_lock:
                # Double-check: another thread might have created it while we were creating
                if exchange_id in self._public_exchanges:
                    # Another thread created it first, use that one and close ours
                    existing_exchange = self._public_exchanges[exchange_id]
                    # Try to close the exchange we just created to avoid resource leak
                    if hasattr(exchange_instance, "close"):
                        try:
                            exchange_instance.close()
                        except Exception:
                            pass  # Ignore errors during cleanup
                    return existing_exchange
                else:
                    # Store our newly created exchange
                    self._public_exchanges[exchange_id] = exchange_instance
                    # Store creation timestamp for age-based cleanup
                    self._exchange_timestamps[exchange_id] = time.time()
            return exchange_instance
        except Exception as exc:
            raise ValueError(f"Cannot initialize exchange {exchange_id}: {exc}")

    def throttled_call(self, func, *args, **kwargs):
        """
        Ensures a minimum delay between REST calls to respect rate limits.

        Args:
            func: Function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func(*args, **kwargs)
        """
        with self._request_lock:
            wait = self.request_pause - (time.time() - self._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            result = func(*args, **kwargs)
            self._last_request_ts = time.time()
            return result

    @property
    def exchange_priority_for_fallback(self):
        """Get list of exchange IDs in priority order for fallback."""
        return self._exchange_priority_for_fallback

    @exchange_priority_for_fallback.setter
    def exchange_priority_for_fallback(self, value):
        """Set list of exchange IDs in priority order for fallback."""
        self._exchange_priority_for_fallback = value

    def cleanup_unused_exchanges(self, max_age_hours: Optional[float] = None):
        """
        Cleanup unused public exchange connections, closing them properly before removal.
        This helps free memory and close unused connections.

        Args:
            max_age_hours: Optional maximum age in hours. If None, cleans up all exchanges.
                          If provided, only exchanges older than max_age_hours will be closed and removed.

        Note: This method ensures proper resource cleanup by closing each exchange
        before removing it from the cache, preventing resource leaks.
        """
        with self._request_lock:
            current_time = time.time()
            keys_to_remove = []

            for exchange_id, exchange in list(self._public_exchanges.items()):
                remove_exchange = False
                if max_age_hours is not None:
                    creation_time = self._exchange_timestamps.get(exchange_id)
                    if creation_time is None:
                        # No timestamp; treat as stale and mark for removal
                        remove_exchange = True
                    else:
                        age_seconds = current_time - creation_time
                        max_age_seconds = max_age_hours * 3600
                        if age_seconds > max_age_seconds:
                            remove_exchange = True
                else:
                    # If no age filter, remove all
                    remove_exchange = True

                if remove_exchange:
                    keys_to_remove.append(exchange_id)

            cleared_count = 0
            for exchange_id in keys_to_remove:
                exchange = self._public_exchanges.pop(exchange_id)
                # Remove timestamp if exists
                if exchange_id in self._exchange_timestamps:
                    del self._exchange_timestamps[exchange_id]
                cleared_count += 1
                # Close exchange before removing from cache
                if hasattr(exchange, "close"):
                    try:
                        exchange.close()
                    except Exception as e:
                        logger.warning(f"Error closing public exchange {exchange_id}: {e}")

            if cleared_count > 0:
                if max_age_hours is not None:
                    logger.info(
                        f"Cleaned up {cleared_count} public exchange connections (older than {max_age_hours} hours)"
                    )
                else:
                    logger.info(f"Cleaned up {cleared_count} public exchange connections")
            else:
                if max_age_hours is not None:
                    logger.debug(f"No exchanges found older than {max_age_hours} hours to clean up")
                else:
                    logger.debug("No exchanges to clean up")

    def close_exchange(self, exchange_id: str):
        """
        Close and remove a specific public exchange connection.

        Args:
            exchange_id: Exchange identifier
        """
        with self._request_lock:
            if exchange_id in self._public_exchanges:
                exchange = self._public_exchanges.pop(exchange_id)
                # Remove timestamp if exists
                if exchange_id in self._exchange_timestamps:
                    del self._exchange_timestamps[exchange_id]
                # Try to close exchange if it has a close method
                if hasattr(exchange, "close"):
                    try:
                        exchange.close()
                    except Exception as e:
                        logger.warning(f"Error closing public exchange {exchange_id}: {e}")
                logger.debug(f"Closed public exchange connection: {exchange_id}")


class ExchangeManager:
    """
    Composite manager that combines AuthenticatedExchangeManager and PublicExchangeManager.

    This class provides a unified interface while maintaining clear separation
    between authenticated and public exchange operations.
    """

    def __init__(self, api_key=None, api_secret=None, testnet=False):
        """
        Initialize ExchangeManager with optional credentials.

        Args:
            api_key: Binance API key (optional, can be set via env or config)
            api_secret: Binance API secret (optional, can be set via env or config)
            testnet: Use Binance testnet if True
        """
        # Initialize sub-managers
        self.authenticated = AuthenticatedExchangeManager(api_key, api_secret, testnet)
        self.public = PublicExchangeManager()

        # Store credentials for backward compatibility
        # Use thread-safe getter functions for API keys
        self.api_key = api_key or os.getenv("BINANCE_API_KEY") or get_binance_api_key()
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET") or get_binance_api_secret()
        self.testnet = testnet

    def normalize_symbol(self, market_symbol: str) -> str:
        """
        Normalize market symbol by converting Binance futures symbols like BTC/USDT:USDT into BTC/USDT.

        This method handles the conversion of exchange-specific symbol formats to a standardized format.
        It first removes contract markers (e.g., ':USDT') and then uses the utility normalize_symbol function.

        Args:
            market_symbol: Market symbol from exchange (e.g., 'BTC/USDT:USDT', 'ETHUSDT')

        Returns:
            str: Normalized symbol in format 'BASE/QUOTE' (e.g., 'BTC/USDT')
        """
        if ":" in market_symbol:
            market_symbol = market_symbol.split(":")[0]
        return normalize_symbol(market_symbol)

    @property
    def exchange_priority_for_fallback(self):
        """Get list of exchange IDs in priority order for OHLCV fallback."""
        return self.public.exchange_priority_for_fallback

    @exchange_priority_for_fallback.setter
    def exchange_priority_for_fallback(self, value):
        """Set list of exchange IDs in priority order for OHLCV fallback."""
        self.public.exchange_priority_for_fallback = value

    def cleanup_unused_exchanges(self, max_age_hours: Optional[float] = None):
        """
        Cleanup unused exchange connections to free memory.

        Args:
            max_age_hours: Optional maximum age in hours for authenticated exchanges.
                          If None, cleans up all unused exchanges.
                          For authenticated exchanges, only removes connections older than max_age_hours.
                          Public exchanges are always cleaned up (age tracking not yet implemented).
        """
        self.authenticated.cleanup_unused_exchanges(max_age_hours)
        self.public.cleanup_unused_exchanges(max_age_hours)

    def close_exchange(self, exchange_id: str, testnet: bool = False, contract_type: str = None):
        """
        Close and remove a specific exchange connection (both authenticated and public).

        Args:
            exchange_id: Exchange identifier
            testnet: Testnet flag (applies to authenticated exchanges)
            contract_type: Contract type (applies to authenticated exchanges)

        This method attempts to close the exchange connection with the given ID for both
        authenticated and public managers (if applicable).
        """
        self.authenticated.close_exchange(exchange_id, testnet, contract_type)
        self.public.close_exchange(exchange_id)
