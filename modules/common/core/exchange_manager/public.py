"""
Public exchange manager for handling public (non-authenticated) exchange connections.

This module provides the PublicExchangeManager class for managing public exchange
connections that do not require API credentials. It handles exchange caching,
rate limiting, and cleanup operations.
"""

import logging
import os
import threading
import time
from typing import Dict, Optional

import ccxt

from .base import ExchangeWrapper, DEFAULT_REQUEST_PAUSE

logger = logging.getLogger(__name__)

# Import fallback defaults
try:
    from config import DEFAULT_EXCHANGE_STRING, DEFAULT_CONTRACT_TYPE
except ImportError:
    DEFAULT_EXCHANGE_STRING = "binance,kraken,kucoin,gate,okx,bybit,mexc,huobi"
    DEFAULT_CONTRACT_TYPE = "future"


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
