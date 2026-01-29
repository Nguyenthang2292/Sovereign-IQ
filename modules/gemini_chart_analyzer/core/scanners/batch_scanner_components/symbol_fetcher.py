"""
Symbol Fetcher Component

Handles retrieval of trading symbols from exchanges with retry logic for transient errors.
"""

import time
from typing import List

from modules.common.core.exchange_manager import PublicExchangeManager
from modules.common.ui.logging import log_error, log_warn
from modules.gemini_chart_analyzer.core.exceptions import DataFetchError


class SymbolFetcher:
    """
    Fetches trading symbols from exchanges with retry logic.

    Provides methods to retrieve all active symbols for a specific quote currency
    with automatic retry on transient errors (network issues, rate limits, etc.).
    """

    def __init__(self, exchange_name: str = "binance", quote_currency: str = "USDT"):
        """
        Initialize SymbolFetcher.

        Args:
            exchange_name: Name of the exchange to fetch from (default: 'binance')
            quote_currency: Quote currency to filter symbols (default: 'USDT')
        """
        self.exchange_name = exchange_name
        self.quote_currency = quote_currency
        self.public_exchange_manager = PublicExchangeManager()

    def get_all_symbols(self, max_retries: int = 3, retry_delay: float = 1.0) -> List[str]:
        """
        Get all trading symbols from exchange with retry logic for transient errors.

        Args:
            max_retries: Maximum number of retry attempts for transient errors (default: 3)
            retry_delay: Initial delay in seconds for exponential backoff (default: 1.0)

        Returns:
            List of symbol strings (e.g., ['BTC/USDT', 'ETH/USDT', ...])
            Empty list if no symbols found (but no error occurred)

        Raises:
            DataFetchError: If symbol fetching fails after all retries or encounters
                          a non-retryable error. The exception includes information
                          about whether the error is retryable and the original exception.
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                # Use public exchange manager (no credentials needed for load_markets)
                exchange = self.public_exchange_manager.connect_to_exchange_with_no_credentials(self.exchange_name)

                # Binance load_markets() hits spot + fapi + dapi; dapi can be slow. Use longer timeout.
                original_timeout = getattr(exchange, "timeout", None)
                if self.exchange_name == "binance":
                    exchange.timeout = 30000  # 30s for load_markets (dapi.binance.com often slow)
                try:
                    # Load markets
                    markets = exchange.load_markets()
                finally:
                    exchange.timeout = original_timeout if original_timeout is not None else 10000

                # Filter by quote currency and active status
                symbols = []
                for symbol, market in markets.items():
                    if (
                        market.get("quote") == self.quote_currency
                        and market.get("active", True)
                        and market.get("type") == "spot"
                    ):  # Only spot markets
                        symbols.append(symbol)

                # Sort alphabetically
                symbols.sort()

                # Success - return symbols (empty list is valid if no symbols match criteria)
                return symbols

            except Exception as e:
                last_exception = e
                error_message = str(e)
                # Include cause chain (e.g. RequestTimeout wraps ReadTimeoutError with "timed out")
                cause_msg = str(e.__cause__).lower() if e.__cause__ else ""

                # Determine if error is retryable (network errors, rate limits, temporary unavailability)
                error_code = getattr(e, "status_code", None) or getattr(e, "code", None)
                if error_code is None and ("503" in error_message or "UNAVAILABLE" in error_message.upper()):
                    error_code = 503
                if error_code is None and ("429" in error_message or "RATE_LIMIT" in error_message.upper()):
                    error_code = 429

                is_retryable = (
                    error_code in [503, 429]
                    or "overloaded" in error_message.lower()
                    or "rate limit" in error_message.lower()
                    or "unavailable" in error_message.lower()
                    or "timeout" in error_message.lower()
                    or "timed out" in error_message.lower()
                    or "timeout" in cause_msg
                    or "timed out" in cause_msg
                    or type(e).__name__ == "RequestTimeout"
                    or "connection" in error_message.lower()
                    or "network" in error_message.lower()
                )

                # Log the error
                if attempt < max_retries - 1 and is_retryable:
                    wait_time = retry_delay * (2**attempt)
                    log_warn(
                        f"Retryable error getting symbols (attempt {attempt + 1}/{max_retries}): {error_message}. "
                        f"Waiting {wait_time}s before retrying..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error or final attempt failed
                    log_error(f"Error getting symbols: {error_message}")
                    if is_retryable:
                        # Final retry failed
                        raise DataFetchError(
                            f"Failed to fetch symbols after {max_retries} attempts: {error_message}"
                        ) from e
                    else:
                        # Non-retryable error
                        raise DataFetchError(f"Failed to fetch symbols (non-retryable error): {error_message}") from e

        # This should never be reached, but just in case
        if last_exception:
            raise DataFetchError(
                f"Failed to fetch symbols after {max_retries} attempts: {last_exception}"
            ) from last_exception
        return []  # unreachable; satisfies return type

    def cleanup(self):
        """Cleanup exchange connections and resources."""
        try:
            if hasattr(self.public_exchange_manager, "cleanup_unused_exchanges"):
                self.public_exchange_manager.cleanup_unused_exchanges()
            clear_fn = getattr(self.public_exchange_manager, "clear", None)
            if callable(clear_fn):
                clear_fn()
        except Exception as e:
            log_warn(f"Error cleaning up public exchange manager: {e}")
