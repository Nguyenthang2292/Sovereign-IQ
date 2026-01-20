"""Symbol fetcher module for batch scanner."""

import sys
import time
from typing import List, Optional

from modules.common.core.exchange_manager import PublicExchangeManager
from modules.common.ui.logging import log_warn


class SymbolFetchError(Exception):
    """Custom exception for symbol fetching errors."""

    def __init__(self, message: str, original_exception: Optional[Exception] = None, is_retryable: bool = False):
        super().__init__(message)
        self.original_exception = original_exception
        self.is_retryable = is_retryable


def get_all_symbols_from_exchange(
    exchange_name: str = "binance", quote_currency: str = "USDT", max_retries: int = 3, retry_delay: float = 1.0
) -> List[str]:
    """
    Get all trading symbols from exchange with retry logic for transient errors.
    """
    saved_stdin = None
    if sys.platform == "win32" and sys.stdin is not None:
        try:
            saved_stdin = sys.stdin
            if hasattr(sys.stdin, "closed") and sys.stdin.closed:
                try:
                    sys.stdin = open("CON", "r", encoding="utf-8", errors="replace")
                    saved_stdin = sys.stdin
                except (OSError, IOError):
                    pass
        except (AttributeError, ValueError, OSError, IOError):
            pass

    try:
        last_exception = None
        public_exchange_manager = PublicExchangeManager()

        for attempt in range(max_retries):
            try:
                exchange = public_exchange_manager.connect_to_exchange_with_no_credentials(exchange_name)
                markets = exchange.load_markets()

                symbols = []
                for symbol, market in markets.items():
                    if (
                        market.get("quote") == quote_currency
                        and market.get("active", True)
                        and market.get("type") == "spot"
                    ):
                        symbols.append(symbol)

                symbols.sort()
                return symbols

            except Exception as e:
                last_exception = e
                error_message = str(e)

                # Determine if error is retryable
                error_code = None
                try:
                    if hasattr(e, "status_code"):
                        error_code = e.status_code
                    elif hasattr(e, "code"):
                        error_code = e.code
                except AttributeError:
                    pass

                if error_code is None:
                    if "503" in error_message or "UNAVAILABLE" in error_message.upper():
                        error_code = 503
                    elif "429" in error_message or "RATE_LIMIT" in error_message.upper():
                        error_code = 429

                is_retryable = (
                    error_code in [503, 429]
                    or "overloaded" in error_message.lower()
                    or "rate limit" in error_message.lower()
                    or "unavailable" in error_message.lower()
                    or "timeout" in error_message.lower()
                    or "connection" in error_message.lower()
                    or "network" in error_message.lower()
                )

                if attempt < max_retries - 1 and is_retryable:
                    wait_time = retry_delay * (2**attempt)
                    log_warn(
                        f"Retryable error getting symbols (attempt {attempt + 1}/{max_retries}): {error_message}. "
                        f"Waiting {wait_time}s before retrying..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    if is_retryable:
                        raise SymbolFetchError(
                            f"Failed to fetch symbols after {max_retries} attempts: {error_message}",
                            original_exception=e,
                            is_retryable=True,
                        ) from e
                    else:
                        raise SymbolFetchError(
                            f"Failed to fetch symbols (non-retryable error): {error_message}",
                            original_exception=e,
                            is_retryable=False,
                        ) from e

        if last_exception:
            raise SymbolFetchError(
                f"Failed to fetch symbols after {max_retries} attempts",
                original_exception=last_exception,
                is_retryable=True,
            ) from last_exception

        return []

    finally:
        if sys.platform == "win32" and saved_stdin is not None:
            try:
                if sys.stdin is None or (hasattr(sys.stdin, "closed") and sys.stdin.closed):
                    if saved_stdin is not None and not (hasattr(saved_stdin, "closed") and saved_stdin.closed):
                        sys.stdin = saved_stdin
                    else:
                        try:
                            sys.stdin = open("CON", "r", encoding="utf-8", errors="replace")
                        except (OSError, IOError):
                            pass
            except (AttributeError, ValueError, OSError, IOError):
                pass
