"""Symbol discovery for spot and futures markets."""

import sys
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

from modules.common.ui.logging import log_error, log_exchange, log_warn
from modules.common.ui.progress_bar import ProgressBar

from .exceptions import SymbolFetchError

if TYPE_CHECKING:
    from .base import DataFetcherBase


class SymbolDiscovery:
    """Handles symbol discovery for spot and futures markets."""

    def __init__(self, base: "DataFetcherBase"):
        """
        Initialize SymbolDiscovery.

        Args:
            base: DataFetcherBase instance for accessing exchange_manager and state
        """
        self.base = base

    def list_binance_futures_symbols(
        self,
        exclude_symbols: Optional[set] = None,
        max_candidates: Optional[int] = None,
        progress_label: str = "Symbol Discovery",
    ) -> List[str]:
        """
        Lists available Binance USDT-M futures symbols sorted by quote volume.

        Args:
            exclude_symbols: Symbols to exclude (normalized format, e.g., 'BTC/USDT').
            max_candidates: Optional maximum number of symbols to return.
            progress_label: Label for the progress bar (default: "Symbol Discovery").

        Returns:
            List of symbol strings sorted by descending volume.
        """
        exclude_symbols = exclude_symbols or set()
        candidates: List[Tuple[str, float]] = []

        try:
            # Use public API - load_markets() doesn't require authentication
            exchange = self.base.exchange_manager.public.connect_to_exchange_with_no_credentials("binance")
        except Exception as exc:
            log_error(f"Unable to connect to Binance: {exc}")
            return []

        try:
            # load_markets() is a public API call, no authentication needed
            markets = self.base.exchange_manager.public.throttled_call(exchange.load_markets)
        except Exception as exc:
            log_error(f"Failed to load Binance markets: {exc}")
            return []

        progress = ProgressBar(len(markets), progress_label)
        seen = set()

        for market in markets.values():
            if self.base.should_stop():
                log_warn("Symbol discovery aborted due to shutdown.")
                break
            if not market.get("contract"):
                progress.update()
                continue
            if market.get("quote") != "USDT":
                progress.update()
                continue
            if not market.get("active", True):
                progress.update()
                continue

            symbol = self.base.exchange_manager.normalize_symbol(market.get("symbol", ""))
            if symbol in exclude_symbols or symbol in seen:
                progress.update()
                continue

            info = market.get("info", {})
            volume_str = info.get("volume") or info.get("quoteVolume") or info.get("turnover")
            try:
                volume = float(volume_str)
            except (TypeError, ValueError):
                volume = 0.0

            candidates.append((symbol, volume))
            seen.add(symbol)
            progress.update()

        progress.finish()

        candidates.sort(key=lambda x: x[1], reverse=True)
        if max_candidates is not None:
            candidates = candidates[:max_candidates]

        symbol_list = [symbol for symbol, _ in candidates]
        log_exchange(f"Discovered {len(symbol_list)} futures symbols from Binance.")
        return symbol_list

    def get_spot_symbols(
        self,
        exchange_name: str = "binance",
        quote_currency: str = "USDT",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> List[str]:
        """
        Get all spot trading symbols from exchange with retry logic for transient errors.

        Args:
            exchange_name: Exchange name (default: 'binance')
            quote_currency: Quote currency to filter (default: 'USDT')
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Base delay between retries in seconds (default: 1.0)

        Returns:
            Sorted list of spot trading symbols

        Raises:
            SymbolFetchError: If fetching symbols fails after all retries
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

            for attempt in range(max_retries):
                try:
                    exchange = self.base.exchange_manager.public.connect_to_exchange_with_no_credentials(exchange_name)
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
