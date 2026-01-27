"""Binance current prices fetching functionality."""

from typing import TYPE_CHECKING

from modules.common.domain import normalize_symbol
from modules.common.ui.logging import log_error, log_exchange, log_success, log_warn
from modules.common.ui.progress_bar import ProgressBar

if TYPE_CHECKING:
    from .base import DataFetcherBase


class BinancePriceFetcher:
    """Handles fetching current prices from Binance."""

    def __init__(self, base: "DataFetcherBase"):
        """
        Initialize BinancePriceFetcher.

        Args:
            base: DataFetcherBase instance for accessing exchange_manager and state
        """
        self.base = base

    def fetch_current_prices_from_binance(self, symbols: list):
        """
        Fetches current prices for all symbols from Binance.

        Args:
            symbols: List of trading symbols to fetch prices for

        Returns:
            None. Updates self.base.market_prices dict with fetched prices.
        """
        if not symbols:
            return

        log_exchange("Fetching current prices from Binance...")
        progress = ProgressBar(len(symbols), "Price Fetch")

        try:
            # Use authenticated manager for authenticated calls
            exchange = self.base.exchange_manager.authenticated.connect_to_binance_with_credentials()
        except ValueError as e:
            log_error(f"Error: {e}")
            return

        fetched_count = 0
        failed_symbols = []

        for symbol in symbols:
            if self.base.should_stop():
                log_warn("Price fetch aborted due to shutdown signal.")
                break
            normalized_symbol = normalize_symbol(symbol)
            try:
                # Use authenticated manager's throttled_call
                ticker = self.base.exchange_manager.authenticated.throttled_call(
                    exchange.fetch_ticker, normalized_symbol
                )
                if ticker and "last" in ticker:
                    self.base.market_prices[symbol] = ticker["last"]
                    fetched_count += 1
                    log_exchange(f"[BINANCE] {normalized_symbol}: {ticker['last']:.8f}")
                else:
                    failed_symbols.append(symbol)
                    log_warn(f"{normalized_symbol}: No price data available")
            except Exception as e:
                failed_symbols.append(symbol)
                log_error(f"Error fetching {normalized_symbol}: {e}")
            finally:
                progress.update()
        progress.finish()

        if failed_symbols:
            log_warn(
                f"Warning: Could not fetch prices for {len(failed_symbols)} symbol(s): {', '.join(failed_symbols)}"
            )

        if fetched_count > 0:
            log_success(f"Successfully fetched prices for {fetched_count}/{len(symbols)} symbols")
