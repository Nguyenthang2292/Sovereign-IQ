"""
Data Fetcher Adapter Component

Adapter for OHLCV data fetching operations used by batch scanner.
"""

from typing import Any, Dict, List

from modules.common.core.data_fetcher import DataFetcher
from modules.common.ui.logging import log_error, log_warn


class DataFetcherAdapter:
    """
    Adapter for data fetching operations in batch scanning.

    Provides simplified interface for fetching OHLCV data with
    minimum candle validation and error handling.
    """

    def __init__(self, data_fetcher: DataFetcher, min_candles: int = 20):
        """
        Initialize DataFetcherAdapter.

        Args:
            data_fetcher: DataFetcher instance to use for fetching
            min_candles: Minimum number of candles required (default: 20)
        """
        self.data_fetcher = data_fetcher
        self.min_candles = min_candles

    def fetch_batch_data(self, symbols: List[str], timeframe: str, limit: int) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV data for a batch of symbols.

        Args:
            symbols: List of symbols to fetch
            timeframe: Timeframe string (e.g., '1h', '4h')
            limit: Number of candles to fetch

        Returns:
            List of dicts with 'symbol' and 'df' keys.
            Only includes symbols with sufficient data (>= min_candles).
        """
        symbols_data = []

        for symbol in symbols:
            try:
                df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
                    symbol=symbol, timeframe=timeframe, limit=limit, check_freshness=False
                )

                if df is not None and not df.empty and len(df) >= self.min_candles:
                    symbols_data.append({"symbol": symbol, "df": df})
                else:
                    log_warn(f"Insufficient data for {symbol}, skipping...")

            except Exception as e:
                log_error(f"Error fetching data for {symbol}: {e}")
                continue

        return symbols_data
