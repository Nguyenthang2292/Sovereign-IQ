"""
Data Provider for ATC Analysis.

This module handles fetching and validating OHLCV data for symbol analysis.
"""

from typing import TYPE_CHECKING, Optional, Tuple

import pandas as pd

from modules.common.utils import log_error

if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher

__all__ = ["DataProvider"]


class DataProvider:
    """
    Handles data fetching and validation for symbol analysis.

    Responsible for:
    - Fetching OHLCV data from exchanges with fallback
    - Validating data availability and freshness
    - Extracting exchange information
    """

    def __init__(self, data_fetcher: "DataFetcher"):
        """
        Initialize DataProvider.

        Args:
            data_fetcher: DataFetcher instance for market data
        """
        self.data_fetcher = data_fetcher

    def fetch_symbol_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> Optional[Tuple[pd.DataFrame, str]]:
        """
        Fetch OHLCV data for a symbol with exchange fallback.

        Args:
            symbol: Symbol to fetch data for
            timeframe: Timeframe for the data
            limit: Number of candles to fetch

        Returns:
            Tuple of (DataFrame, exchange_label) if successful, None if failed
            The DataFrame contains OHLCV data
            The exchange_label is the uppercase exchange identifier
        """
        try:
            # Fetch OHLCV data with fallback
            fetch_res = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=limit,
                timeframe=timeframe,
                check_freshness=True,
            )
            df: Optional[pd.DataFrame] = fetch_res[0]
            exchange_id: Optional[str] = fetch_res[1]

            # Validate data availability
            if df is None or df.empty:
                log_error(f"No data available for {symbol}")
                return None

            # Format exchange label
            exchange_label: str = exchange_id.upper() if exchange_id else "UNKNOWN"

            return df, exchange_label

        except Exception as e:
            log_error(f"Error fetching data for {symbol}: {type(e).__name__}: {e}")
            return None
