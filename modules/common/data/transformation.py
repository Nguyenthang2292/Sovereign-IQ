"""
Data transformation utilities for DataFrames and Series.

This module provides transformation functions for:
- Converting DataFrames to Series
- Normalizing DataFrame structure
"""

from typing import Optional

import pandas as pd


def dataframe_to_close_series(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """
    Converts a fetched OHLCV DataFrame into a pandas Series of closing prices indexed by timestamp.

    Args:
        df: OHLCV DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    Returns:
        pandas Series of closing prices indexed by timestamp, or None if input is invalid
    """
    if df is None or df.empty:
        return None
    if "timestamp" not in df.columns or "close" not in df.columns:
        return None
    series = df.set_index("timestamp")["close"].copy()
    series.name = "close"
    return series


def _normalize_dataframe_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has 'timestamp' column.

    If timestamp is in index (DatetimeIndex), reset it to column.

    Args:
        df: DataFrame to normalize

    Returns:
        DataFrame with 'timestamp' column
    """
    if "timestamp" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    return df


__all__ = ["dataframe_to_close_series", "_normalize_dataframe_timestamp"]
