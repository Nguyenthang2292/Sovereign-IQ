"""
Data utilities for DataFrame/Series manipulation.
"""

import pandas as pd
from typing import Optional, List


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


def validate_ohlcv_input(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate that DataFrame has required columns and is not empty.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        ValueError: If DataFrame is empty or missing required columns
    """
    if df.empty:
        raise ValueError(f"DataFrame is empty. Required columns: {required_columns}")
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

