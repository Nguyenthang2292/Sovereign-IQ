"""
HMM Signal Utilities Module

Utility functions for data validation.
"""

import pandas as pd

from modules.common.utils import log_error


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has required OHLCV columns.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        log_error("Invalid DataFrame: empty or None")
        return False

    required_columns = ["open", "high", "low", "close"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        log_error(f"Missing required columns: {missing_columns}")
        return False

    if len(df) < 20:
        log_error(f"Insufficient data: got {len(df)} rows, need at least 20")
        return False

    return True
