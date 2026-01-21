"""
Data validation utilities for OHLCV DataFrames and price series.

This module provides validation functions for:
- OHLCV DataFrame structure validation
- Price series validation (high, low, close)
- Symbol and timeframe validation
"""

from typing import List

import pandas as pd

from modules.common.ui.logging import log_warn

# Constants
OHLCV_REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


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
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")


def validate_price_series(high: pd.Series, low: pd.Series, close: pd.Series) -> None:
    """
    Validate high, low, close price series have correct types and alignment.

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Raises:
        TypeError: If any of the inputs is not a pandas Series
        ValueError: If any series is empty or indices don't match
    """
    if not isinstance(high, pd.Series) or not isinstance(low, pd.Series) or not isinstance(close, pd.Series):
        raise TypeError("high, low, and close must be pandas Series")

    if high.empty or low.empty or close.empty:
        raise ValueError("high, low, and close series cannot be empty")

    if not high.index.equals(low.index) or not low.index.equals(close.index):
        raise ValueError("high, low, and close must have the same index")


def _validate_symbol_and_timeframe(symbol: str, timeframe: str) -> bool:
    """
    Validate symbol and timeframe are non-empty strings.

    Args:
        symbol: Symbol to validate
        timeframe: Timeframe to validate

    Returns:
        True if both are valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        log_warn(f"Skipping invalid symbol: {symbol}")
        return False
    if not timeframe or not isinstance(timeframe, str):
        log_warn(f"Skipping invalid timeframe '{timeframe}' for symbol '{symbol}'")
        return False
    return True


__all__ = ["validate_ohlcv_input", "validate_price_series", "OHLCV_REQUIRED_COLUMNS", "_validate_symbol_and_timeframe"]
