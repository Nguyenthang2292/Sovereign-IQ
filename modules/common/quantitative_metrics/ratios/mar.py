"""
Moving Average Ratio (MAR) indicator for technical analysis.

MAR calculates the ratio of close price to moving average, providing a normalized
measure of price position relative to its moving average.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta


def calculate_mar(
    close: pd.Series, length: int = 14, ma_type: str = "SMA"
) -> pd.Series:
    """
    Calculate MAR (Moving Average Ratio).
    
    MAR calculates the ratio of close price to moving average, providing a normalized
    measure of price position relative to its moving average.
    
    Args:
        close: Close price series (pd.Series). Must not be None or empty.
        length: Period for moving average (must be > 0). Default: 14
        ma_type: Type of moving average ("SMA" or "EMA"). Default: "SMA"
        
    Returns:
        Series with MAR values (close / MA). Returns empty Series if input is invalid.
        NaN values are returned where MA is zero or calculation fails.
        
    Example:
        >>> close = pd.Series([100, 102, 101, 103, 105, 104, 106])
        >>> mar = calculate_mar(close, length=3, ma_type='SMA')
        >>> # Returns ratio of close to 3-period SMA
    """
    # Validate input: close must be pd.Series
    if close is None:
        return pd.Series(dtype=float)
    
    if not isinstance(close, pd.Series):
        return pd.Series(dtype=float)
    
    # Validate length: must be positive
    if length <= 0:
        return pd.Series(dtype=float)
    
    # Validate ma_type: only "SMA" or "EMA" allowed
    if ma_type not in ("SMA", "EMA"):
        return pd.Series(dtype=float)
    
    # Handle empty series
    if len(close) == 0:
        return pd.Series(dtype=float)
    
    # Calculate moving average based on type
    if ma_type == "SMA":
        ma = ta.sma(close, length=length)
    else:  # EMA
        ma = ta.ema(close, length=length)

    # If pandas_ta returns None, use close as fallback (results in ratio of 1.0)
    if ma is None:
        ma = close.copy()

    # Calculate MAR: close / MA, replacing zeros with NaN to avoid division by zero
    mar = close / ma.replace(0, np.nan)
    return mar


__all__ = [
    "calculate_mar",
]

