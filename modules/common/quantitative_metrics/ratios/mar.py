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
    
    Args:
        close: Close price series
        length: Period for moving average
        ma_type: Type of moving average ("SMA" or "EMA")
        
    Returns:
        Series with MAR values (close / MA)
    """
    if ma_type == "SMA":
        ma = ta.sma(close, length=length)
    else:  # EMA
        ma = ta.ema(close, length=length)

    if ma is None:
        ma = close.copy()

    mar = close / ma.replace(0, np.nan)
    return mar


__all__ = [
    "calculate_mar",
]

