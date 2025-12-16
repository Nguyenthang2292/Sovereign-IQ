"""Range Oscillator main calculation.

This module provides the main Range Oscillator calculation function that
orchestrates the entire indicator computation process. It combines weighted MA,
ATR range, and trend direction to produce oscillator values with color coding.

The module uses helper functions from common indicators:
- weighted_ma: Weighted moving average calculation (from modules.common.indicators.trend)
- atr_range: ATR-based range bands calculation (from modules.common.indicators.volatility)
- trend_direction: Trend direction determination (from modules.common.indicators.trend)

Port of Pine Script Range Oscillator (Zeiierman).
Original: https://creativecommons.org/licenses/by-nc-sa/4.0/
© Zeiierman
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from modules.common.indicators.trend import calculate_weighted_ma, calculate_trend_direction
from modules.common.indicators.volatility import calculate_atr_range


def calculate_range_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    length: int = 50,
    mult: float = 2.0,
    levels_inp: int = 2,
    heat_thresh: int = 1,
    strong_bullish_color: str = "#09ff00",
    strong_bearish_color: str = "#ff0000",
    weak_bearish_color: str = "#800000",
    weak_bullish_color: str = "#008000",
    transition_color: str = "#0000ff",
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Range Oscillator indicator.

    This is the main function that orchestrates the entire Range Oscillator
    calculation process. It combines weighted MA, ATR range, trend direction,
    and heatmap colors to produce a comprehensive oscillator indicator.

    LUỒNG TÍNH TOÁN:
    ----------------
    1. Tính Weighted MA từ close prices
    2. Tính ATR Range từ high/low/close
    3. Xác định Trend Direction (bullish/bearish)
    4. Vectorized calculations cho tất cả bars:
       a. Tính Oscillator = 100 * (close - MA) / RangeATR (vectorized)
       b. Kiểm tra breakouts (upper/lower bounds) (vectorized)
       c. Xác định trend flips (vectorized)
       d. Xác định final color với priority: breakout > transition > trend-based (vectorized)

    Port of Pine Script Range Oscillator (Zeiierman).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        length: Minimum range length (default: 50).
        mult: Range width multiplier (default: 2.0).
        levels_inp: Number of heat levels (default: 2).
        heat_thresh: Minimum touches per level (default: 1).
        strong_bullish_color: Hex color for strong bullish zones.
        strong_bearish_color: Hex color for strong bearish zones.
        weak_bearish_color: Hex color for weak bearish zones.
        weak_bullish_color: Hex color for weak bullish zones.
        transition_color: Hex color for transitions.

    Returns:
        Tuple containing:
        - oscillator: Oscillator values (ranges from -100 to +100)
        - oscillator_color: Hex color strings for each oscillator value
        - ma: Weighted moving average
        - range_atr: ATR-based range
    """
    # Input validation
    if not isinstance(high, pd.Series) or not isinstance(low, pd.Series) or not isinstance(close, pd.Series):
        raise TypeError("high, low, and close must be pandas Series")
    if len(high) == 0 or len(low) == 0 or len(close) == 0:
        raise ValueError("high, low, and close series cannot be empty")
    if not high.index.equals(low.index) or not low.index.equals(close.index):
        raise ValueError("high, low, and close must have the same index")
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")
    if mult <= 0:
        raise ValueError(f"mult must be > 0, got {mult}")
    
    # Step 1: Calculate weighted MA
    ma = calculate_weighted_ma(close, length=length)

    # Step 2: Calculate ATR range
    range_atr = calculate_atr_range(high, low, close, mult=mult)

    # Step 3: Calculate trend direction
    trend_dir = calculate_trend_direction(close, ma)

    # Step 4: Calculate oscillator and colors (vectorized)
    # Align all series to close.index to ensure consistent indexing
    ma_aligned = ma.reindex(close.index)
    range_atr_aligned = range_atr.reindex(close.index)
    trend_dir_aligned = trend_dir.reindex(close.index)
    
    # Create valid mask for all calculations
    valid_mask = (
        range_atr_aligned.notna() & 
        (range_atr_aligned != 0) & 
        ma_aligned.notna() & 
        close.notna()
    )
    
    # Step 4a: Calculate oscillator value (vectorized)
    oscillator = pd.Series(np.nan, index=close.index, dtype="float64")
    oscillator[valid_mask] = 100 * (close[valid_mask] - ma_aligned[valid_mask]) / range_atr_aligned[valid_mask]
    
    # Step 4b: Calculate trend flip detection (vectorized)
    prev_trend_dir = trend_dir_aligned.shift(1).fillna(0)
    trend_flip = (trend_dir_aligned != prev_trend_dir) & valid_mask
    
    # Step 4c: Check for breakouts (vectorized)
    break_up = (close > ma_aligned + range_atr_aligned) & valid_mask
    break_dn = (close < ma_aligned - range_atr_aligned) & valid_mask
    
    # Step 4d: Assign colors with priority: breakout > transition > trend-based (vectorized)
    oscillator_color = pd.Series(None, index=close.index, dtype="object")
    
    # Priority 1: Breakouts
    oscillator_color[break_up] = strong_bullish_color
    oscillator_color[break_dn] = strong_bearish_color
    
    # Priority 2: Trend flips (only where not already set by breakouts)
    trend_flip_mask = trend_flip & ~break_up & ~break_dn
    oscillator_color[trend_flip_mask] = transition_color
    
    # Priority 3: Trend-based colors (only where not set by above)
    remaining_mask = valid_mask & ~break_up & ~break_dn & ~trend_flip
    bullish_trend_mask = remaining_mask & (trend_dir_aligned == 1)
    bearish_trend_mask = remaining_mask & (trend_dir_aligned != 1)
    oscillator_color[bullish_trend_mask] = weak_bullish_color
    oscillator_color[bearish_trend_mask] = weak_bearish_color

    return oscillator, oscillator_color, ma, range_atr


__all__ = [
    "calculate_range_oscillator",
]
