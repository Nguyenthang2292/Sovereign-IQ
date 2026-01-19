"""Range Oscillator main calculation.

This module provides the main Range Oscillator calculation function that
orchestrates the entire indicator computation process. It combines weighted MA
and ATR range to produce oscillator values.

The module uses helper functions from common indicators:
- weighted_ma: Weighted moving average calculation (from modules.common.indicators.trend)
- atr_range: ATR-based range bands calculation (from modules.common.indicators.volatility)

Port of Pine Script Range Oscillator (Zeiierman).
Original: https://creativecommons.org/licenses/by-nc-sa/4.0/
© Zeiierman
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from modules.common.indicators.trend import calculate_weighted_ma
from modules.common.indicators.volatility import calculate_atr_range
from modules.range_oscillator.config.heatmap_config import HeatmapConfig
from modules.range_oscillator.core.heatmap import calculate_heat_colors, calculate_trend_direction


def calculate_range_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    length: int = 50,
    mult: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Range Oscillator indicator.

    This is the main function that orchestrates the entire Range Oscillator
    calculation process. It combines weighted MA and ATR range to produce
    a comprehensive oscillator indicator.

    LUỒNG TÍNH TOÁN:
    ----------------
    1. Tính Weighted MA từ close prices
    2. Tính ATR Range từ high/low/close
    3. Vectorized calculations cho tất cả bars:
       a. Tính Oscillator = 100 * (close - MA) / RangeATR (vectorized)

    Port of Pine Script Range Oscillator (Zeiierman).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        length: Minimum range length (default: 50).
        mult: Range width multiplier (default: 2.0).
        levels_inp: Number of heat levels (default: 2).
        heat_thresh: Minimum touches per level (default: 1).

    Returns:
        Tuple containing:
        - oscillator: Oscillator values. Typically ranges from -100 to +100,
          but is unbounded and can exceed these limits when price deviates
          significantly from the moving average. The ±100 bounds represent
          typical values when price deviation is within the ATR range; extreme
          deviations (where |close - ma| > range_atr) will produce values
          beyond ±100 due to the division by range_atr in the formula.
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

    # Step 3: Calculate oscillator (vectorized)
    # Align all series to close.index to ensure consistent indexing
    ma_aligned = ma.reindex(close.index)
    range_atr_aligned = range_atr.reindex(close.index)

    # Create valid mask for all calculations
    valid_mask = range_atr_aligned.notna() & (range_atr_aligned != 0) & ma_aligned.notna() & close.notna()

    # Step 4a: Calculate oscillator value (vectorized)
    oscillator = pd.Series(np.nan, index=close.index, dtype="float64")
    oscillator[valid_mask] = 100 * (close[valid_mask] - ma_aligned[valid_mask]) / range_atr_aligned[valid_mask]

    return oscillator, ma, range_atr


def calculate_range_oscillator_with_heatmap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    length: int = 50,
    mult: float = 2.0,
    heatmap_config: Optional[HeatmapConfig] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Range Oscillator indicator with heatmap colors.

    This function extends `calculate_range_oscillator()` to also calculate
    heatmap colors and trend direction for visualization purposes.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        length: Minimum range length (default: 50).
        mult: Range width multiplier (default: 2.0).
        heatmap_config: Heatmap configuration. If None, uses default config.

    Returns:
        Tuple containing:
        - oscillator: Oscillator values
        - ma: Weighted moving average
        - range_atr: ATR-based range
        - heat_colors: Heatmap colors as hex strings
        - trend_direction: Trend direction (1=bullish, -1=bearish, 0=neutral)
        - osc_colors: Final indicator colors (includes breakout overrides)
    """
    # Calculate base oscillator
    oscillator, ma, range_atr = calculate_range_oscillator(
        high=high,
        low=low,
        close=close,
        length=length,
        mult=mult,
    )

    # Calculate trend direction
    trend_direction = calculate_trend_direction(close, ma)

    # Calculate heatmap colors (already includes transition color on flips)
    heat_colors = calculate_heat_colors(oscillator, trend_direction, config=heatmap_config)

    # Final oscillator colors (apply breakout overrides matching Pine Script)
    if heatmap_config is None:
        heatmap_config = HeatmapConfig()

    osc_colors = heat_colors.copy()

    # breakout logic: breakUp = close > ma + rangeATR, breakDn = close < ma - rangeATR
    break_up = close > (ma + range_atr)
    break_dn = close < (ma - range_atr)

    osc_colors[break_up] = heatmap_config.strong_bullish_color
    osc_colors[break_dn] = heatmap_config.strong_bearish_color

    return oscillator, ma, range_atr, heat_colors, trend_direction, osc_colors


__all__ = [
    "calculate_range_oscillator",
    "calculate_range_oscillator_with_heatmap",
]
