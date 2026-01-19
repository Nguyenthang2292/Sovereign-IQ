from typing import Optional, Tuple

import pandas as pd

from modules.common.indicators.trend import calculate_weighted_ma
from modules.common.indicators.volatility import calculate_atr_range
from modules.range_oscillator.config.heatmap_config import HeatmapConfig
from modules.range_oscillator.core.oscillator import (
    calculate_range_oscillator,
    calculate_range_oscillator_with_heatmap,
)

"""
Oscillator data utility functions.

This module provides helper functions for oscillator data handling.
"""


def get_oscillator_data(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Helper function to get oscillator data, either from pre-calculated values or by calculating.

    This function avoids redundant calculations when oscillator data is already available.

    Args:
        high: High price series (required if oscillator not provided)
        low: Low price series (required if oscillator not provided)
        close: Close price series (required if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50)
        mult: Range width multiplier (default: 2.0)

    Returns:
        Tuple containing (oscillator, ma, range_atr)
    """
    if oscillator is not None and ma is not None and range_atr is not None:
        # Use pre-calculated values
        # Validate empty data
        if oscillator.empty or ma.empty or range_atr.empty:
            raise ValueError("oscillator, ma, and range_atr cannot be empty")

        # Validate that series contain at least some valid (non-NaN) data
        if not oscillator.notna().any():
            raise ValueError("oscillator contains only NaN values")
        if not ma.notna().any():
            raise ValueError("ma contains only NaN values")
        if not range_atr.notna().any():
            raise ValueError("range_atr contains only NaN values")

        # Validate index alignment
        if not oscillator.index.equals(ma.index) or not ma.index.equals(range_atr.index):
            raise ValueError("oscillator, ma, and range_atr must have the same index")

        return oscillator, ma, range_atr

    # Handle partial pre-calculated values
    if oscillator is not None and (ma is None or range_atr is None):
        # Use provided oscillator, calculate missing ma and/or range_atr
        if high is None or low is None or close is None:
            raise ValueError(
                "high, low, and close are required when oscillator is provided but ma or range_atr is missing"
            )

        # Validate input data
        if high.empty or low.empty or close.empty:
            raise ValueError("high, low, and close cannot be empty")

        if not high.index.equals(low.index) or not low.index.equals(close.index):
            raise ValueError("high, low, and close must have the same index")

        if not oscillator.index.equals(close.index):
            raise ValueError("oscillator must have the same index as close")

        # Calculate missing components
        if ma is None:
            ma = calculate_weighted_ma(close, length=length)

        if range_atr is None:
            range_atr = calculate_atr_range(high, low, close, mult=mult)

        # Validate calculated values
        if ma.empty or range_atr.empty:
            raise ValueError("Calculated ma or range_atr is empty")

        # Note: ma and range_atr may have NaN values at the beginning when data is short
        # This is acceptable behavior - we only validate that they're not ALL NaN
        # But for very short data (< length), ma will be all NaN, which is expected
        if len(close) < length:
            # For short data, allow all NaN in ma (it's expected)
            # But range_atr should still have some values
            if not range_atr.notna().any():
                raise ValueError("Calculated range_atr contains only NaN values")
        else:
            # For sufficient data, validate both
            if not ma.notna().any():
                raise ValueError("Calculated ma contains only NaN values")
            if not range_atr.notna().any():
                raise ValueError("Calculated range_atr contains only NaN values")

        # Validate index alignment
        if not oscillator.index.equals(ma.index) or not ma.index.equals(range_atr.index):
            raise ValueError("oscillator, ma, and range_atr must have the same index")

        return oscillator, ma, range_atr

    elif high is not None and low is not None and close is not None:
        # Calculate oscillator from scratch
        # Validate input data
        if high.empty or low.empty or close.empty:
            raise ValueError("high, low, and close cannot be empty")

        # Validate that series contain at least some valid (non-NaN) data
        if not high.notna().any():
            raise ValueError("high contains only NaN values")
        if not low.notna().any():
            raise ValueError("low contains only NaN values")
        if not close.notna().any():
            raise ValueError("close contains only NaN values")

        # Validate index alignment
        # DEBUG POINT: Input data index alignment check (before calculation)
        # Check: high_index_len, low_index_len, close_index_len, index.equals()
        if not high.index.equals(low.index) or not low.index.equals(close.index):
            raise ValueError("high, low, and close must have the same index")

        oscillator, ma, range_atr = calculate_range_oscillator(
            high=high,
            low=low,
            close=close,
            length=length,
            mult=mult,
        )

        # Validate empty data
        if oscillator.empty or ma.empty or range_atr.empty:
            raise ValueError("Calculated oscillator, ma, or range_atr is empty")

        # Validate that calculated series contain at least some valid (non-NaN) data
        # Note: It's normal for oscillator to have some NaN values (at the beginning),
        # but ma and range_atr should have valid values after the initial period
        if not ma.notna().any():
            raise ValueError("Calculated ma contains only NaN values")
        if not range_atr.notna().any():
            raise ValueError("Calculated range_atr contains only NaN values")

        # Validate index alignment
        # DEBUG POINT: Calculated data index alignment check (after calculation)
        # Check: osc_index_len, ma_index_len, range_atr_index_len, index.equals()
        # Check: osc_notna_count, ma_notna_count, range_atr_notna_count
        if not oscillator.index.equals(ma.index) or not ma.index.equals(range_atr.index):
            raise ValueError("Calculated oscillator, ma, and range_atr must have the same index")

        return oscillator, ma, range_atr
    else:
        raise ValueError("Either provide (oscillator, ma, range_atr) or (high, low, close) with length and mult")


def get_oscillator_data_with_heatmap(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
    heatmap_config: Optional[HeatmapConfig] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Helper function to get oscillator data with heatmap colors.

    This function extends `get_oscillator_data()` to also calculate heatmap colors
    and trend direction for visualization purposes.

    Args:
        high: High price series (required if oscillator not provided)
        low: Low price series (required if oscillator not provided)
        close: Close price series (required if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50)
        mult: Range width multiplier (default: 2.0)
        heatmap_config: Heatmap configuration. If None, uses default config.

    Returns:
        Tuple containing:
        - oscillator: Oscillator values
        - ma: Weighted moving average
        - range_atr: ATR-based range
        - heat_colors: Heatmap colors as hex strings
        - trend_direction: Trend direction (1=bullish, -1=bearish, 0=neutral)
    """
    # Get base oscillator data
    oscillator, ma, range_atr = get_oscillator_data(
        high=high,
        low=low,
        close=close,
        oscillator=oscillator,
        ma=ma,
        range_atr=range_atr,
        length=length,
        mult=mult,
    )

    # Ensure we have close and ma for trend calculation
    if close is None:
        raise ValueError("close is required for heatmap calculation")

    # Calculate base oscillator and heatmap data
    # Use the comprehensive function to ensure consistent color logic (breakouts, flips, etc.)
    oscillator, ma, range_atr, heat_colors, trend_direction, osc_colors = calculate_range_oscillator_with_heatmap(
        high=high, low=low, close=close, length=length, mult=mult, heatmap_config=heatmap_config
    )

    return oscillator, ma, range_atr, heat_colors, trend_direction, osc_colors


__all__ = [
    "get_oscillator_data",
    "get_oscillator_data_with_heatmap",
]
