"""Crossover detection function for signal generation.

This module provides the crossover function to detect upward crossovers
between two series (equivalent to Pine Script ta.crossover).
"""

from __future__ import annotations

import pandas as pd

from modules.common.utils import log_error, log_warn


def crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Detect upward crossover between two series.

    Equivalent to Pine Script `ta.crossover(a, b)`:
    Returns True when series_a currently > series_b AND series_a[1] <= series_b[1]

    Args:
        series_a: First series (typically price).
        series_b: Second series (typically Moving Average).

    Returns:
        Boolean Series: True at indices where crossover occurs.

    Raises:
        ValueError: If series are empty or have incompatible indices.
        TypeError: If inputs are not pandas Series.
    """
    if not isinstance(series_a, pd.Series):
        raise TypeError(f"series_a must be a pandas Series, got {type(series_a)}")

    if not isinstance(series_b, pd.Series):
        raise TypeError(f"series_b must be a pandas Series, got {type(series_b)}")

    if len(series_a) == 0 or len(series_b) == 0:
        log_warn("Empty series provided for crossover detection, returning empty boolean series")
        return pd.Series(dtype="bool", index=series_a.index if len(series_a) > 0 else series_b.index)

    try:
        # Align indices if needed
        if not series_a.index.equals(series_b.index):
            log_warn("series_a and series_b have different indices. Aligning to common indices.")
            common_index = series_a.index.intersection(series_b.index)
            if len(common_index) == 0:
                log_warn("No common indices found between series_a and series_b")
                return pd.Series(dtype="bool", index=series_a.index)
            series_a = series_a.loc[common_index]
            series_b = series_b.loc[common_index]

        prev_a = series_a.shift(1)
        prev_b = series_b.shift(1)

        # Handle NaN values from shift(1) - first value will be NaN
        # NaN comparisons result in False, which is correct for our logic
        result = (series_a > series_b) & (prev_a <= prev_b)

        # Fill NaN values (from shift) with False
        result = result.fillna(False)

        return result

    except Exception as e:
        log_error(f"Error detecting crossover: {e}")
        raise
