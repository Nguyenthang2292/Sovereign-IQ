"""Signal generation functions for Adaptive Trend Classification (ATC).

This module provides functions to generate trading signals from Moving Averages:
- crossover: Detect upward crossover (series_a crosses above series_b)
- crossunder: Detect downward crossover (series_a crosses below series_b)
- generate_signal_from_ma: Generate discrete signals {-1, 0, 1} from price/MA crossovers
"""

from __future__ import annotations

import pandas as pd


def crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Detect upward crossover between two series.

    Equivalent to Pine Script `ta.crossover(a, b)`:
    Returns True when series_a currently > series_b AND series_a[1] <= series_b[1]

    Args:
        series_a: First series (typically price).
        series_b: Second series (typically Moving Average).

    Returns:
        Boolean Series: True at indices where crossover occurs.
    """
    prev_a = series_a.shift(1)
    prev_b = series_b.shift(1)
    return (series_a > series_b) & (prev_a <= prev_b)


def crossunder(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Detect downward crossover between two series.

    Equivalent to Pine Script `ta.crossunder(a, b)`:
    Returns True when series_a currently < series_b AND series_a[1] >= series_b[1]

    Args:
        series_a: First series (typically price).
        series_b: Second series (typically Moving Average).

    Returns:
        Boolean Series: True at indices where crossunder occurs.
    """
    prev_a = series_a.shift(1)
    prev_b = series_b.shift(1)
    return (series_a < series_b) & (prev_a >= prev_b)


def generate_signal_from_ma(
    price: pd.Series,
    ma: pd.Series,
) -> pd.Series:
    """Generate discrete trading signals from price/MA crossovers.

    Port of Pine Script function:
        signal(ma) =>
            var int sig = 0
            if ta.crossover(close, ma)
                sig := 1
            if ta.crossunder(close, ma)
                sig := -1
            sig

    Args:
        price: Price series (typically close prices).
        ma: Moving Average series.

    Returns:
        Series with discrete signal values:
        - 1: Bullish signal (price crosses above MA)
        - -1: Bearish signal (price crosses below MA)
        - 0: No signal (no crossover detected)
    """
    sig = pd.Series(0, index=price.index, dtype="int8")
    up = crossover(price, ma)
    down = crossunder(price, ma)
    sig[up] = 1
    sig[down] = -1
    return sig


__all__ = [
    "crossover",
    "crossunder",
    "generate_signal_from_ma",
]

