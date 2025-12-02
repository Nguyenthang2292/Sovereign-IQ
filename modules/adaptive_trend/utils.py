"""Utility functions for Adaptive Trend Classification (ATC).

This module provides core utility functions used throughout the ATC system:
- rate_of_change: Calculate percentage price change
- diflen: Calculate length offsets for Moving Averages based on robustness
- exp_growth: Calculate exponential growth factor over time
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def rate_of_change(prices: pd.Series) -> pd.Series:
    """Calculate percentage rate of change for price series.

    Equivalent to Pine Script global variable:
        R = (close - close[1]) / close[1]

    Args:
        prices: Price series (typically close prices).

    Returns:
        Series containing percentage change values. First value will be NaN.
    """
    return prices.pct_change()


def diflen(length: int, robustness: str = "Medium") -> Tuple[int, int, int, int, int, int, int, int]:
    """Calculate length offsets for Moving Averages based on robustness setting.

    Port of Pine Script `diflen(length)` function. Returns 8 length values
    (4 positive offsets and 4 negative offsets) based on the robustness parameter.

    Args:
        length: Base length for Moving Average.
        robustness: Robustness setting determining offset spread:
            - "Narrow": Small offsets (±1, ±2, ±3, ±4)
            - "Medium": Medium offsets (±1, ±2, ±4, ±6)
            - "Wide": Large offsets (±1, ±3, ±5, ±7)

    Returns:
        Tuple of 8 integers: (L1, L2, L3, L4, L_1, L_2, L_3, L_4)
        where L1-L4 are positive offsets and L_1-L_4 are negative offsets.
    """
    robustness = robustness or "Medium"

    if robustness == "Narrow":
        L1, L_1 = length + 1, length - 1
        L2, L_2 = length + 2, length - 2
        L3, L_3 = length + 3, length - 3
        L4, L_4 = length + 4, length - 4
    elif robustness == "Medium":
        L1, L_1 = length + 1, length - 1
        L2, L_2 = length + 2, length - 2
        L3, L_3 = length + 4, length - 4
        L4, L_4 = length + 6, length - 6
    else:  # "Wide" hoặc bất kỳ giá trị khác
        L1, L_1 = length + 1, length - 1
        L2, L_2 = length + 3, length - 3
        L3, L_3 = length + 5, length - 5
        L4, L_4 = length + 7, length - 7

    return L1, L2, L3, L4, L_1, L_2, L_3, L_4


def exp_growth(
    L: float,
    index: Optional[pd.Index] = None,
    *,
    cutout: int = 0,
) -> pd.Series:
    """Calculate exponential growth factor over time.

    Port of Pine Script function:
        e(L) =>
            bars = bar_index == 0 ? 1 : bar_index
            x = 1.0
            if time >= cuttime
                x := math.pow(math.e, L * (bar_index - cutout))
            x

    In TradingView, `time` and `bar_index` are global environment variables.
    Here we approximate using positional indices (0, 1, 2, ...) of the Series.

    Args:
        L: Lambda (growth rate parameter).
        index: Time/bar index of the data. If None, creates empty RangeIndex.
        cutout: Number of bars to skip at the beginning (bars before cutout
            will have value 1.0).

    Returns:
        Series containing exponential growth factors e^(L * (bar_index - cutout))
        for bars >= cutout, and 1.0 for bars < cutout.
    """
    if index is None:
        index = pd.RangeIndex(0, 0)

    # Sử dụng vị trí 0..n-1 làm tương đương `bar_index`
    bars = pd.Series(range(len(index)), index=index, dtype="float64")
    # Trong Pine: nếu bar_index == 0 thì bars = 1, còn lại = bar_index
    bars = bars.where(bars != 0, 1.0)

    # Điều kiện "đã qua cutout"
    active = bars >= cutout
    x = pd.Series(1.0, index=index, dtype="float64")
    # Calculate exponential growth for active bars
    if active.any():
        x.loc[active] = (np.e ** (L * (bars[active] - cutout))).astype("float64")
    return x


__all__ = [
    "rate_of_change",
    "diflen",
    "exp_growth",
]

