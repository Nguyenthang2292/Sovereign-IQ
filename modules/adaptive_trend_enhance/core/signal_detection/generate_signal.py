"""Signal generation from Moving Average crossovers.

This module provides the generate_signal_from_ma function to generate
discrete trading signals from price/MA crossovers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.adaptive_trend_enhance.core.compute_equity.utils import njit
from modules.common.system import get_series_pool
from modules.common.utils import log_error, log_warn


@njit(cache=True)
def _apply_signal_persistence(up: np.ndarray, down: np.ndarray, out: np.ndarray) -> None:
    """Apply signal persistence logic (Pine Script 'var' behavior).

    sig = 0
    if up: sig = 1
    elif down: sig = -1
    else: sig = sig[prev]
    """
    n = len(out)
    current_sig = 0  # Default to 0

    for i in range(n):
        if up[i]:
            current_sig = 1
        elif down[i]:
            current_sig = -1

        out[i] = current_sig


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

    Raises:
        ValueError: If price or ma are empty or have incompatible indices.
        TypeError: If inputs are not pandas Series.
    """
    if not isinstance(price, pd.Series):
        raise TypeError(f"price must be a pandas Series, got {type(price)}")

    if not isinstance(ma, pd.Series):
        raise TypeError(f"ma must be a pandas Series, got {type(ma)}")

    if len(price) == 0 or len(ma) == 0:
        log_warn("Empty price or MA series provided, returning empty signal series")
        return pd.Series(dtype="int8", index=price.index if len(price) > 0 else ma.index)

    try:
        # Align indices if needed
        if not price.index.equals(ma.index):
            log_warn("price and ma have different indices. Aligning to common indices.")
            common_index = price.index.intersection(ma.index)
            if len(common_index) == 0:
                log_warn("No common indices found between price and ma")
                return pd.Series(dtype="int8", index=price.index)
            price = price.loc[common_index]
            ma = ma.loc[common_index]

        # Check for excessive NaN values
        price_nan_count = price.isna().sum()
        ma_nan_count = ma.isna().sum()
        total_bars = len(price)

        if price_nan_count > 0:
            nan_pct = (price_nan_count / total_bars) * 100
            if nan_pct > 10:  # Warn if more than 10% NaN
                log_warn(
                    f"Price series contains {price_nan_count} NaN values ({nan_pct:.1f}%). "
                    f"This may affect signal generation."
                )

        if ma_nan_count > 0:
            nan_pct = (ma_nan_count / total_bars) * 100
            if nan_pct > 10:  # Warn if more than 10% NaN
                log_warn(
                    f"MA series contains {ma_nan_count} NaN values ({nan_pct:.1f}%). This may affect signal generation."
                )

        # Optimization: Use Numba for persistence instead of expensive Pandas chain
        # to avoid 4x allocation overhead.

        # Calculate boolean arrays (crossover/crossunder)
        up = crossover(price, ma)
        down = crossunder(price, ma)

        # Acquire pooled series for result
        sig_series = get_series_pool().acquire(len(price), dtype=np.int8, index=price.index)

        # Get boolean arrays (crossover/crossunder return Series or bool array)
        if isinstance(up, pd.Series):
            up_vals = up.values
        else:
            up_vals = up

        if isinstance(down, pd.Series):
            down_vals = down.values
        else:
            down_vals = down

        # Run Numba kernel directly into pooled buffer
        _apply_signal_persistence(up_vals, down_vals, sig_series.values)

        return sig_series

    except Exception as e:
        log_error(f"Error generating signal from MA: {e}")
        raise
