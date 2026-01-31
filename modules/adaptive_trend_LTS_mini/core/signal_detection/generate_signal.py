"""Signal generation from Moving Average crossovers.

This module provides the generate_signal_from_ma function to generate
discrete trading signals from price/MA crossovers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.core.compute_equity.utils import njit
from modules.common.system import get_series_pool
from modules.common.utils import log_error, log_warn

from .crossover import crossover
from .crossunder import crossunder


@njit(cache=True)
def _apply_signal_persistence(up: np.ndarray, down: np.ndarray, out: np.ndarray) -> None:
    """Apply signal persistence logic (Pine Script 'var' behavior).

    sig = 0
    if up: sig = 1
    elif down: sig = -1
    else: sig = sig[prev]

    DESIGN NOTE - Signal Initialization:
    The initial signal value is 0 (neutral) at the start of the series.
    This is CORRECT behavior matching Pine Script's 'var' keyword:
    - First bar: sig = 0 (neutral)
    - If up[0] is True: sig becomes 1
    - If down[0] is True: sig becomes -1
    - Otherwise: sig stays 0

    Edge case handling:
    - If first bar has down=True: sig becomes -1 (not 0)
    - This is handled correctly because we check conditions every bar
    - No special "initialization bias" - just pure signal persistence

    This matches Pine Script behavior exactly:
    var int sig = 0  # Initialized once to 0
    if crossover: sig := 1
    if crossunder: sig := -1
    """
    n = len(up)
    current_sig = 0
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

        # Get boolean arrays from crossover/crossunder (always return Series)
        up_vals = up.values
        down_vals = down.values

        # Run Numba kernel directly into pooled buffer
        sig_vals = sig_series.values

        # Handle non-writeable arrays (pool might return read-only buffer)
        if not sig_vals.flags.writeable:
            # Copy the array and create new Series (can't use pool)
            sig_vals = sig_vals.copy()
            _apply_signal_persistence(up_vals, down_vals, sig_vals)
            sig_series = pd.Series(sig_vals, index=price.index, dtype=np.int8)
        else:
            # Use pooled buffer directly (optimal path)
            _apply_signal_persistence(up_vals, down_vals, sig_vals)
            # sig_series already has updated values in-place

        return sig_series

    except Exception as e:
        log_error(f"Error generating signal from MA: {e}")
        raise
