"""
Main equity series calculation function.

This module provides the high-level equity_series function that calculates
equity curves from trading signals and returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS_mini.core.compute_equity.core import _calculate_equity_core, get_equity_floor
from modules.adaptive_trend_LTS_mini.utils.cache_manager import get_cache_manager
from modules.adaptive_trend_LTS_mini.utils.exp_growth import exp_growth
from modules.common.system import get_series_pool
from modules.common.utils import log_error, log_warn

__all__ = ["equity_series"]


def equity_series(
    starting_equity: float,
    sig: pd.Series,
    rate_of_change_series: pd.Series,
    *,
    lambda_val: float,
    decay_val: float,
    verbose: bool = False,
) -> pd.Series:
    """Calculate equity curve from trading signals and returns.

    Port of Pine Script function:
        eq(starting_equity, sig, rate_of_change) =>
            r = rate_of_change * e(lambda_val)  # Adjusted return with growth factor
            d = 1 - decay_val     # Decay multiplier
            var float a = 0.0
            if (sig[1] > 0)
                a := r      # Long position
            else if (sig[1] < 0)
                a := -r     # Short position
            var float e = na
            if na(e[1])
                e := starting_equity
            else
                e := (e[1] * d) * (1 + a)  # Apply decay and return
            if (e < 0.25)
                e := 0.25   # Floor at 0.25
            e

    Simulates equity curve evolution:
    - Long signals (sig > 0): Add adjusted return
    - Short signals (sig < 0): Subtract adjusted return
    - No signal (sig == 0): No change
    - Applies decay factor each period
    - Minimum equity floor at 0.25

    Performance:
    - Uses Numba JIT compilation for the core calculation loop
    - Optimized for large datasets (millions of rows)

    Args:
        starting_equity: Initial equity value.
        sig: Signal series with values {-1, 0, 1}:
            - 1: Long position
            - -1: Short position
            - 0: No position
        rate_of_change_series: Rate of change series (percentage change).
        lambda_val: Lambda (growth rate) for exponential growth factor.
        decay_val: Decay factor (0-1), applied each period.
        cutout: Number of bars to skip at beginning (returns NaN for these bars).
            Values before cutout are set to np.nan for proper handling in
            statistical calculations and plotting (use dropna() if needed).
        verbose: If True, log warnings about NaNs and floor hits. Default is False.

    Returns:
        Equity curve Series with same index as sig. Values before cutout
        are np.nan (not pd.NA), minimum value is 0.25.

    Raises:
        ValueError: If input parameters are invalid.
        TypeError: If input types are incorrect.
    """
    # Input validation
    if sig is None or rate_of_change_series is None:
        raise ValueError("sig and rate_of_change_series cannot be None")

    if len(sig) == 0 or len(rate_of_change_series) == 0:
        if verbose:
            log_warn("Empty input series provided, returning empty equity series")
        return pd.Series(dtype="float64")

    if not isinstance(sig, pd.Series):
        raise TypeError(f"sig must be a pandas Series, got {type(sig)}")

    if not isinstance(rate_of_change_series, pd.Series):
        raise TypeError(f"rate_of_change_series must be a pandas Series, got {type(rate_of_change_series)}")

    if starting_equity <= 0:
        raise ValueError(f"starting_equity must be > 0, got {starting_equity}")

    if not (0 <= decay_val <= 1):
        raise ValueError(f"decay_val must be between 0 and 1, got {decay_val}")

    if not isinstance(lambda_val, (int, float)) or np.isnan(lambda_val) or np.isinf(lambda_val):
        raise ValueError(f"lambda_val must be a finite number, got {lambda_val}")

    # NOTE: cutout is always 0 now as slicing happens early in compute_atc_signals
    # If equity_series is called directly with cutout > 0, warn user
    cutout = 0
    if verbose and cutout > 0:
        log_warn(
            f"cutout parameter ({cutout}) is ignored in equity_series. "
            f"Cutout should be applied at compute_atc_signals level."
        )

    # Check index compatibility
    if not sig.index.equals(rate_of_change_series.index):
        if verbose:
            log_warn(
                f"sig and rate_of_change_series have different indices. "
                f"sig length: {len(sig)}, rate_of_change_series length: {len(rate_of_change_series)}. "
                f"Attempting to align indices."
            )
        # Align indices by taking intersection
        common_index = sig.index.intersection(rate_of_change_series.index)
        if len(common_index) == 0:
            raise ValueError("sig and rate_of_change_series have no common indices")
        sig = sig.loc[common_index]
        rate_of_change_series = rate_of_change_series.loc[common_index]

    index = sig.index

    # Check for excessive NaN values
    sig_nan_count = sig.isna().sum()
    r_nan_count = rate_of_change_series.isna().sum()
    total_bars = len(sig)

    if verbose and sig_nan_count > 0:
        nan_pct = (sig_nan_count / total_bars) * 100
        log_warn(
            f"Signal series contains {sig_nan_count} NaN values ({nan_pct:.1f}%). "
            f"These will be treated as no position (0)."
        )

    if verbose and r_nan_count > 0:
        nan_pct = (r_nan_count / total_bars) * 100
        log_warn(
            f"Return series contains {r_nan_count} NaN values ({nan_pct:.1f}%). These will be treated as zero return."
        )

    try:
        # Try to get from cache
        cache = get_cache_manager()
        cached_equity = cache.get_equity(
            signal=sig,
            rate_of_change=rate_of_change_series,
            lambda_val=lambda_val,
            decay_val=decay_val,
            starting_equity=starting_equity,
        )

        if cached_equity is not None:
            return cached_equity

        # rate_of_change multiplied by e(lambda_val) (growth factor)
        growth = exp_growth(lambda_val=lambda_val, index=index, cutout=cutout)
        r = rate_of_change_series * growth
        d = 1.0 - decay_val

        # Shift signals by 1 period (sig[1] in Pine Script)
        # NOTE: This shift is for INTERNAL equity calculation only
        # The returned equity series is NOT shifted - it represents equity at current bar
        sig_shifted = sig.shift(1)

        # Convert to numpy arrays for Numba (ensure ndarray, not ExtensionArray)
        r_values = np.asarray(r.values, dtype=np.float64)
        sig_prev_values = np.asarray(sig_shifted.values, dtype=np.float64)

        # Use SeriesPool to reduce allocation overhead
        # We acquire a series of the correct length and index
        # Then calculate directly into its underlying array
        equity = get_series_pool().acquire(len(index), dtype=np.float64, index=index)
        out_arr: np.ndarray = np.asarray(equity.values, dtype=np.float64)

        # Calculate equity using optimized Numba function, writing directly to out_arr
        # Note: _calculate_equity_core may return a copy if the input array wasn't writable
        result_array = _calculate_equity_core(
            r_values=r_values,
            sig_prev_values=sig_prev_values,
            starting_equity=starting_equity,
            decay_multiplier=d,
            cutout=cutout,
            out=out_arr,
            floor_val=get_equity_floor(),
        )

        # If a copy was made, or we wrote to a buffer other than equity.values, use result
        if result_array is not out_arr:
            equity = pd.Series(result_array, index=index, dtype=np.float64)
        elif out_arr is not equity.values:
            equity = pd.Series(out_arr, index=index, dtype=np.float64)

        # Cache the result
        cache.put_equity(
            signal=sig,
            rate_of_change=rate_of_change_series,
            lambda_val=lambda_val,
            decay_val=decay_val,
            starting_equity=starting_equity,
            equity=equity,
        )

        # Check for floor hits (equity at minimum value)
        if verbose:
            floor_hits = (equity == 0.25).sum()
            if floor_hits > 0:
                floor_pct = (floor_hits / len(equity)) * 100
                log_warn(
                    f"Equity floor (0.25) was hit {floor_hits} times ({floor_pct:.1f}% of bars). "
                    f"This may indicate high drawdown periods."
                )

        return equity

    except Exception as e:
        log_error(f"Error calculating equity series: {e}")
        raise
