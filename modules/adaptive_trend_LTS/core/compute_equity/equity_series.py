"""
Main equity series calculation function.

This module provides the high-level equity_series function that calculates
equity curves from trading signals and returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.utils.cache_manager import get_cache_manager
from modules.adaptive_trend_LTS.utils.exp_growth import exp_growth
from modules.common.system import get_series_pool
from modules.common.utils import log_error, log_warn

from .core import _calculate_equity_core

__all__ = ["equity_series"]


def equity_series(
    starting_equity: float,
    sig: pd.Series,
    R: pd.Series,
    *,
    L: float,
    De: float,
    verbose: bool = False,
) -> pd.Series:
    """Calculate equity curve from trading signals and returns.

    Port of Pine Script function:
        eq(starting_equity, sig, R) =>
            r = R * e(La)  # Adjusted return with growth factor
            d = 1 - De     # Decay multiplier
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
        R: Rate of change series (percentage change).
        L: Lambda (growth rate) for exponential growth factor.
        De: Decay factor (0-1), applied each period.
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
    if sig is None or R is None:
        raise ValueError("sig and R cannot be None")

    if len(sig) == 0 or len(R) == 0:
        if verbose:
            log_warn("Empty input series provided, returning empty equity series")
        return pd.Series(dtype="float64")

    if not isinstance(sig, pd.Series):
        raise TypeError(f"sig must be a pandas Series, got {type(sig)}")

    if not isinstance(R, pd.Series):
        raise TypeError(f"R must be a pandas Series, got {type(R)}")

    if starting_equity <= 0:
        raise ValueError(f"starting_equity must be > 0, got {starting_equity}")

    if not (0 <= De <= 1):
        raise ValueError(f"De must be between 0 and 1, got {De}")

    if not isinstance(L, (int, float)) or np.isnan(L) or np.isinf(L):
        raise ValueError(f"L must be a finite number, got {L}")

    # cutout is always 0 now as slicing happens early in compute_atc_signals
    cutout = 0

    # Check index compatibility
    if not sig.index.equals(R.index):
        if verbose:
            log_warn(
                f"sig and R have different indices. "
                f"sig length: {len(sig)}, R length: {len(R)}. "
                f"Attempting to align indices."
            )
        # Align indices by taking intersection
        common_index = sig.index.intersection(R.index)
        if len(common_index) == 0:
            raise ValueError("sig and R have no common indices")
        sig = sig.loc[common_index]
        R = R.loc[common_index]

    index = sig.index

    # Check for excessive NaN values
    sig_nan_count = sig.isna().sum()
    r_nan_count = R.isna().sum()
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
        cached_equity = cache.get_equity(signal=sig, R=R, L=L, De=De, starting_equity=starting_equity)

        if cached_equity is not None:
            return cached_equity

        # R multiplied by e(L) (growth factor)
        growth = exp_growth(L=L, index=index, cutout=cutout)
        r = R * growth
        d = 1.0 - De

        # Shift signals by 1 period (sig[1] in Pine Script)
        sig_shifted = sig.shift(1)

        # Convert to numpy arrays for Numba
        r_values = r.values
        sig_prev_values = sig_shifted.values

        # Use SeriesPool to reduce allocation overhead
        # We acquire a series of the correct length and index
        # Then calculate directly into its underlying array
        equity = get_series_pool().acquire(len(index), dtype=np.float64, index=index)

        # Calculate equity using optimized Numba function, writing directly to equity.values
        # Note: _calculate_equity_core may return a copy if the input array wasn't writable
        result_array = _calculate_equity_core(
            r_values=r_values,
            sig_prev_values=sig_prev_values,
            starting_equity=starting_equity,
            decay_multiplier=d,
            cutout=cutout,
            out=equity.values,
        )

        # If a copy was made, update the Series
        if result_array is not equity.values:
            equity = pd.Series(result_array, index=index, dtype=np.float64)

        # Cache the result
        cache.put_equity(signal=sig, R=R, L=L, De=De, starting_equity=starting_equity, equity=equity)

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
