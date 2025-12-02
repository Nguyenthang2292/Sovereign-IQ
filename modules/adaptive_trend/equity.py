"""Equity calculations for Adaptive Trend Classification (ATC).

This module provides functions to calculate equity curves based on trading
signals and returns. The equity curve simulates performance of a trading
strategy using exponential growth factors and decay rates.

Performance optimization:
- Uses Numba JIT compilation for the core equity calculation loop
- Replaces pd.NA with np.nan for better compatibility with float64 dtype
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:
    # Fallback if numba is not installed
    def njit(*args, **kwargs):
        """No-op decorator when numba is not available."""
        def decorator(func):
            return func
        return decorator

from .utils import exp_growth


@njit(cache=True)
def _calculate_equity_core(
    r_values: np.ndarray,
    sig_prev_values: np.ndarray,
    starting_equity: float,
    decay_multiplier: float,
    cutout: int,
) -> np.ndarray:
    """Core equity calculation function optimized with Numba.

    This function performs the recursive equity calculation in a compiled loop
    for maximum performance. It handles NaN values and applies the equity floor.

    Args:
        r_values: Array of adjusted returns (R * growth_factor).
        sig_prev_values: Array of previous period signals (shifted by 1).
        starting_equity: Initial equity value.
        decay_multiplier: Decay multiplier (1.0 - De).
        cutout: Number of bars to skip at beginning.

    Returns:
        Array of equity values. Values before cutout are np.nan.
    """
    n = len(r_values)
    e_values = np.full(n, np.nan, dtype=np.float64)

    # Use np.nan to indicate "not initialized" instead of None (Numba doesn't support Optional)
    prev_e = np.nan

    for i in range(n):
        if i < cutout:
            e_values[i] = np.nan
            continue

        r_i = r_values[i]
        sig_prev = sig_prev_values[i]

        # Handle NaN in signal or return
        if np.isnan(sig_prev) or np.isnan(r_i):
            a = 0.0
        elif sig_prev == 0:
            a = 0.0
        elif sig_prev > 0:
            a = r_i
        else:  # sig_prev < 0
            a = -r_i

        # Calculate current equity
        if np.isnan(prev_e):
            e_curr = starting_equity
        else:
            e_curr = (prev_e * decay_multiplier) * (1.0 + a)

        # Apply floor
        if e_curr < 0.25:
            e_curr = 0.25

        prev_e = e_curr
        e_values[i] = e_curr

    return e_values


def equity_series(
    starting_equity: float,
    sig: pd.Series,
    R: pd.Series,
    *,
    L: float,
    De: float,
    cutout: int = 0,
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

    Returns:
        Equity curve Series with same index as sig. Values before cutout
        are np.nan (not pd.NA), minimum value is 0.25.
    """
    if sig is None or R is None or len(sig) == 0:
        return pd.Series(dtype="float64")

    index = sig.index
    # R nhân với e(L) (growth factor)
    growth = exp_growth(L=L, index=index, cutout=cutout)
    r = R * growth
    d = 1.0 - De

    # Shift signals by 1 period (sig[1] in Pine Script)
    sig_shifted = sig.shift(1)

    # Convert to numpy arrays for Numba
    r_values = r.values
    sig_prev_values = sig_shifted.values

    # Calculate equity using optimized Numba function
    e_values = _calculate_equity_core(
        r_values=r_values,
        sig_prev_values=sig_prev_values,
        starting_equity=starting_equity,
        decay_multiplier=d,
        cutout=cutout,
    )

    # Create Series with np.nan (not pd.NA) for float64 compatibility
    equity = pd.Series(e_values, index=index, dtype="float64")
    return equity


__all__ = [
    "equity_series",
]

