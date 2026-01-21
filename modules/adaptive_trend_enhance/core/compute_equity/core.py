"""
Core equity calculation functions.

This module provides optimized calculation functions for equity curves:
- _calculate_equity_core: Single signal Numba JIT optimized calculation
- _calculate_equity_vectorized: Vectorized calculation for multiple signals
"""

from __future__ import annotations

import numpy as np

from .utils import njit

__all__ = ["_calculate_equity_core", "_calculate_equity_vectorized"]


def _calculate_equity_vectorized(
    starting_equities: np.ndarray,
    sig_prev_values: np.ndarray,
    r_values: np.ndarray,
    decay_multiplier: float,
    cutout: int,
    out: np.ndarray = None,
) -> np.ndarray:
    """Vectorized equity calculation for multiple signals at once.

    This function calculates equity curves for multiple signals simultaneously
    using pure NumPy operations, which is faster than calling _calculate_equity_core
    multiple times in a loop.

    Args:
        starting_equities: Array of starting equity values (n_signals,)
        sig_prev_values: Array of previous period signals (n_signals, n_bars)
        r_values: Array of adjusted returns (n_bars,)
        decay_multiplier: Decay multiplier (1.0 - De)
        cutout: Number of bars to skip at beginning
        out: Optional output array (n_signals, n_bars). If provided, minimizes allocation.

    Returns:
        Array of equity values (n_signals, n_bars). Values before cutout are np.nan.
    """
    n_signals, n_bars = sig_prev_values.shape
    if out is not None:
        if out.shape != (n_signals, n_bars):
            raise ValueError(f"out array must have shape ({n_signals}, {n_bars})")
        e_values = out
        # We need to explicitly clear previous/garbage data if reusing dirty array?
        # The logic sets each column completely?
        # Logic sets e_values[:, i]. Yes, it writes every index.
        # But if reusing, ensure we don't assume zeros anywhere.
    else:
        e_values = np.full((n_signals, n_bars), np.nan, dtype=np.float64)

    # Initialize previous equity values
    prev_e = np.full(n_signals, np.nan, dtype=np.float64)

    for i in range(n_bars):
        if i < cutout:
            e_values[:, i] = np.nan
            continue

        r_i = r_values[i]
        sig_prev = sig_prev_values[:, i]  # (n_signals,)

        # Calculate 'a' for all signals at once (vectorized)
        # Handle NaN: treat as 0
        valid_mask = ~(np.isnan(sig_prev) | np.isnan(r_i))
        a = np.zeros(n_signals, dtype=np.float64)

        # Long positions: a = r
        long_mask = valid_mask & (sig_prev > 0)
        a[long_mask] = r_i

        # Short positions: a = -r
        short_mask = valid_mask & (sig_prev < 0)
        a[short_mask] = -r_i

        # Calculate current equity for all signals
        # e_curr = (prev_e * decay_multiplier) * (1.0 + a)
        # If prev_e is NaN, use starting_equity
        nan_mask = np.isnan(prev_e)
        e_curr = np.where(nan_mask, starting_equities, (prev_e * decay_multiplier) * (1.0 + a))

        # Apply floor
        e_curr = np.maximum(e_curr, 0.25)

        prev_e = e_curr.copy()
        e_values[:, i] = e_curr

    return e_values


@njit(cache=True)
def _calculate_equity_core_impl(
    r_values: np.ndarray,
    sig_prev_values: np.ndarray,
    starting_equity: float,
    decay_multiplier: float,
    cutout: int,
    out: np.ndarray,
) -> np.ndarray:
    """Core equity calculation implementation (JIT)."""
    n = len(r_values)
    e_values = out

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


def _calculate_equity_core(
    r_values: np.ndarray,
    sig_prev_values: np.ndarray,
    starting_equity: float,
    decay_multiplier: float,
    cutout: int,
    out: np.ndarray = None,
) -> np.ndarray:
    """Core equity calculation function optimized with Numba.

    Wrapper around JIT implementation to handle memory allocation.
    """
    if out is None:
        n = len(r_values)
        out = np.full(n, np.nan, dtype=np.float64)

    return _calculate_equity_core_impl(r_values, sig_prev_values, starting_equity, decay_multiplier, cutout, out)
