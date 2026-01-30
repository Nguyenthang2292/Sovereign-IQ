"""
Core equity calculation functions.

This module provides optimized calculation functions for equity curves:
- _calculate_equity_core: Single signal Numba JIT optimized calculation
- _calculate_equity_vectorized: Vectorized calculation for multiple signals (serial nested loops)
- _calculate_equities_parallel: Parallel calculation for multiple signals (uses prange)
- calculate_equity: Unified interface with automatic strategy selection

IMPORTANT - Equity Floor Behavior:
The DEFAULT_EQUITY_FLOOR (0.25) prevents equity from dropping below this value.
This is a SAFETY MECHANISM to prevent numerical instability but MASKS TRUE RISK.
Example: A 90% loss will show as only 17% loss (from 1.0 to 0.25 floor).

For accurate risk assessment, consider:
1. Setting DEFAULT_EQUITY_FLOOR = 0.0 (no floor)
2. Monitoring equity values before floor application via logging
3. Using drawdown metrics instead of floor-equity for risk calculation

Performance Guidelines:
- Single signal: Use _calculate_equity_core (fastest for individual signals)
- Multiple signals (n < 50): Use _calculate_equity_vectorized (lower overhead)
- Multiple signals (n >= 50): Use _calculate_equities_parallel (parallel speedup)
- Automatic selection: Use calculate_equity() which selects optimal strategy

Typical Performance:
- Single signal: ~0.1-0.5ms per signal
- Vectorized (n=20): ~2-5ms total
- Parallel (n=100): ~5-15ms total with 4+ cores
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .utils import njit, prange

try:
    from pandas.errors import PerformanceWarning
except ImportError:

    class PerformanceWarning(UserWarning):
        """Warning for performance-related issues."""

        pass


# Default equity floor value - configurable via environment variable
# Set ATC_EQUITY_FLOOR=0.0 to disable floor for accurate risk calculation
# ⚠️ WARNING: Floor masks true losses. 90% loss shows as 17% loss.
_DEFAULT_FLOOR = 0.25


def _get_equity_floor_from_env() -> float:
    """Get equity floor from environment with validation.

    Returns:
        Validated equity floor value.

    Raises:
        Warning if environment variable is invalid.
    """
    import warnings

    env_value = os.environ.get("ATC_EQUITY_FLOOR")
    if env_value is None:
        return _DEFAULT_FLOOR

    try:
        value = float(env_value)
        if value < 0:
            warnings.warn(
                f"ATC_EQUITY_FLOOR must be non-negative, got {value}. Using default {_DEFAULT_FLOOR}",
                UserWarning,
                stacklevel=2,
            )
            return _DEFAULT_FLOOR
        if value > 1.0:
            warnings.warn(
                f"ATC_EQUITY_FLOOR should typically be <= 1.0, got {value}. This may cause unexpected behavior.",
                UserWarning,
                stacklevel=2,
            )
        return value
    except ValueError as e:
        warnings.warn(
            f"Invalid ATC_EQUITY_FLOOR value '{env_value}': {e}. Using default {_DEFAULT_FLOOR}",
            UserWarning,
            stacklevel=2,
        )
        return _DEFAULT_FLOOR


DEFAULT_EQUITY_FLOOR: float = _get_equity_floor_from_env()

# Signal thresholds for position determination
# Signals > LONG_SIGNAL_THRESHOLD are treated as long positions
# Signals < SHORT_SIGNAL_THRESHOLD are treated as short positions
# Signals between thresholds are treated as neutral (no position)
_LONG_SIGNAL_THRESHOLD = 0.5
_SHORT_SIGNAL_THRESHOLD = -0.5

# Equity threshold below which we treat starting equity as effectively zero
# This prevents numerical issues with very small equity values
_ZERO_EQUITY_THRESHOLD = 1e-10


def get_equity_floor() -> float:
    """Get current equity floor value."""
    return DEFAULT_EQUITY_FLOOR


def set_equity_floor(floor: Optional[float]) -> None:
    """Set equity floor value globally.

    Args:
        floor: Floor value (0.0 to disable, or positive value).
               None resets to default (_DEFAULT_FLOOR).

    Note:
        Setting floor=0.0 provides accurate risk calculation but may cause
        numerical issues with very small equity values.
    """
    global DEFAULT_EQUITY_FLOOR
    if floor is None:
        DEFAULT_EQUITY_FLOOR = _DEFAULT_FLOOR
    else:
        DEFAULT_EQUITY_FLOOR = float(floor)


__all__ = [
    # Core calculation functions
    "_calculate_equity_core",
    "_calculate_equity_vectorized",
    "_calculate_equities_parallel",
    "calculate_equity",
    # Configuration
    "DEFAULT_EQUITY_FLOOR",
    "get_equity_floor",
    "set_equity_floor",
    # Benchmarking
    "benchmark_strategies",
    "print_benchmark_results",
]


@njit(cache=True)
def _calculate_equity_core_impl(
    r_values: np.ndarray,  # shape: (n_bars,), dtype: float64
    sig_prev_values: np.ndarray,  # shape: (n_bars,), dtype: float64
    starting_equity: float,
    decay_multiplier: float,
    cutout: int,
    out: np.ndarray,  # shape: (n_bars,), dtype: float64
    floor_val: float,
) -> np.ndarray:  # shape: (n_bars,), dtype: float64
    """Core equity calculation implementation (JIT).

    Note:
        This function is JIT-compiled by Numba. Type annotations are for
        documentation only and not enforced by Numba at runtime.

    Args:
        r_values: Log returns (n_bars,)
        sig_prev_values: Previous signals (n_bars,)
        starting_equity: Starting equity value
        decay_multiplier: Equity decay multiplier
        cutout: Bars to skip at start
        out: Pre-allocated output array
        floor_val: Equity floor value

    Returns:
        Equity curve array (n_bars,)
    """
    n = len(r_values)
    e_values = out

    # Use np.nan to indicate "not initialized" instead of None (Numba doesn't support Optional)
    prev_e = np.nan

    # If weight is exactly 0, force it to 0 and return early
    if starting_equity < _ZERO_EQUITY_THRESHOLD:
        for i in range(n):
            e_values[i] = 0.0
        return e_values

    for i in range(n):
        if i < cutout:
            e_values[i] = np.nan
            continue

        r_i = r_values[i]
        sig_prev = sig_prev_values[i]

        # Handle NaN in signal or return
        a_val = 0.0
        if not (np.isnan(sig_prev) or np.isnan(r_i)):
            if sig_prev > _LONG_SIGNAL_THRESHOLD:
                a_val = r_i
            elif sig_prev < _SHORT_SIGNAL_THRESHOLD:
                a_val = -r_i

        # Calculate current equity
        if np.isnan(prev_e):
            e_curr = starting_equity
        else:
            e_curr = (prev_e * decay_multiplier) * (1.0 + a_val)

        # Apply floor
        if e_curr < floor_val:
            e_curr = floor_val

        prev_e = e_curr
        e_values[i] = e_curr

    return e_values


@njit(parallel=True, cache=True)
def _calculate_equities_parallel_impl(
    starting_equities: np.ndarray,  # shape: (n_signals,), dtype: float64
    sig_prev_values: np.ndarray,  # shape: (n_signals, n_bars), dtype: float64
    r_values: np.ndarray,  # shape: (n_bars,), dtype: float64
    decay_multiplier: float,
    cutout: int,
    out: np.ndarray,  # shape: (n_signals, n_bars), dtype: float64
    floor_val: float,
) -> np.ndarray:  # shape: (n_signals, n_bars), dtype: float64
    """Numba-parallel equity calculation for multiple signals.

    Parallelizes the calculation across the 'signals' dimension using prange.

    Note:
        This function is JIT-compiled by Numba with parallel=True. Type
        annotations are for documentation only.

    Args:
        starting_equities: Starting equity values (n_signals,)
        sig_prev_values: Previous signals (n_signals, n_bars)
        r_values: Log returns (n_bars,)
        decay_multiplier: Equity decay multiplier
        cutout: Bars to skip at start
        out: Pre-allocated output array
        floor_val: Equity floor value

    Returns:
        Equity curves array (n_signals, n_bars)
    """
    n_signals, n_bars = sig_prev_values.shape

    for s in prange(n_signals):
        p_e = np.nan
        s_eq = starting_equities[s]

        # If weight is exactly 0, force it to 0 and skip math
        if s_eq < 1e-10:
            for i in range(n_bars):
                out[s, i] = 0.0
            continue

        for i in range(n_bars):
            if i < cutout:
                out[s, i] = np.nan
                continue

            rv = r_values[i]
            sp = sig_prev_values[s, i]

            # Handle NaN in signal or return
            av = 0.0
            if not (np.isnan(sp) or np.isnan(rv)):
                if sp > 0.5:
                    av = rv
                elif sp < -0.5:
                    av = -rv

            # Calculate current equity
            if np.isnan(p_e):
                ec = s_eq
            else:
                ec = (p_e * decay_multiplier) * (1.0 + av)

            # Apply floor
            if ec < floor_val:
                ec = floor_val

            out[s, i] = ec
            p_e = ec

    return out


@njit(cache=True)
def _calculate_equity_vectorized_jit_impl(
    starting_equities: np.ndarray,  # shape: (n_signals,), dtype: float64
    sig_prev_values: np.ndarray,  # shape: (n_signals, n_bars), dtype: float64
    r_values: np.ndarray,  # shape: (n_bars,), dtype: float64
    decay_multiplier: float,
    cutout: int,
    out: np.ndarray,  # shape: (n_signals, n_bars), dtype: float64
    floor_val: float,
) -> np.ndarray:  # shape: (n_signals, n_bars), dtype: float64
    """JIT version of vectorized equity calculation.

    Serial nested loops implementation. Best for small to medium n_signals
    where parallel overhead is not justified.

    Note:
        This function is JIT-compiled by Numba. Type annotations are for
        documentation only.

    Args:
        starting_equities: Starting equity values (n_signals,)
        sig_prev_values: Previous signals (n_signals, n_bars)
        r_values: Log returns (n_bars,)
        decay_multiplier: Equity decay multiplier
        cutout: Bars to skip at start
        out: Pre-allocated output array
        floor_val: Equity floor value

    Returns:
        Equity curves array (n_signals, n_bars)
    """
    n_signals, n_bars = sig_prev_values.shape
    e_vals = out

    p_e = np.empty(n_signals, dtype=np.float64)
    for s in range(n_signals):
        p_e[s] = np.nan

    for i in range(n_bars):
        rv = r_values[i]

        for s in range(n_signals):
            s_eq = starting_equities[s]

            # If weight is exactly 0, force it to 0 and skip math
            if s_eq < 1e-10:
                e_vals[s, i] = 0.0
                p_e[s] = 0.0
                continue

            if i < cutout:
                e_vals[s, i] = np.nan
                continue

            sp = sig_prev_values[s, i]

            # Handle NaN/0
            av = 0.0
            if not (np.isnan(sp) or np.isnan(rv)):
                if sp > 0.5:
                    av = rv
                elif sp < -0.5:
                    av = -rv

            cur_p_e = p_e[s]
            if np.isnan(cur_p_e):
                val = s_eq
            else:
                val = (cur_p_e * decay_multiplier) * (1.0 + av)

            # Apply floor
            if val < floor_val:
                val = floor_val

            e_vals[s, i] = val
            p_e[s] = val

    return e_vals


def _calculate_equity_core(
    r_values: np.ndarray,
    sig_prev_values: np.ndarray,
    starting_equity: float,
    decay_multiplier: float,
    cutout: int,
    out: Optional[np.ndarray] = None,
    floor_val: Optional[float] = None,
) -> np.ndarray:
    """Core equity calculation function optimized with Numba.

    Calculates equity curve for a single signal using JIT-compiled implementation.
    This is the fastest option for individual signal equity calculation.

    Args:
        r_values: Log returns array (n_bars,), e.g., np.log(close[1:] / close[:-1])
        sig_prev_values: Previous period signals (n_bars,), values in {-1, 0, 1}
                         1 = long, -1 = short, 0 = neutral
        starting_equity: Initial equity value, typically 1.0
        decay_multiplier: Equity decay per bar (1.0 - decay_rate), e.g., 0.99 for 1% decay
        cutout: Number of initial bars to skip (set to NaN), typically indicator warmup period
        out: Optional pre-allocated output array (n_bars,) for memory efficiency
        floor_val: Minimum equity value (default: DEFAULT_EQUITY_FLOOR = 0.25)

    Returns:
        Equity curve array (n_bars,). Values before cutout are NaN.
        Equity at bar i represents cumulative P&L from trading signals.

    Example:
        >>> # Simple long-only strategy
        >>> returns = np.array([0.01, -0.02, 0.015, 0.005])
        >>> signals = np.array([1.0, 1.0, 1.0, 1.0])  # All long
        >>> equity = _calculate_equity_core(
        ...     r_values=returns,
        ...     sig_prev_values=signals,
        ...     starting_equity=1.0,
        ...     decay_multiplier=1.0,  # No decay
        ...     cutout=0
        ... )
        >>> # equity[0] = 1.0 * (1 + 0.01) = 1.01
        >>> # equity[1] = 1.01 * (1 - 0.02) = 0.9898
        >>> # etc.

    Performance:
        ~0.1-0.5ms per signal on modern CPUs
    """
    if floor_val is None:
        floor_val = DEFAULT_EQUITY_FLOOR

    # Ensure out is initialized correctly for implementation
    n = len(r_values)
    res_out: np.ndarray
    if out is None:
        res_out = np.full(n, np.nan, dtype=np.float64)
    else:
        res_out = out
        if not res_out.flags.writeable:
            import warnings

            warnings.warn(
                "Output array is not writeable, creating copy. This may impact performance for large arrays.",
                PerformanceWarning,
                stacklevel=2,
            )
            res_out = res_out.copy()

    return _calculate_equity_core_impl(
        r_values, sig_prev_values, starting_equity, decay_multiplier, cutout, res_out, floor_val
    )


def _calculate_equities_parallel(
    starting_equities: np.ndarray,
    sig_prev_values: np.ndarray,
    r_values: np.ndarray,
    decay_multiplier: float,
    cutout: int,
    out: Optional[np.ndarray] = None,
    floor_val: Optional[float] = None,
) -> np.ndarray:
    """Parallel equity calculation for multiple signals optimized with Numba.

    Uses Numba's prange for parallel processing across signals. Best for large
    numbers of signals (n >= 50) where parallelization overhead is justified.

    Args:
        starting_equities: Array of starting equity values (n_signals,)
        sig_prev_values: Array of previous period signals (n_signals, n_bars)
        r_values: Array of log returns (n_bars,)
        decay_multiplier: Decay multiplier (1.0 - decay_rate)
        cutout: Number of bars to skip at beginning
        out: Optional output array (n_signals, n_bars)
        floor_val: Optional equity floor

    Returns:
        Array of equity values (n_signals, n_bars). Values before cutout are NaN.

    Example:
        >>> # Calculate equity for 100 different signals in parallel
        >>> n_signals = 100
        >>> n_bars = 1000
        >>> starting_equities = np.ones(n_signals)
        >>> signals = np.random.choice([-1, 0, 1], size=(n_signals, n_bars))
        >>> returns = np.random.randn(n_bars) * 0.01
        >>> equities = _calculate_equities_parallel(
        ...     starting_equities=starting_equities,
        ...     sig_prev_values=signals,
        ...     r_values=returns,
        ...     decay_multiplier=0.999,
        ...     cutout=50
        ... )
        >>> equities.shape
        (100, 1000)

    Performance:
        ~5-15ms for 100 signals x 1000 bars on 4+ cores
        Parallel speedup scales with number of cores up to n_signals

    See Also:
        _calculate_equity_vectorized: Better for smaller n_signals (< 50)
    """
    if floor_val is None:
        floor_val = DEFAULT_EQUITY_FLOOR

    n_signals, n_bars = sig_prev_values.shape
    res_out: np.ndarray
    if out is None:
        # Match input dtype
        res_out = np.full((n_signals, n_bars), np.nan, dtype=starting_equities.dtype)
    else:
        res_out = out
        if not res_out.flags.writeable:
            import warnings

            warnings.warn(
                "Output array is not writeable, creating copy. This may impact performance for large arrays.",
                PerformanceWarning,
                stacklevel=2,
            )
            res_out = res_out.copy()

    return _calculate_equities_parallel_impl(
        starting_equities, sig_prev_values, r_values, decay_multiplier, cutout, res_out, floor_val
    )


def _calculate_equity_vectorized(
    starting_equities: np.ndarray,
    sig_prev_values: np.ndarray,
    r_values: np.ndarray,
    decay_multiplier: float,
    cutout: int,
    out: Optional[np.ndarray] = None,
    floor_val: Optional[float] = None,
) -> np.ndarray:
    """Vectorized equity calculation for multiple signals at once.

    This function calculates equity curves for multiple signals simultaneously
    using Numba JIT serial nested loops. Best for small to medium numbers of
    signals (n < 50) where parallel overhead is not justified.

    Args:
        starting_equities: Array of starting equity values (n_signals,)
        sig_prev_values: Array of previous period signals (n_signals, n_bars)
        r_values: Array of log returns (n_bars,)
        decay_multiplier: Decay multiplier (1.0 - decay_rate)
        cutout: Number of bars to skip at beginning
        out: Optional output array (n_signals, n_bars). If provided, minimizes allocation.
        floor_val: Optional equity floor.

    Returns:
        Array of equity values (n_signals, n_bars). Values before cutout are np.nan.

    Example:
        >>> # Calculate equity for 20 different signals
        >>> n_signals = 20
        >>> n_bars = 500
        >>> starting_equities = np.ones(n_signals)
        >>> signals = np.random.choice([-1, 0, 1], size=(n_signals, n_bars))
        >>> returns = np.random.randn(n_bars) * 0.01
        >>> equities = _calculate_equity_vectorized(
        ...     starting_equities=starting_equities,
        ...     sig_prev_values=signals,
        ...     r_values=returns,
        ...     decay_multiplier=0.999,
        ...     cutout=20
        ... )
        >>> equities.shape
        (20, 500)

    Performance:
        ~2-5ms for 20 signals x 500 bars
        Lower overhead than parallel version for small n_signals

    See Also:
        _calculate_equities_parallel: Better for larger n_signals (>= 50)
    """
    if floor_val is None:
        floor_val = DEFAULT_EQUITY_FLOOR

    n_signals, n_bars = sig_prev_values.shape

    # Correct handling for 'out'
    res_out: np.ndarray
    if out is None:
        res_out = np.full((n_signals, n_bars), np.nan, dtype=starting_equities.dtype)
    else:
        res_out = out
        if not res_out.flags.writeable:
            res_out = res_out.copy()

    return _calculate_equity_vectorized_jit_impl(
        starting_equities, sig_prev_values, r_values, decay_multiplier, cutout, res_out, floor_val
    )


# Threshold for automatic strategy selection
_PARALLEL_THRESHOLD = 50  # Use parallel for n_signals >= this value


def calculate_equity(
    starting_equities: np.ndarray,
    sig_prev_values: np.ndarray,
    r_values: np.ndarray,
    decay_multiplier: float,
    cutout: int,
    out: Optional[np.ndarray] = None,
    floor_val: Optional[float] = None,
    force_strategy: Optional[str] = None,
) -> np.ndarray:
    """Unified equity calculation interface with automatic strategy selection.

    This is the recommended entry point for equity calculations. It automatically
    selects the optimal implementation based on input size:
    - Single signal (1D): Uses _calculate_equity_core
    - Multiple signals (n < 50): Uses _calculate_equity_vectorized
    - Multiple signals (n >= 50): Uses _calculate_equities_parallel

    Args:
        starting_equities: Starting equity value(s). Can be:
                          - float or scalar array for single signal
                          - 1D array (n_signals,) for multiple signals
        sig_prev_values: Previous period signals. Can be:
                        - 1D array (n_bars,) for single signal
                        - 2D array (n_signals, n_bars) for multiple signals
        r_values: Array of log returns (n_bars,)
        decay_multiplier: Decay multiplier (1.0 - decay_rate)
        cutout: Number of bars to skip at beginning
        out: Optional output array for memory efficiency
        floor_val: Optional equity floor (default: DEFAULT_EQUITY_FLOOR)
        force_strategy: Force specific strategy: 'core', 'vectorized', or 'parallel'
                       None (default) = automatic selection

    Returns:
        Equity curve(s). Shape matches sig_prev_values shape.

    Example:
        >>> # Single signal - automatically uses _calculate_equity_core
        >>> returns = np.random.randn(1000) * 0.01
        >>> signals = np.random.choice([-1, 0, 1], size=1000)
        >>> equity = calculate_equity(
        ...     starting_equities=1.0,
        ...     sig_prev_values=signals,
        ...     r_values=returns,
        ...     decay_multiplier=0.999,
        ...     cutout=20
        ... )
        >>> equity.shape
        (1000,)

        >>> # 20 signals - automatically uses _calculate_equity_vectorized
        >>> signals = np.random.choice([-1, 0, 1], size=(20, 1000))
        >>> equities = calculate_equity(
        ...     starting_equities=np.ones(20),
        ...     sig_prev_values=signals,
        ...     r_values=returns,
        ...     decay_multiplier=0.999,
        ...     cutout=20
        ... )
        >>> equities.shape
        (20, 1000)

        >>> # 100 signals - automatically uses _calculate_equities_parallel
        >>> signals = np.random.choice([-1, 0, 1], size=(100, 1000))
        >>> equities = calculate_equity(
        ...     starting_equities=np.ones(100),
        ...     sig_prev_values=signals,
        ...     r_values=returns,
        ...     decay_multiplier=0.999,
        ...     cutout=20
        ... )
        >>> equities.shape
        (100, 1000)

    Performance:
        Strategy selection overhead is negligible (~microseconds).
        See individual function docstrings for performance characteristics.

    See Also:
        _calculate_equity_core: Single signal calculation
        _calculate_equity_vectorized: Small batch calculation
        _calculate_equities_parallel: Large batch parallel calculation
    """
    # Handle single signal case
    if sig_prev_values.ndim == 1:
        if force_strategy and force_strategy != "core":
            import warnings

            warnings.warn(
                f"force_strategy='{force_strategy}' ignored for single signal (using 'core')", UserWarning, stacklevel=2
            )
        # Convert scalar to float if needed
        starting_eq = float(starting_equities) if np.isscalar(starting_equities) else float(starting_equities.item())
        return _calculate_equity_core(r_values, sig_prev_values, starting_eq, decay_multiplier, cutout, out, floor_val)

    # Multiple signals case
    n_signals = sig_prev_values.shape[0]

    # Validate starting_equities shape
    starting_eq_arr = np.asarray(starting_equities)
    if starting_eq_arr.ndim == 0:
        starting_eq_arr = np.full(n_signals, float(starting_eq_arr))
    elif len(starting_eq_arr) != n_signals:
        raise ValueError(
            f"starting_equities length ({len(starting_eq_arr)}) must match number of signals ({n_signals})"
        )

    # Strategy selection
    if force_strategy == "vectorized":
        return _calculate_equity_vectorized(
            starting_eq_arr, sig_prev_values, r_values, decay_multiplier, cutout, out, floor_val
        )
    elif force_strategy == "parallel":
        return _calculate_equities_parallel(
            starting_eq_arr, sig_prev_values, r_values, decay_multiplier, cutout, out, floor_val
        )
    elif force_strategy is not None:
        raise ValueError(f"Invalid force_strategy: {force_strategy}. Use 'core', 'vectorized', or 'parallel'")

    # Automatic selection based on number of signals
    if n_signals < _PARALLEL_THRESHOLD:
        return _calculate_equity_vectorized(
            starting_eq_arr, sig_prev_values, r_values, decay_multiplier, cutout, out, floor_val
        )
    else:
        return _calculate_equities_parallel(
            starting_eq_arr, sig_prev_values, r_values, decay_multiplier, cutout, out, floor_val
        )


def benchmark_strategies(
    n_signals_list: list[int] = [1, 10, 20, 50, 100, 200],
    n_bars: int = 1000,
    n_iterations: int = 10,
) -> dict:
    """Benchmark different equity calculation strategies.

    Runs performance tests to compare core, vectorized, and parallel strategies
    across different numbers of signals. Useful for tuning _PARALLEL_THRESHOLD.

    Args:
        n_signals_list: List of signal counts to test
        n_bars: Number of bars per signal
        n_iterations: Number of iterations for timing

    Returns:
        Dictionary with benchmark results:
        {
            'n_signals': [...],
            'core_times': [...],      # Only for n_signals=1
            'vectorized_times': [...],
            'parallel_times': [...],
            'auto_times': [...]       # Using calculate_equity
        }

    Example:
        >>> results = benchmark_strategies(
        ...     n_signals_list=[1, 10, 50, 100],
        ...     n_bars=1000,
        ...     n_iterations=20
        ... )
        >>> # Print results
        >>> for i, n in enumerate(results['n_signals']):
        ...     print(f"n_signals={n}:")
        ...     print(f"  Vectorized: {results['vectorized_times'][i]:.3f}ms")
        ...     print(f"  Parallel: {results['parallel_times'][i]:.3f}ms")

    Note:
        First run may be slower due to Numba JIT compilation.
        Results vary based on CPU cores and system load.
    """
    import time

    results = {
        "n_signals": n_signals_list,
        "core_times": [],
        "vectorized_times": [],
        "parallel_times": [],
        "auto_times": [],
    }

    for n_signals in n_signals_list:
        print(f"Benchmarking n_signals={n_signals}...")

        # Generate test data
        if n_signals == 1:
            signals = np.random.choice([-1, 0, 1], size=n_bars)
            starting_eq = 1.0
        else:
            signals = np.random.choice([-1, 0, 1], size=(n_signals, n_bars))
            starting_eq = np.ones(n_signals)

        returns = np.random.randn(n_bars) * 0.01
        decay = 0.999
        cutout = 20

        # Warmup (JIT compilation)
        _ = calculate_equity(starting_eq, signals, returns, decay, cutout)

        # Benchmark core (only for single signal)
        if n_signals == 1:
            start = time.perf_counter()
            for _ in range(n_iterations):
                _ = _calculate_equity_core(returns, signals, starting_eq, decay, cutout)
            elapsed = (time.perf_counter() - start) * 1000 / n_iterations
            results["core_times"].append(elapsed)
        else:
            results["core_times"].append(None)

        # Benchmark vectorized
        if n_signals > 1:
            start = time.perf_counter()
            for _ in range(n_iterations):
                _ = _calculate_equity_vectorized(starting_eq, signals, returns, decay, cutout)
            elapsed = (time.perf_counter() - start) * 1000 / n_iterations
            results["vectorized_times"].append(elapsed)
        else:
            results["vectorized_times"].append(None)

        # Benchmark parallel
        if n_signals > 1:
            start = time.perf_counter()
            for _ in range(n_iterations):
                _ = _calculate_equities_parallel(starting_eq, signals, returns, decay, cutout)
            elapsed = (time.perf_counter() - start) * 1000 / n_iterations
            results["parallel_times"].append(elapsed)
        else:
            results["parallel_times"].append(None)

        # Benchmark auto-selection
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = calculate_equity(starting_eq, signals, returns, decay, cutout)
        elapsed = (time.perf_counter() - start) * 1000 / n_iterations
        results["auto_times"].append(elapsed)

    return results


def print_benchmark_results(results: dict) -> None:
    """Pretty print benchmark results.

    Args:
        results: Dictionary from benchmark_strategies()

    Example:
        >>> results = benchmark_strategies()
        >>> print_benchmark_results(results)
        Benchmark Results (times in ms):
        ================================
        n_signals=1:
          Core:       0.123 ms
          Auto:       0.125 ms
        ...
    """
    print("\nBenchmark Results (times in ms):")
    print("=" * 60)

    for i, n in enumerate(results["n_signals"]):
        print(f"\nn_signals={n}:")

        if results["core_times"][i] is not None:
            print(f"  Core:       {results['core_times'][i]:7.3f} ms")

        if results["vectorized_times"][i] is not None:
            print(f"  Vectorized: {results['vectorized_times'][i]:7.3f} ms")

        if results["parallel_times"][i] is not None:
            print(f"  Parallel:   {results['parallel_times'][i]:7.3f} ms")

        print(f"  Auto:       {results['auto_times'][i]:7.3f} ms")

        # Show speedup for multi-signal cases
        if results["vectorized_times"][i] and results["parallel_times"][i]:
            speedup = results["vectorized_times"][i] / results["parallel_times"][i]
            if speedup > 1.0:
                print(f"  → Parallel is {speedup:.2f}x faster")
            else:
                print(f"  → Vectorized is {1 / speedup:.2f}x faster")

    print("\n" + "=" * 60)
    print(f"Parallel threshold: {_PARALLEL_THRESHOLD} signals")
    print("Recommendation: Use parallel for n_signals >= threshold")
