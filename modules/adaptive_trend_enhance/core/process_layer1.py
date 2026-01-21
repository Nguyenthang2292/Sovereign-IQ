"""Layer 1 Processing functions for Adaptive Trend Classification (ATC).

This module provides functions for processing signals from multiple Moving
Averages in Layer 1 of the ATC system:
- weighted_signal: Calculate weighted average signal from multiple signals and weights
- cut_signal: Discretize continuous signal into {-1, 0, 1}
- trend_sign: Determine trend direction (+1 for bullish, -1 for bearish, 0 for neutral)
- _layer1_signal_for_ma: Calculate Layer 1 signal for a specific MA type
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from modules.adaptive_trend_enhance.utils.rate_of_change import rate_of_change
from modules.common.utils import log_error, log_warn

from .compute_equity import _calculate_equity_vectorized, equity_series
from .memory_manager import get_memory_manager
from .signal_detection import generate_signal_from_ma

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    THREADING_AVAILABLE = True
except ImportError:
    THREADING_AVAILABLE = False


def weighted_signal(
    signals: Iterable[pd.Series],
    weights: Iterable[pd.Series],
) -> pd.Series:
    """Calculate weighted average signal from multiple signals and weights.

    Optimized Version:
    - Uses NumPy for calculations
    - Validates index consistency
    - Uses hardware and memory management

    Args:
        signals: Iterable of signal series.
        weights: Iterable of weight series.

    Returns:
        pd.Series: Weighted average signal rounded to 2 decimal places.
    """
    signals = list(signals)
    weights = list(weights)

    if len(signals) != len(weights):
        raise ValueError(
            f"signals and weights must have the same length, got {len(signals)} signals and {len(weights)} weights"
        )

    if not signals:
        log_warn("Empty signals/weights provided, returning empty series")
        return pd.Series(dtype="float64")

    # Use memory manager for tracking
    mem_manager = get_memory_manager()

    with mem_manager.track_memory("weighted_signal"):
        # Validate all inputs are Series and have same index
        first_index = signals[0].index
        for i, (sig, wgt) in enumerate(zip(signals, weights)):
            if not isinstance(sig, pd.Series):
                raise TypeError(f"signals[{i}] must be a pandas Series")
            if not isinstance(wgt, pd.Series):
                raise TypeError(f"weights[{i}] must be a pandas Series")

            if not sig.index.equals(first_index):
                log_warn(f"signals[{i}] has different index, aligning...")
                signals[i] = sig.reindex(first_index)
            if not wgt.index.equals(first_index):
                log_warn(f"weights[{i}] has different index, aligning...")
                weights[i] = wgt.reindex(first_index)

        # Convert to NumPy for performance
        n_bars = len(first_index)

        # Pre-allocate NumPy arrays (Task 7)
        num_arr = np.zeros(n_bars, dtype=np.float64)
        den_arr = np.zeros(n_bars, dtype=np.float64)

        # Use GPU if available and workload is large enough (Part of Task 2/3)
        # However, for this specific operation (sum of products),
        # CPU NumPy is usually faster unless data is very large.
        # We'll stick to NumPy for now but ensure it's efficient.

        for sig, wgt in zip(signals, weights):
            # Task 6: Convert Pandas operations to NumPy operations
            s_val = sig.values
            w_val = wgt.values

            # Use NumPy vectorization
            num_arr += s_val * w_val
            den_arr += w_val

        # Handle division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            res_arr = np.divide(num_arr, den_arr)
            res_arr = np.where(np.isfinite(res_arr), res_arr, np.nan)

        result = pd.Series(res_arr, index=first_index, dtype="float64").round(2)
        return result


def cut_signal(
    x: pd.Series,
    threshold: float = 0.49,
    long_threshold: float | None = None,
    short_threshold: float | None = None,
    cutout: int = 0,
) -> pd.Series:
    """Discretize continuous signal into {-1, 0, 1} based on threshold.

    Port of Pine Script function:
        Cut(x) =>
            c = x > 0.49 ? 1 : x < -0.49 ? -1 : 0
            c

    Args:
        x: Continuous signal series.
        threshold: Threshold for discretization (default: 0.49).
            Used if long_threshold and short_threshold are not provided.
        long_threshold: Threshold for LONG signals (default: None, uses threshold).
            Values > long_threshold → 1.
        short_threshold: Threshold for SHORT signals (default: None, uses -threshold).
            Values < short_threshold → -1.
        cutout: Number of bars to skip at beginning (force to 0).
            Values > long_threshold → 1, values < short_threshold → -1, else → 0.

    Returns:
        Series with discrete values {-1, 0, 1}.

    Raises:
        ValueError: If threshold is invalid.
        TypeError: If x is not a pandas Series.
    """
    if not isinstance(x, pd.Series):
        raise TypeError(f"x must be a pandas Series, got {type(x)}")  # pyright: ignore[reportUnreachable]

    # Determine actual thresholds
    if long_threshold is None:
        long_threshold = threshold
    if short_threshold is None:
        short_threshold = -threshold

    if long_threshold <= short_threshold:
        raise ValueError(f"long_threshold ({long_threshold}) must be > short_threshold ({short_threshold})")

    if cutout < 0:
        raise ValueError(f"cutout must be >= 0, got {cutout}")

    if len(x) == 0:
        log_warn("Empty signal series provided, returning empty series")
        return pd.Series(dtype="int8", index=x.index)

    try:
        mem_manager = get_memory_manager()
        with mem_manager.track_memory("cut_signal"):
            c = pd.Series(0, index=x.index, dtype="int8")

        # Handle NaN values: treat as 0 (no signal)
        valid_mask = ~x.isna()

        if valid_mask.any():
            c.loc[valid_mask & (x > long_threshold)] = 1
            c.loc[valid_mask & (x < short_threshold)] = -1

        # Enforce cutout: set first 'cutout' bars to 0
        if cutout > 0 and cutout < len(c):
            c.iloc[:cutout] = 0
            # Also ensure NaN handling aligns if needed, though int8 has no NaN

        # Check for excessive NaN values
        nan_count = (~valid_mask).sum()
        if nan_count > 0:
            nan_pct = (nan_count / len(x)) * 100
            if nan_pct > 10:  # Warn if more than 10% NaN
                log_warn(
                    f"Input signal contains {nan_count} NaN values ({nan_pct:.1f}%). "
                    f"These will be treated as 0 (no signal)."
                )

        return c

    except Exception as e:
        log_error(f"Error discretizing signal: {e}")
        raise


def trend_sign(signal: pd.Series, *, strategy: bool = False) -> pd.Series:
    """Determine trend direction from signal series.

    Numeric version (without colors) of Pine Script function:
        trendcol(signal) =>
            c = strategy ? (signal[1] > 0 ? colup : coldw)
                         : (signal > 0) ? colup : coldw

    Args:
        signal: Signal series.
        strategy: If True, uses signal[1] (previous bar) instead of current signal,
            matching Pine Script behavior.

    Returns:
        Series with trend direction values:
        - +1: Bullish trend (signal > 0)
        - -1: Bearish trend (signal < 0)
        - 0: Neutral (signal == 0)

    Raises:
        TypeError: If signal is not a pandas Series.
    """
    if not isinstance(signal, pd.Series):
        raise TypeError(f"signal must be a pandas Series, got {type(signal)}")  # pyright: ignore[reportUnreachable]

    if len(signal) == 0:
        log_warn("Empty signal series provided, returning empty series")
        return pd.Series(dtype="int8", index=signal.index)

    try:
        mem_manager = get_memory_manager()
        with mem_manager.track_memory("trend_sign"):
            base = signal.shift(1) if strategy else signal
            result = pd.Series(0, index=signal.index, dtype="int8")

        # Handle NaN values: treat as 0 (neutral)
        valid_mask = ~base.isna()

        if valid_mask.any():
            result.loc[valid_mask & (base > 0)] = 1
            result.loc[valid_mask & (base < 0)] = -1

        return result

    except Exception as e:
        log_error(f"Error determining trend sign: {e}")
        raise


def _layer1_signal_for_ma(
    prices: pd.Series,
    ma_tuple: Tuple[pd.Series, ...],
    *,
    L: float,
    De: float,
    cutout: int = 0,
    R: Optional[pd.Series] = None,
) -> Tuple[pd.Series, Tuple[pd.Series, ...], Tuple[pd.Series, ...]]:
    """Calculate Layer 1 signal for a specific Moving Average type.

    Port of Pine Script logic block:
        E   = eq(1, signal(MA),   R), sE   = signal(MA)
        E1  = eq(1, signal(MA1),  R), sE1  = signal(MA1)
        ...
        EMA_Signal = Signal(sE, E, sE1, E1, ..., sE_4, E_4)

    For each of 9 MAs:
    1. Generate signal from price/MA crossover
    2. Calculate equity curve from signal
    3. Weight signals by their equity curves to get final Layer 1 signal

    Performance optimization:
    - Accept R (rate_of_change) as optional parameter to avoid recalculation
    - If R is None, it will be calculated internally (for backwards compatibility)

    Args:
        prices: Price series (typically close prices).
        ma_tuple: Tuple of 9 MA Series: (MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4).
        L: Lambda (growth rate) for equity calculations.
        De: Decay factor for equity calculations.
        cutout: Number of bars to skip at beginning.
        R: Pre-calculated rate of change series. If None, will be calculated internally.

    Returns:
        Tuple containing:
        - signal_series: Weighted Layer 1 signal for this MA type
        - signals_tuple: Tuple of 9 individual signals (s, s1, s2, s3, s4, s_1, s_2, s_3, s_4)
        - equity_tuple: Tuple of 9 equity curves (E, E1, E2, E3, E4, E_1, E_2, E_3, E_4)

    Raises:
        ValueError: If ma_tuple doesn't have exactly 9 elements or inputs are invalid.
        TypeError: If inputs are not pandas Series.
    """
    # Input validation
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pandas Series, got {type(prices)}")

    if len(prices) == 0:
        raise ValueError("prices cannot be empty")

    if not isinstance(ma_tuple, tuple):
        raise TypeError(f"ma_tuple must be a tuple, got {type(ma_tuple)}")

    EXPECTED_MA_COUNT = 9
    if len(ma_tuple) != EXPECTED_MA_COUNT:
        raise ValueError(f"ma_tuple must contain exactly {EXPECTED_MA_COUNT} MA series, got {len(ma_tuple)}")

    # Validate all MAs are Series
    for i, ma in enumerate(ma_tuple):
        if not isinstance(ma, pd.Series):
            raise TypeError(f"ma_tuple[{i}] must be a pandas Series, got {type(ma)}")
        if len(ma) == 0:
            raise ValueError(f"ma_tuple[{i}] cannot be empty")

    # Validate parameters
    if not isinstance(L, (int, float)) or np.isnan(L) or np.isinf(L):
        raise ValueError(f"L must be a finite number, got {L}")

    if not (0 <= De <= 1):
        raise ValueError(f"De must be between 0 and 1, got {De}")

    if cutout < 0:
        raise ValueError(f"cutout must be >= 0, got {cutout}")

    try:
        mem_manager = get_memory_manager()
        with mem_manager.track_memory("_layer1_signal_for_ma"):
            # Unpack MA tuple
            (
                MA,
                MA1,
                MA2,
                MA3,
                MA4,
                MA_1,
                MA_2,
                MA_3,
                MA_4,
            ) = ma_tuple

            if R is None:
                R = rate_of_change(prices)

            # Generate signals for all MAs (optimized with list comprehension)
            ma_list = [MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4]
            signals = [generate_signal_from_ma(prices, ma) for ma in ma_list]

            # Unpack signals for return tuple (maintaining original variable names)
            s, s1, s2, s3, s4, s_1, s_2, s_3, s_4 = signals

            # Calculate equity curves - try vectorized version first if all signals have same length
            try:
                # Check if we can use vectorized version
                signal_lengths = [len(sig) for sig in signals]
                if len(set(signal_lengths)) == 1 and all(len(sig) == len(R) for sig in signals):
                    # All signals have same length, use vectorized version
                    from modules.adaptive_trend.utils import exp_growth

                    # Prepare data for vectorized calculation
                    index = signals[0].index
                    growth = exp_growth(L=L, index=index, cutout=cutout)
                    r = R * growth
                    d = 1.0 - De

                    # Shift all signals
                    sig_shifted_list = [sig.shift(1) for sig in signals]

                    # Convert to numpy arrays
                    r_values = r.values
                    sig_prev_values = np.array([sig.values for sig in sig_shifted_list])  # (9, n_bars)
                    starting_equities = np.ones(9, dtype=np.float64)  # All start at 1.0

                    # Calculate all equities at once
                    e_values_array = _calculate_equity_vectorized(
                        starting_equities=starting_equities,
                        sig_prev_values=sig_prev_values,
                        r_values=r_values,
                        decay_multiplier=d,
                        cutout=cutout,
                    )

                    # Convert back to Series
                    equities = [pd.Series(e_values_array[i], index=index, dtype="float64") for i in range(9)]
                else:
                    # Fallback to sequential calculation
                    equities = [equity_series(1.0, sig, R, L=L, De=De, cutout=cutout) for sig in signals]
            except Exception:
                # Fallback to sequential calculation on any error
                log_warn("Vectorized equity calculation failed, using sequential version")
                equities = [equity_series(1.0, sig, R, L=L, De=De, cutout=cutout) for sig in signals]

            # Unpack equities for return tuple (maintaining original variable names)
            E, E1, E2, E3, E4, E_1, E_2, E_3, E_4 = equities

            # Calculate weighted signal
            signal_series = weighted_signal(
                signals=signals,
                weights=equities,
            )

            return (
                signal_series,
                (s, s1, s2, s3, s4, s_1, s_2, s_3, s_4),
                (E, E1, E2, E3, E4, E_1, E_2, E_3, E_4),
            )

    except Exception as e:
        log_error(f"Error calculating Layer 1 signal for MA: {e}")
        raise


__all__ = [
    "weighted_signal",
    "cut_signal",
    "trend_sign",
    "_layer1_signal_for_ma",
]
