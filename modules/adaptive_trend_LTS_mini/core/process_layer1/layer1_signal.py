"""Layer 1 signal calculation for Moving Averages.

This module provides the _layer1_signal_for_ma function to calculate
Layer 1 signal for a specific Moving Average type.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS_mini.core.compute_equity import _calculate_equity_vectorized, equity_series
from modules.adaptive_trend_LTS_mini.core.signal_detection import generate_signal_from_ma
from modules.adaptive_trend_LTS_mini.utils.rate_of_change import rate_of_change
from modules.common.system import get_array_pool
from modules.common.utils import log_error, log_warn


def _layer1_signal_for_ma(
    prices: pd.Series,
    ma_tuple: Tuple[pd.Series, ...],
    *,
    lambda_val: float,
    decay_val: float,
    rate_of_change_series: Optional[pd.Series] = None,
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
    - Accept rate_of_change_series (rate_of_change) as optional parameter to avoid recalculation
    - If rate_of_change_series is None, it will be calculated internally (for backwards compatibility)

    Args:
        prices: Price series (typically close prices).
        ma_tuple: Tuple of 9 MA Series: (MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4).
        lambda_val: Lambda (growth rate) for equity calculations.
        decay_val: Decay factor for equity calculations.
        cutout: Number of bars to skip at beginning.
        rate_of_change_series: Pre-calculated rate of change series. If None, will be calculated internally.

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
    if not isinstance(lambda_val, (int, float)) or np.isnan(lambda_val) or np.isinf(lambda_val):
        raise ValueError(f"lambda_val must be a finite number, got {lambda_val}")

    if not (0 <= decay_val <= 1):
        raise ValueError(f"decay_val must be between 0 and 1, got {decay_val}")

    try:
        # REMOVED: mem_manager = get_memory_manager()
        # REMOVED: with mem_manager.track_memory("_layer1_signal_for_ma"):

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

        if rate_of_change_series is None:
            rate_of_change_series = rate_of_change(prices)

        # Generate signals for all MAs (optimized with list comprehension)
        ma_list = [MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4]
        signals = [generate_signal_from_ma(prices, ma) for ma in ma_list]

        # Unpack signals for return tuple (maintaining original variable names)
        s, s1, s2, s3, s4, s_1, s_2, s_3, s_4 = signals

        # Calculate equity curves - try vectorized version first if all signals have same length
        try:
            # Check if we can use vectorized version
            signal_lengths = [len(sig) for sig in signals]
            if len(set(signal_lengths)) == 1 and all(len(sig) == len(rate_of_change_series) for sig in signals):
                # All signals have same length, use vectorized version
                from modules.adaptive_trend_LTS_mini.utils.exp_growth import exp_growth

                # Prepare data for vectorized calculation
                index = signals[0].index
                growth = exp_growth(lambda_val=lambda_val, index=index, cutout=0)
                r = rate_of_change_series * growth
                d = 1.0 - decay_val

                # Shift all signals and vectorize
                # Optimize: Use ArrayPool for input buffer to avoid 9 shift() allocations and 1 big array
                n_bars = len(signals[0])
                pool = get_array_pool()

                # Acquire dirty buffer (9, N) - used as INPUT buffer only
                # NOTE: This buffer is ONLY for input data. The output array is
                # allocated separately by _calculate_equity_vectorized (out=None),
                # so there is NO race condition when we release the buffer.
                sig_prev_values = pool.acquire_dirty((9, n_bars), dtype=np.float64)

                try:
                    # Fill buffer directly (equivalent to shift(1))
                    # NOTE: This shift is for INTERNAL equity calculation only (to use sig[1] as per Pine Script)
                    # The actual signal values returned to caller are NOT shifted
                    for i, sig in enumerate(signals):
                        vals = sig.values
                        sig_prev_values[i, 1:] = vals[:-1]
                        sig_prev_values[i, 0] = np.nan

                    r_values = r.values
                    starting_equities = np.ones(9, dtype=np.float64)

                    # Calculate all equities at once
                    # MEMORY SAFETY: _calculate_equity_vectorized allocates a NEW array when out=None,
                    # so e_values_array is completely independent of sig_prev_values buffer.
                    # Safe to release sig_prev_values immediately after calculation.
                    e_values_array = _calculate_equity_vectorized(
                        starting_equities=starting_equities,
                        sig_prev_values=sig_prev_values,
                        r_values=np.asarray(r_values, dtype=np.float64),
                        decay_multiplier=d,
                        cutout=0,
                    )
                finally:
                    # Always release input buffer - safe because output is separate allocation
                    pool.release(sig_prev_values)

                # Convert back to Series
                equities = [pd.Series(e_values_array[i], index=index, dtype="float64") for i in range(9)]
            else:
                # Fallback to sequential calculation
                equities = [
                    equity_series(1.0, sig, rate_of_change_series, lambda_val=lambda_val, decay_val=decay_val)
                    for sig in signals
                ]
        except (ValueError, TypeError, MemoryError, RuntimeError) as e:
            # Fallback to sequential calculation on specific recoverable errors
            log_warn(f"Vectorized equity calculation failed ({type(e).__name__}: {e}), using sequential version")
            equities = [
                equity_series(1.0, sig, rate_of_change_series, lambda_val=lambda_val, decay_val=decay_val)
                for sig in signals
            ]

        # Unpack equities for return tuple (maintaining original variable names)
        E, E1, E2, E3, E4, E_1, E_2, E_3, E_4 = equities

        # Calculate weighted signal
        from .weighted_signal import weighted_signal

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
