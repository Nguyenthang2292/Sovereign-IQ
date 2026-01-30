"""Average signal calculation for ATC final output.

This module provides the function to calculate the final Average_Signal by
weighting Layer 1 signals with Layer 2 equity curves.
"""

from __future__ import annotations

from typing import Dict, cast

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.core.rust_backend import RUST_AVAILABLE

if RUST_AVAILABLE:
    try:
        from atc_rust import calculate_average_signal_cuda
    except ImportError:
        calculate_average_signal_cuda = None
else:
    calculate_average_signal_cuda = None

try:
    from modules.common.utils import log_debug, log_warn
except ImportError:
    # Fallback logging if common utils not available
    def log_debug(msg: str) -> None:  # pragma: no cover
        print(f"[DEBUG] {msg}")

    def log_warn(msg: str) -> None:  # pragma: no cover
        print(f"[WARN] {msg}")


def calculate_average_signal(
    layer1_signals: Dict[str, pd.Series],
    layer2_equities: Dict[str, pd.Series],
    ma_configs: list,
    prices: pd.Series,
    long_threshold: float,
    short_threshold: float,
    cutout: int = 0,
    strategy_mode: bool = False,
    precision: str = "float64",
    use_cuda: bool = False,
) -> pd.Series:
    """Calculate the final Average_Signal from Layer 1 signals and Layer 2 equities.

    This function performs vectorized computation of the weighted average signal
    where each Layer 1 signal is weighted by its corresponding Layer 2 equity curve.

    Args:
        layer1_signals: Dictionary of Layer 1 signals keyed by MA type.
        layer2_equities: Dictionary of Layer 2 equity curves keyed by MA type.
        ma_configs: List of (ma_type, length, initial_weight) tuples.
        prices: Price series (for index reference).
        long_threshold: Threshold for LONG signals.
        short_threshold: Threshold for SHORT signals.
        cutout: Number of bars to skip at beginning. Values before cutout are set to NaN
            (not 0.0) to be consistent with equity calculations and indicate "no valid data".
            Must be >= 0 and < len(prices). If cutout >= len(prices), a warning is issued
            and the entire series is returned as NaN.
        strategy_mode: If True, shift signal by 1 bar (default: False).

    Returns:
        Series containing the Average_Signal. Cutout period contains NaN values.

    Raises:
        ValueError: If cutout is negative.
    """
    # Validate cutout parameter
    n_bars = len(prices)
    if cutout < 0:
        raise ValueError(f"cutout must be >= 0, got {cutout}")
    if cutout >= n_bars:
        log_warn(f"cutout={cutout} >= n_bars={n_bars}. All bars will be set to NaN.")
        # Return NaN series
        return pd.Series(np.nan, index=prices.index, dtype=np.float32 if precision == "float32" else np.float64)
    log_debug("Computing Average_Signal (vectorized)...")

    index = prices.index
    n_bars = len(index)

    # Initialize accumulators (Task 8.5: Vectorized version)
    dtype = np.float32 if precision == "float32" else np.float64

    # Filter valid configs
    valid_configs = [
        (ma, length, weight) for ma, length, weight in ma_configs if ma in layer1_signals and ma in layer2_equities
    ]

    if not valid_configs:
        result = pd.Series(0.0, index=index, dtype=dtype)
        # Apply cutout NaN to maintain consistency with documented behavior
        if cutout > 0 and cutout < n_bars:
            result.iloc[:cutout] = np.nan
        return result

    # Convert to 2D matrices for broadcasting using pandas for strict alignment
    # Task 8.5: Performance optimized alignment
    s_list = []
    e_list = []

    for ma_type, _, _ in valid_configs:
        # Align each component to the main index (prices.index)
        # MUST NOT fillna(0.0) here to match Original NaN propagation logic
        s_aligned = layer1_signals[ma_type].reindex(index)
        e_aligned = layer2_equities[ma_type].reindex(index)

        s_list.append(s_aligned.values)
        e_list.append(e_aligned.values)

    # Shape: (n_mas, n_bars)
    S_np = np.stack(s_list)
    E_np = np.stack(e_list)

    # Initialize result array with placeholder
    avg_signal_array: np.ndarray = np.array([])
    cuda_failed = False

    # Check for NaN values before calculation
    if np.any(np.isnan(S_np)):
        nan_count = np.sum(np.isnan(S_np))
        log_warn(f"Layer 1 signals contain {nan_count} NaN values before calculation")
    if np.any(np.isnan(E_np)):
        nan_count = np.sum(np.isnan(E_np))
        log_warn(f"Layer 2 equities contain {nan_count} NaN values before calculation")

    # Try CUDA path first if available
    cuda_failed = False
    if use_cuda and calculate_average_signal_cuda is not None:
        try:
            # Calculate final average using CUDA kernel
            # Note: CUDA kernel receives cutout parameter but may not apply it internally
            cuda_result = calculate_average_signal_cuda(
                S_np.astype(np.float64),
                E_np.astype(np.float64),
                float(long_threshold),
                float(short_threshold),
                int(cutout),
            )
            # Handle potential NaN/inf from CUDA - replace with 0.0 (neutral signal)
            avg_signal_array = np.where(np.isfinite(cuda_result), cuda_result, 0.0)
        except Exception as e:
            log_warn(f"CUDA Average Signal failed, falling back to CPU: {e}")
            cuda_failed = True

    # CPU fallback path (if CUDA not used or failed)
    if not use_cuda or calculate_average_signal_cuda is None or cuda_failed:
        # NaN detection and handling
        # Check for NaN values in inputs that could affect calculation
        s_nan_mask = np.isnan(S_np)
        e_nan_mask = np.isnan(E_np)

        if np.any(s_nan_mask):
            nan_count = np.sum(s_nan_mask)
            log_warn(f"Layer 1 signals contain {nan_count} NaN values, treating as neutral (0.0)")

        if np.any(e_nan_mask):
            nan_count = np.sum(e_nan_mask)
            log_warn(f"Layer 2 equities contain {nan_count} NaN values, replacing with 0.0")
            # Replace NaN equities with 0 to prevent them from affecting weighted average
            E_np = np.where(e_nan_mask, 0.0, E_np)

        # Vectorized discretization (Task 8.5)
        # NaN in S_np will result in False for both comparisons, giving 0.0 (neutral)
        with np.errstate(invalid="ignore"):
            C = np.where(S_np > long_threshold, 1.0, np.where(S_np < short_threshold, -1.0, 0.0))

        # Parallel calculation across components
        # nom = sum(signal_discrete * equity_weight), den = sum(equity_weight)
        nom_array = np.sum(C * E_np, axis=0)
        den_array = np.sum(E_np, axis=0)

        # Handle zero denominator case (when all equity weights are zero)
        zero_den_mask = den_array == 0
        if np.any(zero_den_mask):
            zero_count = np.sum(zero_den_mask)
            log_warn(f"Sum of equity weights is zero for {zero_count} bars, returning neutral signal (0.0)")
            # Replace zero denominators with 1.0 to avoid division by zero
            # Since nominator is also 0 when all equities are 0, result will be 0/1 = 0 (neutral)
            den_array = np.where(zero_den_mask, 1.0, den_array)

        # Calculate final average (no special error handling needed now)
        avg_signal_array = nom_array / den_array

    # avg_signal_array is guaranteed to be assigned at this point
    assert avg_signal_array is not None, "avg_signal_array should be assigned by CUDA or CPU path"

    # Apply cutout to average signal array for both CUDA and CPU paths
    # Use NaN (not 0.0) for cutout period to be consistent with equity calculations
    # and to indicate "no valid data" rather than "neutral signal"
    # Note: We apply this unconditionally as CUDA kernel may not handle cutout internally
    if cutout > 0 and cutout < n_bars:
        avg_signal_array[:cutout] = np.nan

    # Create series from the result array
    result_series = pd.Series(avg_signal_array, index=index, dtype=dtype)

    if strategy_mode:
        # Strategy mode: Delay signal by 1 bar to avoid repainting (Pine Script behavior)
        # NOTE: This is the ONLY shift applied to the final output signal.
        # Any shifts in Layer 1/2 equity calculations are INTERNAL only and do not
        # affect the signals passed to this function. There is NO "double shift" bug.
        # PRESERVE NaN VALUES: Cutout period should remain NaN (not 0.0)

        # FIX: Do not fillna(0) for indices where result_series was NaN (data missing)
        # only fillna(0) for the very first bar created by shift() if it's not in cutout
        shifted = result_series.shift(1)

        # Identify where we had valid data but shift() created a NaN (the first valid bar)
        valid_mask = result_series.notna()
        shifted_nan_mask = shifted.isna()

        # Bars that were valid but now NaN after shift and NOT in cutout
        to_fill_mask = valid_mask & shifted_nan_mask
        if cutout > 0:
            # exclude cutout range from filling
            cutout_indices = index[:cutout]
            to_fill_mask.loc[cutout_indices] = False

        result_series = shifted.copy()
        result_series.loc[to_fill_mask] = 0.0

    log_debug("Completed Average_Signal")
    return cast(pd.Series, result_series)
