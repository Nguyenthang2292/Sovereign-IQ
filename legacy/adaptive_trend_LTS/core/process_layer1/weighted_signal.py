"""Weighted signal calculation for Layer 1 processing.

This module provides the weighted_signal function to calculate weighted
average signal from multiple signals and weights.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from modules.common.utils import log_warn


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
            # Check for NaN values introduced by reindex
            nan_count = signals[i].isna().sum()
            if nan_count > 0:
                log_warn(f"signals[{i}] has {nan_count} NaN values after alignment, may affect calculation")
        if not wgt.index.equals(first_index):
            log_warn(f"weights[{i}] has different index, aligning...")
            weights[i] = wgt.reindex(first_index)
            # Check for NaN values introduced by reindex
            nan_count = weights[i].isna().sum()
            if nan_count > 0:
                log_warn(f"weights[{i}] has {nan_count} NaN values after alignment, may affect calculation")

    # Convert matching series to a 2D matrix for vectorized math (Task 8.5)
    # Shape: (n_components, n_bars)
    # Explicitly convert to list of ndarrays to satisfy type checker
    s_arrays: list[np.ndarray] = [np.array(sig.values, dtype=np.float64) for sig in signals]
    w_arrays: list[np.ndarray] = [np.array(wgt.values, dtype=np.float64) for wgt in weights]
    s_matrix = np.stack(s_arrays)
    w_matrix = np.stack(w_arrays)

    # num = sum(s * w), den = sum(w)
    num_arr = np.sum(s_matrix * w_matrix, axis=0)
    den_arr = np.sum(w_matrix, axis=0)

    # Handle zero denominator case (when all weights are zero)
    zero_mask = den_arr == 0
    if np.any(zero_mask):
        zero_count = np.sum(zero_mask)
        log_warn(f"Sum of weights is zero for {zero_count} bars, returning neutral signal (0.0)")
        # Replace zero denominators with 1.0 to avoid division by zero
        # Since numerator is also 0 when all weights are 0, result will be 0/1 = 0 (neutral)
        den_arr = np.where(zero_mask, 1.0, den_arr)

    # Calculate weighted average (no special error handling needed now)
    res_arr = num_arr / den_arr

    result = pd.Series(res_arr, index=first_index, dtype="float64").round(2)
    return result
