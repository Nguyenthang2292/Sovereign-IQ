"""Weighted signal calculation for Layer 1 processing.

This module provides the weighted_signal function to calculate weighted
average signal from multiple signals and weights.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from modules.common.system import get_memory_manager
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

        # Convert matching series to a 2D matrix for vectorized math (Task 8.5)
        # Shape: (n_components, n_bars)
        s_matrix = np.stack([sig.values for sig in signals])
        w_matrix = np.stack([wgt.values for wgt in weights])

        # num = sum(s * w), den = sum(w)
        num_arr = np.sum(s_matrix * w_matrix, axis=0)
        den_arr = np.sum(w_matrix, axis=0)

        # Handle division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            res_arr = np.divide(num_arr, den_arr)
            res_arr = np.where(np.isfinite(res_arr), res_arr, np.nan)

        result = pd.Series(res_arr, index=first_index, dtype="float64").round(2)
        return result
