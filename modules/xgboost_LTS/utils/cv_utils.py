"""Shared cross-validation utilities for XGBoost LTS."""

import numpy as np


def apply_cv_gap(train_idx: np.ndarray, test_idx: np.ndarray, gap: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply time-series gap between train and test indices to prevent target leakage.

    Args:
        train_idx: Training indices from a CV split
        test_idx: Test indices from a CV split
        gap: Number of trailing training rows to remove as leakage buffer

    Returns:
        Tuple of (filtered_train_idx, filtered_test_idx). Arrays may be empty if
        there is not enough data after applying the gap.
    """
    train_idx_array = np.asarray(train_idx)
    test_idx_array = np.asarray(test_idx)

    if gap < 0:
        raise ValueError(f"gap must be >= 0, got {gap}")

    if len(train_idx_array) == 0 or len(test_idx_array) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    if gap > 0:
        if len(train_idx_array) <= gap:
            return np.array([], dtype=int), np.array([], dtype=int)
        train_idx_filtered = train_idx_array[:-gap]
    else:
        train_idx_filtered = train_idx_array

    if len(train_idx_filtered) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    min_test_start = train_idx_filtered[-1] + gap + 1
    test_idx_filtered = test_idx_array[test_idx_array >= min_test_start]

    return train_idx_filtered, test_idx_filtered
