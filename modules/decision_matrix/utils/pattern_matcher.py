from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class PatternMatcher:
    """
    Match current feature values with historical patterns using thresholds.

    Finds historical data points within ±threshold of current value.
    """

    @staticmethod
    @lru_cache(maxsize=128)
    def _to_numpy(data_tuple: Tuple[Tuple[float, ...], ...]) -> np.ndarray:
        """
        Convert tuple of tuples to numpy array with caching.

        Args:
            data_tuple: Hashable tuple representation of training data

        Returns:
            Numpy array of training data
        """
        return np.array(data_tuple)

    def match_patterns(
        self,
        training_matrix: Union[List[List[float]], np.ndarray],
        current_value: float,
        threshold: float,
        check_positive: bool = True,
        bounds: Optional[np.ndarray] = None,
    ) -> Tuple[int, int]:
        """
        Match patterns and count pass/fail.

        Args:
            training_matrix: Shuffled training data as numpy array or list [[value, label], ...]
            current_value: Current feature value to match
            threshold: Matching threshold
            check_positive: Whether to check if value > 0 (for volume)
            bounds: Pre-computed bounds array [lower_bound, upper_bound]. If provided,
                   skips threshold calculation.

        Returns:
            Tuple of (pass_count, fail_count) for matched patterns
        """
        # Convert to numpy if needed (with caching for list inputs)
        if isinstance(training_matrix, np.ndarray):
            matrix_arr = training_matrix
        else:
            # Convert list to hashable tuple for caching
            # Only cache if data is small enough (less than 1000 rows) to avoid memory issues
            if len(training_matrix) < 1000:
                data_tuple = tuple(tuple(row) for row in training_matrix)
                matrix_arr = self._to_numpy(data_tuple)
            else:
                matrix_arr = np.array(training_matrix)

        if len(matrix_arr) == 0:
            return 0, 0

        # Pine Script loops from 0 to train - 2 (skipping the last one)
        # So we slice [:-1]
        data_to_check = matrix_arr[:-1]

        if len(data_to_check) == 0:
            return 0, 0

        # Extract values (col 0) and labels (col 1)
        values = data_to_check[:, 0]
        labels = data_to_check[:, 1]

        # Use pre-computed bounds if provided, otherwise compute
        if bounds is not None and len(bounds) >= 2:
            lower_bound, upper_bound = bounds[0], bounds[1]
        else:
            lower_bound = current_value - threshold
            upper_bound = current_value + threshold

        # Build mask directly with required conditions
        # This avoids creating an initial ones array and is more efficient
        if check_positive:
            mask = (values > 0) & (values >= lower_bound) & (values <= upper_bound)
        else:
            mask = (values >= lower_bound) & (values <= upper_bound)

        # Filter labels based on mask
        matched_labels = labels[mask]

        # Count passes (label == 1) and fails (label != 1)
        # Assuming label is 0 or 1, pass_count is sum of matched_labels (if 1s)
        # But to be safe (if labels are just != 1 for fail), we explicitly check
        pass_count = np.sum(matched_labels == 1)
        # fail count is the rest
        fail_count = len(matched_labels) - pass_count

        return int(pass_count), int(fail_count)

    def match_all_features(
        self,
        x1_matrix: Union[List[List[float]], np.ndarray],
        x2_matrix: Union[List[List[float]], np.ndarray],
        x1_value: float,
        x2_value: float,
        x1_threshold: float,
        x2_threshold: float,
    ) -> Tuple[int, int, int, int]:
        """
        Match patterns for both features.

        Args:
            x1_matrix: Training data for feature 1
            x2_matrix: Training data for feature 2
            x1_value: Current value for feature 1
            x2_value: Current value for feature 2
            x1_threshold: Threshold for feature 1
            x2_threshold: Threshold for feature 2

        Returns:
            Tuple of (y1_pass, y1_fail, y2_pass, y2_fail)
        """
        # Pre-compute bounds for vectorization
        x1_bounds = np.array([x1_value - x1_threshold, x1_value + x1_threshold])
        x2_bounds = np.array([x2_value - x2_threshold, x2_value + x2_threshold])

        y1_pass, y1_fail = self.match_patterns(
            x1_matrix, x1_value, x1_threshold, check_positive=True, bounds=x1_bounds
        )
        y2_pass, y2_fail = self.match_patterns(
            x2_matrix, x2_value, x2_threshold, check_positive=False, bounds=x2_bounds
        )

        return y1_pass, y1_fail, y2_pass, y2_fail


__all__ = ["PatternMatcher"]
