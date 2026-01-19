"""
Random Forest Pattern Matching Core (Pine Script Algorithm).

This module implements a pattern-based classification algorithm from Pine Script.
It is NOT the sklearn RandomForestClassifier - it uses pattern matching and voting mechanisms.

⚠️ IMPORTANT: This is DIFFERENT from modules.random_forest
- This module (modules.decision_matrix.core.random_forest_core): Pine Script pattern matching algorithm
- modules.random_forest: sklearn RandomForestClassifier wrapper (ML-based signals)

This algorithm:
- Stores training data as matrices [feature, label]
- Shuffles data randomly
- Matches patterns using thresholds
- Votes based on pass/fail counts
- Calculates accuracy metrics

For sklearn-based ML Random Forest, see modules.random_forest instead.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np

from modules.common.utils import log_debug
from modules.decision_matrix.utils.pattern_matcher import PatternMatcher
from modules.decision_matrix.utils.shuffle import ShuffleMechanism


@dataclass
class RandomForestCore:
    """
    Random Forest Pattern Matching Algorithm (Pine Script Implementation).

    ⚠️ NOTE: This is NOT sklearn's RandomForestClassifier.
    This is a pattern-based classification algorithm from Pine Script.

    For sklearn-based ML Random Forest, use modules.random_forest instead.

    Based on Pine Script implementation:
    - Stores training data as matrices [feature, label]
    - Shuffles data randomly
    - Matches patterns using thresholds
    - Votes based on pass/fail counts
    - Calculates accuracy metrics

    Attributes:
        training_length: Number of historical bars for training (default: 850)
        random_seed: Optional seed for reproducible random shuffling.
                     If None, uses unpredictable random state.
    """

    training_length: int = 850
    random_seed: Optional[int] = None
    pattern_matcher: PatternMatcher = field(default_factory=PatternMatcher)
    shuffle_mechanism: ShuffleMechanism = field(default_factory=lambda: ShuffleMechanism())

    def __post_init__(self):
        """Initialize shuffle mechanism with seed if provided."""
        # Type validation for random_seed
        if self.random_seed is not None and not isinstance(self.random_seed, int):
            raise TypeError(f"random_seed must be int or None, got {type(self.random_seed)}")
        
        if self.random_seed is not None:
            self.shuffle_mechanism = ShuffleMechanism(seed=self.random_seed)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"RandomForestCore(training_length={self.training_length}, seed={self.random_seed})"

    def classify(
        self,
        x1_matrix: Union[List[List[float]], np.ndarray],
        x2_matrix: Union[List[List[float]], np.ndarray],
        current_x1: float,
        current_x2: float,
        x1_threshold: float,
        x2_threshold: float,
    ) -> dict:
        """
        Perform Random Forest classification.

        Args:
            x1_matrix: Training data for feature 1 [[value, label], ...]
            x2_matrix: Training data for feature 2 [[value, label], ...]
            current_x1: Current value for feature 1
            current_x2: Current value for feature 2
            x1_threshold: Threshold for feature 1 matching
            x2_threshold: Threshold for feature 2 matching

        Returns:
            Dictionary with classification results:
            - vote: Final prediction (0 or 1)
            - accuracy: Model accuracy percentage
            - y1_pass, y1_fail: Pass/fail counts for feature 1
            - y2_pass, y2_fail: Pass/fail counts for feature 2
            - x1_vote, x2_vote: Individual votes for each feature
            - x1_accuracy, x2_accuracy: Individual feature accuracies
        """
        # Convert inputs to numpy arrays for vectorization (once at the start)
        # np.asarray is faster than np.array if already numpy array
        x1_arr = np.asarray(x1_matrix)
        x2_arr = np.asarray(x2_matrix)

        train = len(x1_arr)

        if train == 0:
            return self._empty_result()

        x1m_random = self.shuffle_mechanism.shuffle_matrix(x1_arr, train)
        x2m_random = self.shuffle_mechanism.shuffle_matrix(x2_arr, train)

        # Pre-compute threshold bounds for vectorization
        # Vectorized bounds calculation: [lower_bound, upper_bound] for each feature
        x1_bounds = np.array([current_x1 - x1_threshold, current_x1 + x1_threshold])
        x2_bounds = np.array([current_x2 - x2_threshold, current_x2 + x2_threshold])

        y1_pass, y1_fail = self.pattern_matcher.match_patterns(
            x1m_random, current_x1, x1_threshold, check_positive=True, bounds=x1_bounds
        )
        y2_pass, y2_fail = self.pattern_matcher.match_patterns(
            x2m_random, current_x2, x2_threshold, check_positive=False, bounds=x2_bounds
        )

        x1_vote = 1 if y1_pass > y1_fail else 0
        x2_vote = 1 if y2_pass > y2_fail else 0

        passes = y1_pass + y2_pass
        fails = y1_fail + y2_fail
        vote = 1 if passes > fails else 0

        # Calculate accuracies for both features in one batch operation
        x1_accuracy, x2_accuracy = self._calculate_accuracies_batch(
            [x1m_random, x2m_random],
            [x1_vote, x2_vote],
        )

        overall_accuracy = self._calculate_overall_accuracy(x1m_random, x2m_random, vote)

        log_debug(f"Classifying with {len(x1_arr)} training samples")
        log_debug(f"y1_pass={y1_pass}, y1_fail={y1_fail}")
        log_debug(f"Final vote={vote}, accuracy={overall_accuracy:.2f}%")

        return {
            "vote": vote,
            "accuracy": overall_accuracy,
            "y1_pass": y1_pass,
            "y1_fail": y1_fail,
            "y2_pass": y2_pass,
            "y2_fail": y2_fail,
            "x1_vote": x1_vote,
            "x2_vote": x2_vote,
            "x1_accuracy": x1_accuracy,
            "x2_accuracy": x2_accuracy,
        }

    def _calculate_accuracy(
        self,
        training_matrix: Union[List[List[float]], np.ndarray],
        vote: int,
    ) -> float:
        """
        Calculate individual feature accuracy using vectorization.

        Args:
            training_matrix: Training data [[value, label], ...]
            vote: Feature vote (0 or 1)

        Returns:
            Accuracy percentage (0-100)
        """
        # Convert to numpy array if needed (e.g. called from tests)
        if not isinstance(training_matrix, np.ndarray):
            matrix = np.array(training_matrix)
        else:
            matrix = training_matrix

        if len(matrix) == 0:
            return 0.0

        # Vectorized comparison: check matching labels
        labels = matrix[:, 1]

        # Count matches (pass) and mismatches (fail)
        pass_count = np.sum(labels == vote)
        total = len(labels)

        if total == 0:
            return 0.0

        return (pass_count / total) * 100

    def _calculate_accuracies_batch(
        self,
        matrices: List[np.ndarray],
        votes: List[int],
    ) -> List[float]:
        """
        Calculate accuracies for multiple features in one vectorized pass.

        Args:
            matrices: List of training matrices [[value, label], ...]
            votes: List of feature votes (0 or 1)

        Returns:
            List of accuracy percentages (0-100)
        """
        accuracies = []
        for matrix, vote in zip(matrices, votes):
            if len(matrix) == 0:
                accuracies.append(0.0)
            else:
                # Vectorized: extract labels and count matches
                labels = matrix[:, 1]
                pass_count = np.sum(labels == vote)
                total = len(labels)
                accuracy = (pass_count / total) * 100 if total > 0 else 0.0
                accuracies.append(accuracy)
        return accuracies

    def _calculate_overall_accuracy(
        self,
        x1_matrix: Union[List[List[float]], np.ndarray],
        x2_matrix: Union[List[List[float]], np.ndarray],
        vote: int,
    ) -> float:
        """
        Calculate overall model accuracy using vectorization.

        Args:
            x1_matrix: Training data for feature 1 [[value, label], ...]
            x2_matrix: Training data for feature 2 [[value, label], ...]
            vote: Final vote (0 or 1)

        Returns:
            Overall accuracy percentage (0-100)
        """
        # Convert to numpy array if needed
        if not isinstance(x1_matrix, np.ndarray):
            x1_arr = np.array(x1_matrix)
        else:
            x1_arr = x1_matrix

        if len(x1_arr) == 0:
            return 0.0

        # Vectorized comparison: check matching labels
        # Use x1_matrix labels (same as x2_matrix labels since they're paired)
        labels = x1_arr[:, 1]

        success = np.sum(labels == vote)
        total = len(labels)

        if total == 0:
            return 0.0

        return (success / total) * 100

    def _empty_result(self) -> dict:
        """Return empty result when no training data."""
        return {
            "vote": 0,
            "accuracy": 0.0,
            "y1_pass": 0,
            "y1_fail": 0,
            "y2_pass": 0,
            "y2_fail": 0,
            "x1_vote": 0,
            "x2_vote": 0,
            "x1_accuracy": 0.0,
            "x2_accuracy": 0.0,
        }


__all__ = ["RandomForestCore"]
