"""
Unit tests for Pattern Matcher.

Tests the threshold-based pattern matching logic to ensure it matches Pine Script behavior.
"""

import pytest

from modules.decision_matrix.utils.pattern_matcher import PatternMatcher


class TestPatternMatcher:
    """Test PatternMatcher functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.matcher = PatternMatcher()

    def test_match_patterns_x1_check_positive_true(self):
        """Test X1 pattern matching with check_positive=True (Pine Script line 121)."""
        # Training data: [[value, label], ...]
        training_matrix = [
            [0.5, 1],  # Within threshold, label=1
            [0.6, 0],  # Within threshold, label=0
            [-0.1, 1],  # Negative value, should be skipped
            [0.0, 1],  # Zero value, should be skipped
            [1.5, 1],  # Outside threshold
            [0.55, 1],  # Within threshold, label=1
            [0.45, 0],  # Last row - should be skipped by [:-1]
        ]

        current_value = 0.5
        threshold = 0.1  # Match range: [0.4, 0.6]

        # With check_positive=True (X1 logic)
        pass_count, fail_count = self.matcher.match_patterns(
            training_matrix, current_value, threshold, check_positive=True
        )

        # Expected matches (skipping last row and non-positive values):
        # [0.5, 1] -> pass (within [0.4, 0.6], label=1)
        # [0.6, 0] -> fail (within [0.4, 0.6], label=0)
        # [-0.1, 1] -> skipped (negative)
        # [0.0, 1] -> skipped (zero)
        # [1.5, 1] -> not matched (outside threshold)
        # [0.55, 1] -> pass (within [0.4, 0.6], label=1)
        # [0.45, 0] -> skipped (last row)

        assert pass_count == 2  # [0.5, 1] and [0.55, 1]
        assert fail_count == 1  # [0.6, 0]

    def test_match_patterns_x2_check_positive_false(self):
        """Test X2 pattern matching with check_positive=False (Pine Script line 131)."""
        # Training data with negative and zero values
        training_matrix = [
            [0.5, 1],  # Within threshold, label=1
            [0.6, 0],  # Within threshold, label=0
            [-0.1, 1],  # Negative value, should still be checked for X2
            [0.0, 1],  # Zero value, should still be checked for X2
            [1.5, 1],  # Outside threshold
            [0.55, 1],  # Within threshold, label=1
            [0.45, 0],  # Last row - should be skipped by [:-1]
        ]

        current_value = 0.5
        threshold = 0.1  # Match range: [0.4, 0.6]

        # With check_positive=False (X2 logic)
        pass_count, fail_count = self.matcher.match_patterns(
            training_matrix, current_value, threshold, check_positive=False
        )

        # Expected matches (skipping last row only, NOT skipping negative/zero):
        # [0.5, 1] -> pass
        # [0.6, 0] -> fail (at boundary)
        # [-0.1, 1] -> not matched (outside threshold)
        # [0.0, 1] -> not matched (outside threshold)
        # [1.5, 1] -> not matched (outside threshold)
        # [0.55, 1] -> pass
        # [0.45, 0] -> skipped (last row)

        assert pass_count == 2  # [0.5, 1] and [0.55, 1]
        assert fail_count == 1  # [0.6, 0]

    def test_match_patterns_skip_last_row(self):
        """Test that pattern matching skips last row (Pine Script train - 2)."""
        # Simple training data
        training_matrix = [
            [0.5, 1],  # Match
            [0.6, 1],  # Match
            [0.7, 1],  # This is the last row, should be skipped
        ]

        current_value = 0.6
        threshold = 0.2  # Match range: [0.4, 0.8]

        pass_count, fail_count = self.matcher.match_patterns(
            training_matrix, current_value, threshold, check_positive=False
        )

        # Should only count first 2 rows (skip last)
        assert pass_count == 2
        assert fail_count == 0

    def test_match_patterns_empty_matrix(self):
        """Test pattern matching with empty training matrix."""
        training_matrix = []
        current_value = 0.5
        threshold = 0.1

        pass_count, fail_count = self.matcher.match_patterns(
            training_matrix, current_value, threshold, check_positive=True
        )

        assert pass_count == 0
        assert fail_count == 0

    def test_match_patterns_threshold_boundaries(self):
        """Test pattern matching at threshold boundaries."""
        training_matrix = [
            [0.4, 1],  # Exactly at lower boundary
            [0.6, 0],  # Exactly at upper boundary
            [0.39, 1],  # Just below lower boundary
            [0.61, 0],  # Just above upper boundary
            [0.5, 1],  # Exactly at current value
            [999, 1],  # Last row - skipped
        ]

        current_value = 0.5
        threshold = 0.1  # Match range: [0.4, 0.6]

        pass_count, fail_count = self.matcher.match_patterns(
            training_matrix, current_value, threshold, check_positive=False
        )

        # Should match: [0.4, 1], [0.6, 0], [0.5, 1]
        assert pass_count == 2  # [0.4, 1] and [0.5, 1]
        assert fail_count == 1  # [0.6, 0]

    def test_match_all_features(self):
        """Test matching patterns for both X1 and X2 features."""
        x1_matrix = [
            [50, 1],  # Match X1
            [60, 0],  # Match X1
            [-10, 1],  # Skipped for X1 (negative)
            [999, 1],  # Last row - skipped
        ]

        x2_matrix = [
            [100, 1],  # Match X2
            [120, 0],  # Match X2
            [-50, 1],  # Match X2 (negative OK for X2)
            [999, 1],  # Last row - skipped
        ]

        y1_pass, y1_fail, y2_pass, y2_fail = self.matcher.match_all_features(
            x1_matrix, x2_matrix, x1_value=55, x2_value=110, x1_threshold=10, x2_threshold=20
        )

        # X1 (check_positive=True, range [45, 65]):
        # [50, 1] -> pass
        # [60, 0] -> fail
        # [-10, 1] -> skipped (negative)
        assert y1_pass == 1
        assert y1_fail == 1

        # X2 (check_positive=False, range [90, 130]):
        # [100, 1] -> pass
        # [120, 0] -> fail
        # [-50, 1] -> not matched (outside range)
        assert y2_pass == 1
        assert y2_fail == 1

    def test_match_patterns_all_positive_labels(self):
        """Test pattern matching when all matched labels are 1."""
        training_matrix = [
            [0.5, 1],
            [0.55, 1],
            [0.6, 1],
            [999, 0],  # Last row - skipped
        ]

        current_value = 0.55
        threshold = 0.1

        pass_count, fail_count = self.matcher.match_patterns(
            training_matrix, current_value, threshold, check_positive=False
        )

        assert pass_count == 3
        assert fail_count == 0

    def test_match_patterns_all_negative_labels(self):
        """Test pattern matching when all matched labels are 0."""
        training_matrix = [
            [0.5, 0],
            [0.55, 0],
            [0.6, 0],
            [999, 1],  # Last row - skipped
        ]

        current_value = 0.55
        threshold = 0.1

        pass_count, fail_count = self.matcher.match_patterns(
            training_matrix, current_value, threshold, check_positive=False
        )

        assert pass_count == 0
        assert fail_count == 3

    def test_match_patterns_no_matches_within_threshold(self):
        """Test pattern matching when no values fall within threshold."""
        training_matrix = [
            [1.0, 1],
            [2.0, 1],
            [3.0, 0],
            [999, 1],  # Last row - skipped
        ]

        current_value = 0.5
        threshold = 0.1  # Range: [0.4, 0.6]

        pass_count, fail_count = self.matcher.match_patterns(
            training_matrix, current_value, threshold, check_positive=False
        )

        assert pass_count == 0
        assert fail_count == 0
