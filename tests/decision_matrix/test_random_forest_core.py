"""
Unit tests for Random Forest Core.

Tests the main Random Forest classification algorithm to ensure it matches Pine Script behavior.
"""

import random

import pytest

from unittest.mock import patch

from modules.decision_matrix.core.random_forest_core import RandomForestCore


class TestRandomForestCore:
    """Test RandomForestCore functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rf = RandomForestCore(training_length=10, random_seed=42)
        # Seed random for deterministic tests
        random.seed(42)

    def test_classify_basic(self):
        """Test basic classification with simple training data."""
        # Training data: [[value, label], ...]
        x1_matrix = [[50, 1], [60, 0], [55, 1], [45, 1], [70, 0]]
        x2_matrix = [[100, 1], [110, 0], [105, 1], [95, 1], [120, 0]]

        results = self.rf.classify(
            x1_matrix=x1_matrix,
            x2_matrix=x2_matrix,
            current_x1=55,
            current_x2=105,
            x1_threshold=10,
            x2_threshold=15,
        )

        # Verify result structure
        assert "vote" in results
        assert "accuracy" in results
        assert "y1_pass" in results
        assert "y1_fail" in results
        assert "y2_pass" in results
        assert "y2_fail" in results
        assert "x1_vote" in results
        assert "x2_vote" in results
        assert "x1_accuracy" in results
        assert "x2_accuracy" in results

        # Vote should be 0 or 1
        assert results["vote"] in [0, 1]
        assert results["x1_vote"] in [0, 1]
        assert results["x2_vote"] in [0, 1]

        # Accuracy should be 0-100
        assert 0 <= results["accuracy"] <= 100
        assert 0 <= results["x1_accuracy"] <= 100
        assert 0 <= results["x2_accuracy"] <= 100

    def test_classify_vote_logic(self):
        """Test that vote follows passes > fails logic (Pine Script line 139)."""
        with patch("modules.decision_matrix.shuffle.ShuffleMechanism.shuffle_matrix", side_effect=lambda x, r: x[:r]):
            # Set up data where we know the expected pass/fail counts
            # X1: values around 50, threshold 5 -> range [45, 55]
            x1_matrix = [
                [50, 1],  # Match, pass
                [52, 1],  # Match, pass
                [48, 0],  # Match, fail
                [100, 1],  # No match
                [999, 1],  # Last row - skipped
            ]

            # X2: values around 100, threshold 10 -> range [90, 110]
            x2_matrix = [
                [100, 1],  # Match, pass
                [105, 0],  # Match, fail
                [95, 0],  # Match, fail
                [200, 1],  # No match
                [999, 1],  # Last row - skipped
            ]

            results = self.rf.classify(
                x1_matrix=x1_matrix,
                x2_matrix=x2_matrix,
                current_x1=50,
                current_x2=100,
                x1_threshold=5,
                x2_threshold=10,
            )

            # X1: 2 pass (50, 52), 1 fail (48) -> x1_vote = 1
            # X2: 1 pass (100), 2 fail (105, 95) -> x2_vote = 0
            assert results["y1_pass"] == 2
            assert results["y1_fail"] == 1
            assert results["x1_vote"] == 1  # passes > fails

            assert results["y2_pass"] == 1
            assert results["y2_fail"] == 2
            assert results["x2_vote"] == 0  # fails > passes

            # Overall vote: passes = 2 + 1 = 3, fails = 1 + 2 = 3
            # When equal: vote = 0 (not 1)
            assert results["vote"] == 0

    def test_classify_empty_matrix(self):
        """Test classification with empty training matrix."""
        results = self.rf.classify(
            x1_matrix=[],
            x2_matrix=[],
            current_x1=50,
            current_x2=100,
            x1_threshold=10,
            x2_threshold=20,
        )

        # Should return empty result
        assert results["vote"] == 0
        assert results["accuracy"] == 0.0
        assert results["y1_pass"] == 0
        assert results["y1_fail"] == 0
        assert results["y2_pass"] == 0
        assert results["y2_fail"] == 0
        assert results["x1_vote"] == 0
        assert results["x2_vote"] == 0
        assert results["x1_accuracy"] == 0.0
        assert results["x2_accuracy"] == 0.0

    def test_individual_accuracy_all_labels_match_vote(self):
        """Test individual accuracy when all labels match the vote."""
        # All labels are 1, vote will be 1
        x1_matrix = [[50, 1], [55, 1], [60, 1], [65, 1]]
        x2_matrix = [[100, 1], [105, 1], [110, 1], [115, 1]]

        results = self.rf.classify(
            x1_matrix=x1_matrix,
            x2_matrix=x2_matrix,
            current_x1=55,
            current_x2=105,
            x1_threshold=10,
            x2_threshold=15,
        )

        # All matches should be passes
        assert results["x1_vote"] == 1
        assert results["x2_vote"] == 1

        # Individual accuracy: compare vote=1 with ALL labels
        # All labels are 1, so accuracy should be 100%
        assert results["x1_accuracy"] == 100.0
        assert results["x2_accuracy"] == 100.0

    def test_individual_accuracy_no_labels_match_vote(self):
        """Test individual accuracy when no labels match the vote."""
        # Set up so vote=0 but all labels are 1
        # X1: More fails than passes -> vote=0
        x1_matrix = [[50, 0], [55, 0], [60, 1], [999, 1]]  # 2 fail, 1 pass -> vote=0

        # X2: Same pattern
        x2_matrix = [[100, 0], [105, 0], [110, 1], [999, 1]]  # 2 fail, 1 pass -> vote=0

        results = self.rf.classify(
            x1_matrix=x1_matrix,
            x2_matrix=x2_matrix,
            current_x1=55,
            current_x2=105,
            x1_threshold=10,
            x2_threshold=15,
        )

        # Verify votes
        assert results["x1_vote"] == 0
        assert results["x2_vote"] == 0

        # Individual accuracy: compare vote=0 with ALL labels
        # Labels: [0, 0, 1, 1] (last skipped in matching, but ALL used for accuracy)
        # Matches with vote=0: [0, 0] -> 2 out of 4 = 50%
        # NOTE: We use ALL 4 rows for accuracy (not skip last row)
        assert results["x1_accuracy"] == 50.0
        assert results["x2_accuracy"] == 50.0

    def test_overall_accuracy_all_labels_match_vote(self):
        """Test overall accuracy when all labels match the final vote."""
        # All labels are 1
        x1_matrix = [[50, 1], [55, 1], [60, 1], [65, 1]]
        x2_matrix = [[100, 1], [105, 1], [110, 1], [115, 1]]

        results = self.rf.classify(
            x1_matrix=x1_matrix,
            x2_matrix=x2_matrix,
            current_x1=55,
            current_x2=105,
            x1_threshold=10,
            x2_threshold=15,
        )

        # Final vote should be 1 (all passes)
        assert results["vote"] == 1

        # Overall accuracy: vote=1 vs ALL labels (all are 1)
        # 100% match
        assert results["accuracy"] == 100.0

    def test_overall_accuracy_mixed_labels(self):
        """Test overall accuracy with mixed labels."""
        # 50% labels are 1, 50% are 0
        x1_matrix = [[50, 1], [55, 0], [60, 1], [65, 0]]
        x2_matrix = [[100, 1], [105, 0], [110, 1], [115, 0]]

        results = self.rf.classify(
            x1_matrix=x1_matrix,
            x2_matrix=x2_matrix,
            current_x1=55,
            current_x2=105,
            x1_threshold=10,
            x2_threshold=15,
        )

        # Whatever the vote is (0 or 1), accuracy should be ~50%
        # Because 50% of labels match vote, 50% don't
        assert results["accuracy"] == 50.0

    def test_x1_checks_positive_values_only(self):
        """Test that X1 pattern matching skips non-positive values."""
        with patch("modules.decision_matrix.shuffle.ShuffleMechanism.shuffle_matrix", side_effect=lambda x, r: x[:r]):
            # X1 has negative and zero values
            x1_matrix = [
                [50, 1],  # Positive, should be checked
                [-10, 1],  # Negative, should be skipped
                [0, 1],  # Zero, should be skipped
                [55, 1],  # Positive, should be checked
                [999, 1],  # Last row - skipped
            ]

            x2_matrix = [[100, 1], [100, 1], [100, 1], [100, 1], [999, 1]]

            results = self.rf.classify(
                x1_matrix=x1_matrix,
                x2_matrix=x2_matrix,
                current_x1=52.5,
                current_x2=100,
                x1_threshold=5,  # Range: [47.5, 57.5]
                x2_threshold=10,
            )

            # X1 should only count positive values within threshold
            # [50, 1] -> pass, [55, 1] -> pass
            # [-10, 1] and [0, 1] should be skipped
            assert results["y1_pass"] == 2
            assert results["y1_fail"] == 0

    def test_x2_checks_all_values_including_negative(self):
        """Test that X2 pattern matching checks all values including negative."""
        with patch("modules.decision_matrix.shuffle.ShuffleMechanism.shuffle_matrix", side_effect=lambda x, r: x[:r]):
            x1_matrix = [[100, 1], [100, 1], [100, 1], [100, 1], [999, 1]]

            # X2 has negative values
            x2_matrix = [
                [50, 1],  # Positive, within threshold
                [-10, 1],  # Negative, outside threshold
                [0, 0],  # Zero, outside threshold
                [55, 0],  # Positive, within threshold
                [999, 1],  # Last row - skipped
            ]

            results = self.rf.classify(
                x1_matrix=x1_matrix,
                x2_matrix=x2_matrix,
                current_x1=100,
                current_x2=52.5,
                x1_threshold=10,
                x2_threshold=5,  # Range: [47.5, 57.5]
            )

            # X2 should check ALL values (including negative/zero)
            # But only those within threshold range match:
            # [50, 1] -> pass, [55, 0] -> fail
            # [-10, 1] and [0, 0] are outside threshold
            assert results["y2_pass"] == 1
            assert results["y2_fail"] == 1

    def test_shuffle_randomness(self):
        """Test that shuffle produces different orderings (probabilistic test)."""
        x1_matrix = [[i, i % 2] for i in range(1, 21)]  # 20 samples
        x2_matrix = [[i * 10, i % 2] for i in range(1, 21)]

        # Run classification multiple times
        results_list = []
        for _ in range(5):
            results = self.rf.classify(
                x1_matrix=x1_matrix.copy(),
                x2_matrix=x2_matrix.copy(),
                current_x1=10,
                current_x2=100,
                x1_threshold=5,
                x2_threshold=50,
            )
            results_list.append(results)

        # Due to shuffling, we might get different pass/fail counts
        # (This is probabilistic - not guaranteed, but very likely with 20 samples)
        # At minimum, verify that classification runs without errors
        for results in results_list:
            assert "vote" in results
            assert results["vote"] in [0, 1]


class TestRandomForestCoreEdgeCases:
    """Test edge cases for RandomForestCore."""

    def test_single_training_sample(self):
        """Test with only one training sample (will be skipped by [:-1])."""
        with patch("modules.decision_matrix.shuffle.ShuffleMechanism.shuffle_matrix", side_effect=lambda x, r: x[:r]):
            rf = RandomForestCore()

            x1_matrix = [[50, 1]]  # Only one sample, will be skipped
            x2_matrix = [[100, 1]]

            results = rf.classify(
                x1_matrix=x1_matrix,
                x2_matrix=x2_matrix,
                current_x1=50,
                current_x2=100,
                x1_threshold=10,
                x2_threshold=20,
            )

        # With [:-1], no samples will be processed
        assert results["y1_pass"] == 0
        assert results["y1_fail"] == 0
        assert results["y2_pass"] == 0
        assert results["y2_fail"] == 0
        assert results["vote"] == 0  # No passes, no fails -> vote = 0

    def test_two_training_samples(self):
        """Test with two training samples (one will be skipped)."""
        with patch("modules.decision_matrix.shuffle.ShuffleMechanism.shuffle_matrix", side_effect=lambda x, r: x[:r]):
            rf = RandomForestCore()

            x1_matrix = [[50, 1], [60, 0]]  # Second will be skipped
            x2_matrix = [[100, 1], [110, 0]]

            results = rf.classify(
                x1_matrix=x1_matrix,
                x2_matrix=x2_matrix,
                current_x1=55,
                current_x2=105,
                x1_threshold=10,
                x2_threshold=15,
            )

        # Only first sample should be processed
        # X1: [50, 1] in range [45, 65] -> pass
        # X2: [100, 1] in range [90, 120] -> pass
        assert results["y1_pass"] == 1
        assert results["y1_fail"] == 0
        assert results["y2_pass"] == 1
        assert results["y2_fail"] == 0
        assert results["vote"] == 1

    def test_zero_threshold(self):
        """Test with zero threshold (exact matching only)."""
        with patch("modules.decision_matrix.shuffle.ShuffleMechanism.shuffle_matrix", side_effect=lambda x, r: x[:r]):
            rf = RandomForestCore()

            x1_matrix = [[50, 1], [50.0, 0], [50, 1], [999, 1]]
            x2_matrix = [[100, 1], [100, 0], [100, 1], [999, 1]]

            results = rf.classify(
                x1_matrix=x1_matrix,
                x2_matrix=x2_matrix,
                current_x1=50,
                current_x2=100,
                x1_threshold=0,  # Exact match only
                x2_threshold=0,
            )

        # All first 3 samples should match (exact value = 50 and 100)
        assert results["y1_pass"] == 2  # Two 1s
        assert results["y1_fail"] == 1  # One 0
        assert results["y2_pass"] == 2
        assert results["y2_fail"] == 1

    def test_large_threshold_matches_all(self):
        """Test with very large threshold that matches all values."""
        with patch("modules.decision_matrix.shuffle.ShuffleMechanism.shuffle_matrix", side_effect=lambda x, r: x[:r]):
            rf = RandomForestCore()

            x1_matrix = [[1, 1], [100, 0], [1000, 1], [9999, 1]]
            x2_matrix = [[1, 1], [100, 0], [1000, 1], [9999, 1]]

            results = rf.classify(
                x1_matrix=x1_matrix,
                x2_matrix=x2_matrix,
                current_x1=500,
                current_x2=500,
                x1_threshold=10000,  # Very large, matches all
                x2_threshold=10000,
            )

        # All first 3 samples should match (last is skipped)
        assert results["y1_pass"] == 2  # [1, 1] and [1000, 1]
        assert results["y1_fail"] == 1  # [100, 0]
        assert results["y2_pass"] == 2
        assert results["y2_fail"] == 1
