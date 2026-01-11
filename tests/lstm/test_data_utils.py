
import numpy as np
import pytest

from config.lstm import TRAIN_TEST_SPLIT, VALIDATION_SPLIT
from modules.lstm.utils.data_utils import split_train_test_data
from modules.lstm.utils.data_utils import split_train_test_data

"""
Tests for data utility functions.
"""




class TestSplitTrainTestData:
    """Test suite for split_train_test_data function."""

    @pytest.fixture
    def sample_data(self, seeded_random):
        """Create sample data for testing."""
        n_samples = 1000
        n_features = 50
        sequence_length = 60

        X = seeded_random.standard_normal((n_samples, sequence_length, n_features))
        y = seeded_random.integers(-1, 2, size=n_samples)

        return X, y

    def test_basic_split(self, sample_data):
        """Test basic train/validation/test split."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_test_data(X, y)

        # Check shapes
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)

        # Check that all data is accounted for
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(X)

        # Check ratios are approximately correct
        train_ratio = len(X_train) / len(X)
        val_ratio = len(X_val) / len(X)
        test_ratio = len(X_test) / len(X)

        assert train_ratio == pytest.approx(TRAIN_TEST_SPLIT, abs=0.05)
        assert val_ratio == pytest.approx(VALIDATION_SPLIT, abs=0.05)
        assert test_ratio == pytest.approx(1 - TRAIN_TEST_SPLIT - VALIDATION_SPLIT, abs=0.05)

    def test_custom_ratios(self, sample_data):
        """Test split with custom ratios."""
        X, y = sample_data
        train_ratio = 0.8
        val_ratio = 0.1

        X_train, X_val, X_test, y_train, y_val, y_test = split_train_test_data(
            X, y, train_ratio=train_ratio, validation_ratio=val_ratio
        )

        # Check ratios
        assert len(X_train) / len(X) == pytest.approx(train_ratio, abs=0.05)
        assert len(X_val) / len(X) == pytest.approx(val_ratio, abs=0.05)
        assert len(X_test) / len(X) == pytest.approx(1 - train_ratio - val_ratio, abs=0.05)

    def test_no_shuffle(self, sample_data):
        """Test split without shuffling (for time series data)."""
        X, y = sample_data

        X_train, X_val, X_test, y_train, y_val, y_test = split_train_test_data(X, y, shuffle=False)

        # First samples should be in train set
        assert np.array_equal(X_train[0], X[0])
        # Last samples should be in test set
        assert np.array_equal(X_test[-1], X[-1])

    def test_with_shuffle(self, sample_data):
        """Test split with shuffling."""
        X, y = sample_data

        X_train, X_val, X_test, y_train, y_val, y_test = split_train_test_data(X, y, shuffle=True, random_state=42)

        # With shuffling, the concatenated splits should not match original order
        X_concat = np.concatenate([X_train, X_val, X_test])
        # Check that at least some portion of the data is reordered
        matches = sum(np.array_equal(X_concat[i], X[i]) for i in range(min(100, len(X))))
        assert matches < 90  # Expect most positions to differ with shuffle

    def test_return_indices(self, sample_data):
        """Test split with return_indices=True."""
        X, y = sample_data

        result = split_train_test_data(X, y, return_indices=True, shuffle=False)
        X_train, X_val, X_test, y_train, y_val, y_test, test_indices = result

        # Check that test_indices correspond to test set
        assert len(test_indices) == len(X_test)
        assert np.array_equal(X_test, X[test_indices])
        assert np.array_equal(y_test, y[test_indices])

    def test_return_indices_with_shuffle(self, sample_data):
        """Test return_indices with shuffle=True."""
        X, y = sample_data

        result = split_train_test_data(X, y, return_indices=True, shuffle=True, random_state=42)
        X_train, X_val, X_test, y_train, y_val, y_test, test_indices = result

        # Check that test_indices correspond to test set
        assert len(test_indices) == len(X_test)
        assert np.array_equal(X_test, X[test_indices])
        assert np.array_equal(y_test, y[test_indices])

    def test_random_state_reproducibility(self, sample_data):
        """Test that random_state ensures reproducibility."""
        X, y = sample_data

        result1 = split_train_test_data(X, y, shuffle=True, random_state=42)
        result2 = split_train_test_data(X, y, shuffle=True, random_state=42)

        X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = result1
        X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = result2

        # Results should be identical
        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(X_val1, X_val2)
        assert np.array_equal(X_test1, X_test2)
        assert np.array_equal(y_train1, y_train2)
        assert np.array_equal(y_val1, y_val2)
        assert np.array_equal(y_test1, y_test2)

    def test_invalid_input_not_numpy_array(self):
        """Test with non-numpy array input."""
        X = [[1, 2, 3], [4, 5, 6]]
        y = [0, 1]

        with pytest.raises(ValueError, match="X and y must be numpy arrays"):
            split_train_test_data(X, y)

    def test_length_mismatch(self):
        """Test with mismatched X and y lengths."""
        X = np.random.randn(100, 60, 50)
        y = np.random.randint(-1, 2, size=99)  # Different length

        with pytest.raises(ValueError, match="X and y length mismatch"):
            split_train_test_data(X, y)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        X = np.random.randn(5, 60, 50)
        y = np.random.randint(-1, 2, size=5)

        with pytest.raises(ValueError, match="Insufficient data"):
            split_train_test_data(X, y)

    def test_invalid_ratios_zero(self):
        """Test with zero ratio."""
        X = np.random.randn(100, 60, 50)
        y = np.random.randint(-1, 2, size=100)

        with pytest.raises(ValueError, match="Ratios must be between 0 and 1"):
            split_train_test_data(X, y, train_ratio=0)

    def test_invalid_ratios_one(self):
        """Test with ratio equal to 1."""
        X = np.random.randn(100, 60, 50)
        y = np.random.randint(-1, 2, size=100)

        with pytest.raises(ValueError, match="Ratios must be between 0 and 1"):
            split_train_test_data(X, y, train_ratio=1.0)

    def test_invalid_ratios_sum_to_one(self):
        """Test with ratios that sum to 1 or more."""
        X = np.random.randn(100, 60, 50)
        y = np.random.randint(-1, 2, size=100)

        with pytest.raises(ValueError, match="Sum of ratios must be less than 1"):
            split_train_test_data(X, y, train_ratio=0.7, validation_ratio=0.3)

    def test_negative_ratios(self):
        """Test with negative ratios."""
        X = np.random.randn(100, 60, 50)
        y = np.random.randint(-1, 2, size=100)

        with pytest.raises(ValueError, match="Ratios must be between 0 and 1"):
            split_train_test_data(X, y, train_ratio=-0.1)

    def test_minimum_data_size(self):
        """Test with exactly minimum required data."""
        X = np.random.randn(10, 60, 50)
        y = np.random.randint(-1, 2, size=10)

        # Should work with exactly 10 samples
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_test_data(X, y)

        assert len(X_train) + len(X_val) + len(X_test) == 10
        assert all(len(arr) > 0 for arr in [X_train, X_val, X_test])

    def test_large_dataset(self, seeded_random):
        """Test with large dataset."""
        n_samples = 10000
        X = seeded_random.standard_normal((n_samples, 60, 50))
        y = seeded_random.integers(-1, 2, size=n_samples)

        X_train, X_val, X_test, y_train, y_val, y_test = split_train_test_data(X, y)

        assert len(X_train) + len(X_val) + len(X_test) == n_samples
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0

    def test_1d_arrays(self):
        """Test with single feature dimension (should work but may not be typical use case)."""
        X = np.random.randn(100, 60, 1)  # Single feature
        y = np.random.randint(-1, 2, size=100)

        X_train, X_val, X_test, y_train, y_val, y_test = split_train_test_data(X, y)

        assert len(X_train) + len(X_val) + len(X_test) == 100

    def test_regression_targets(self, seeded_random):
        """Test with regression targets (continuous values)."""
        X = seeded_random.standard_normal((1000, 60, 50))
        y = seeded_random.standard_normal(1000)  # Continuous targets

        # Use shuffle=False to preserve order for comparison
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_test_data(X, y, shuffle=False)

        assert len(y_train) == len(X_train)
        assert len(y_val) == len(X_val)
        assert len(y_test) == len(X_test)

        # Check that target values are preserved (with shuffle=False, order is preserved)
        assert np.allclose(np.concatenate([y_train, y_val, y_test]), y, atol=1e-10)
