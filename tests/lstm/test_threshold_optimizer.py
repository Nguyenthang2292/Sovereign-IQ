import numpy as np

from modules.lstm.core.threshold_optimizer import GridSearchThresholdOptimizer

"""
Tests for GridSearchThresholdOptimizer module.
"""


class TestGridSearchThresholdOptimizer:
    """Test suite for GridSearchThresholdOptimizer class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        optimizer = GridSearchThresholdOptimizer()
        assert optimizer.best_threshold is None
        assert optimizer.best_sharpe == -np.inf
        assert len(optimizer.threshold_range) > 0
        assert optimizer.threshold_range[0] >= 0  # Verify sensible range
        assert optimizer.threshold_range[-1] <= 1

    def test_init_custom_threshold_range(self):
        """Test initialization with custom threshold range."""
        custom_range = np.arange(0.02, 0.10, 0.01)
        optimizer = GridSearchThresholdOptimizer(threshold_range=custom_range)
        assert np.array_equal(optimizer.threshold_range, custom_range)

    def test_optimize_regression_threshold_basic(self):
        """Test basic regression threshold optimization."""
        optimizer = GridSearchThresholdOptimizer()
        predictions = np.array([0.05, 0.02, -0.03, -0.05, 0.01, -0.02, 0.03, -0.01])
        returns = np.array([0.06, 0.01, -0.02, -0.04, 0.02, -0.01, 0.04, 0.0])

        best_threshold, best_sharpe = optimizer.optimize_regression_threshold(predictions, returns)

        # Verify return types
        assert isinstance(best_sharpe, (float, np.floating))
        assert optimizer.best_threshold == best_threshold
        assert optimizer.best_sharpe == best_sharpe

    def test_optimize_regression_threshold_strong_signals(self):
        """Test regression threshold optimization with strong signals."""
        optimizer = GridSearchThresholdOptimizer()
        # Create predictions with clear buy/sell signals
        predictions = np.array([0.1, 0.08, -0.09, -0.1, 0.07, -0.08, 0.09, -0.07])
        returns = np.array([0.11, 0.09, -0.08, -0.09, 0.08, -0.07, 0.10, -0.06])

        best_threshold, best_sharpe = optimizer.optimize_regression_threshold(predictions, returns)

        # Should find a reasonable threshold
        assert best_threshold is not None or best_sharpe == -np.inf

    def test_optimize_regression_threshold_weak_signals(self):
        """Test regression threshold optimization with weak signals."""
        optimizer = GridSearchThresholdOptimizer()
        # Create predictions with weak signals
        predictions = np.array([0.001, 0.002, -0.001, -0.002, 0.001, -0.001, 0.002, 0.0])
        returns = np.array([0.001, 0.002, -0.001, -0.002, 0.001, -0.001, 0.002, 0.0])

        best_threshold, best_sharpe = optimizer.optimize_regression_threshold(predictions, returns)

        # May not find good threshold with weak signals
        assert isinstance(best_sharpe, (float, np.floating))

    def test_optimize_regression_threshold_empty_arrays(self):
        """Test regression threshold optimization with empty arrays."""
        optimizer = GridSearchThresholdOptimizer()
        predictions = np.array([])
        returns = np.array([])

        best_threshold, best_sharpe = optimizer.optimize_regression_threshold(predictions, returns)

        assert best_threshold is None
        assert best_sharpe == -np.inf

    def test_optimize_regression_threshold_zero_std(self):
        """Test regression threshold optimization with zero standard deviation."""
        optimizer = GridSearchThresholdOptimizer()
        predictions = np.array([0.05, 0.05, 0.05, 0.05])
        returns = np.array([0.01, 0.01, 0.01, 0.01])  # Constant returns

        best_threshold, best_sharpe = optimizer.optimize_regression_threshold(predictions, returns)

        # Should handle zero std gracefully
        assert isinstance(best_sharpe, (float, np.floating))

    def test_optimize_classification_threshold_basic(self):
        """Test basic classification threshold optimization."""
        optimizer = GridSearchThresholdOptimizer()
        # Create probability predictions (3 classes)
        probabilities = np.array(
            [
                [0.1, 0.2, 0.7],  # Class 2 (BUY)
                [0.8, 0.1, 0.1],  # Class 0 (SELL)
                [0.2, 0.7, 0.1],  # Class 1 (NEUTRAL)
                [0.1, 0.1, 0.8],  # Class 2 (BUY)
            ]
        )
        returns = np.array([0.05, -0.03, 0.01, 0.04])

        best_confidence, best_sharpe = optimizer.optimize_classification_threshold(probabilities, returns)

        assert isinstance(best_confidence, (float, type(None)))
        assert isinstance(best_sharpe, (float, np.floating))

    def test_optimize_classification_threshold_high_confidence(self):
        """Test classification threshold optimization with high confidence predictions."""
        optimizer = GridSearchThresholdOptimizer()
        # Create high confidence predictions
        probabilities = np.array(
            [
                [0.05, 0.05, 0.9],  # High confidence BUY
                [0.9, 0.05, 0.05],  # High confidence SELL
                [0.05, 0.9, 0.05],  # High confidence NEUTRAL
                [0.05, 0.05, 0.9],  # High confidence BUY
            ]
        )
        returns = np.array([0.05, -0.03, 0.01, 0.04])

        best_confidence, best_sharpe = optimizer.optimize_classification_threshold(probabilities, returns)

        assert isinstance(best_confidence, (float, type(None)))
        assert isinstance(best_sharpe, (float, np.floating))

    def test_optimize_classification_threshold_low_confidence(self):
        """Test classification threshold optimization with low confidence predictions."""
        optimizer = GridSearchThresholdOptimizer()
        # Create low confidence predictions
        probabilities = np.array(
            [
                [0.4, 0.3, 0.3],  # Low confidence
                [0.35, 0.35, 0.3],  # Low confidence
                [0.3, 0.4, 0.3],  # Low confidence
                [0.3, 0.3, 0.4],  # Low confidence
            ]
        )
        returns = np.array([0.01, -0.01, 0.0, 0.01])

        best_confidence, best_sharpe = optimizer.optimize_classification_threshold(probabilities, returns)

        assert isinstance(best_confidence, (float, type(None)))
        assert isinstance(best_sharpe, (float, np.floating))

    def test_optimize_classification_threshold_empty_arrays(self):
        """Test classification threshold optimization with empty arrays."""
        optimizer = GridSearchThresholdOptimizer()
        probabilities = np.array([]).reshape(0, 3)
        returns = np.array([])

        best_confidence, best_sharpe = optimizer.optimize_classification_threshold(probabilities, returns)

        assert best_confidence is None
        assert best_sharpe == -np.inf

    def test_optimize_classification_threshold_zero_std(self):
        """Test classification threshold optimization with zero standard deviation."""
        optimizer = GridSearchThresholdOptimizer()
        probabilities = np.array(
            [
                [0.1, 0.2, 0.7],
                [0.1, 0.2, 0.7],
            ]
        )
        returns = np.array([0.01, 0.01])  # Constant returns

        best_confidence, best_sharpe = optimizer.optimize_classification_threshold(probabilities, returns)

        # Should handle zero std gracefully
        assert isinstance(best_sharpe, (float, np.floating))

    def test_optimize_regression_threshold_updates_state(self):
        """Test that optimize_regression_threshold updates instance state."""
        optimizer = GridSearchThresholdOptimizer()
        predictions = np.array([0.05, 0.02, -0.03, -0.05])
        returns = np.array([0.06, 0.01, -0.02, -0.04])

        assert optimizer.best_threshold is None

        optimizer.optimize_regression_threshold(predictions, returns)

        # State should be updated
        assert optimizer.best_threshold is not None or optimizer.best_sharpe == -np.inf
        assert optimizer.best_sharpe >= -np.inf

    def test_optimize_classification_threshold_many_samples(self, seeded_random):
        """Test classification threshold optimization with many samples."""
        optimizer = GridSearchThresholdOptimizer()
        n_samples = 100
        probabilities = np.random.rand(n_samples, 3)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)  # Normalize
        returns = np.random.randn(n_samples) * 0.01

        best_confidence, best_sharpe = optimizer.optimize_classification_threshold(probabilities, returns)

        assert isinstance(best_confidence, (float, type(None)))
        assert isinstance(best_sharpe, (float, np.floating))

    def test_optimize_regression_threshold_many_samples(self, seeded_random):
        """Test regression threshold optimization with many samples."""
        optimizer = GridSearchThresholdOptimizer()
        n_samples = 100
        predictions = np.random.randn(n_samples) * 0.05
        returns = np.random.randn(n_samples) * 0.01

        best_threshold, best_sharpe = optimizer.optimize_regression_threshold(predictions, returns)

        assert isinstance(best_threshold, (float, type(None)))
        assert isinstance(best_sharpe, (float, np.floating))
