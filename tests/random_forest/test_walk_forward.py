"""Tests for walk-forward optimization functionality."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path for imports
ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from modules.random_forest.utils.walk_forward import (
    ModelDriftDetector,
    ModelVersionManager,
    WalkForwardValidator,
    should_retrain_model,
)


class TestWalkForwardValidator:
    """Test walk-forward validation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time-series data."""
        np.random.seed(42)
        n_samples = 500
        features = pd.DataFrame(np.random.randn(n_samples, 10), columns=[f"feature_{i}" for i in range(10)])
        target = pd.Series(np.random.randint(0, 3, n_samples), name="target")
        return features, target

    def test_walk_forward_expanding_window(self, sample_data):
        """Test expanding window walk-forward validation."""
        features, target = sample_data
        validator = WalkForwardValidator(n_splits=3, expanding_window=True, gap=10)

        splits = validator.split(features, target)

        assert len(splits) > 0
        # Check that train sets are expanding
        if len(splits) >= 2:
            train_size_1 = len(splits[0][0])
            train_size_2 = len(splits[1][0])
            assert train_size_2 > train_size_1, "Expanding window should grow"

    def test_walk_forward_rolling_window(self, sample_data):
        """Test rolling window walk-forward validation."""
        features, target = sample_data
        validator = WalkForwardValidator(n_splits=3, expanding_window=False, gap=10)

        splits = validator.split(features, target)

        assert len(splits) > 0
        # All splits should have similar train sizes (within reasonable variance)
        # Note: Rolling window may have some variance due to step size calculations
        if len(splits) >= 2:
            train_sizes = [len(split[0]) for split in splits]
            # Rolling window sizes should be similar (allow some variance)
            size_variance = max(train_sizes) - min(train_sizes)
            # Variance should be less than 50% of average size
            avg_size = sum(train_sizes) / len(train_sizes)
            assert size_variance < avg_size * 0.5, f"Rolling window sizes should be similar (variance={size_variance}, avg={avg_size})"

    def test_walk_forward_gap_validation(self, sample_data):
        """Test that gap is properly applied between train and test sets."""
        features, target = sample_data
        gap = 20
        validator = WalkForwardValidator(n_splits=2, expanding_window=True, gap=gap)

        splits = validator.split(features, target)

        if len(splits) > 0:
            features_train, features_test, _, _ = splits[0]
            # Check that there's a gap between train end and test start
            train_end_idx = len(features_train) - 1
            test_start_idx = features_test.index[0]
            # Gap should be at least the specified gap
            assert test_start_idx - train_end_idx >= gap, f"Gap should be at least {gap}"

    def test_walk_forward_insufficient_data(self):
        """Test handling of insufficient data."""
        # Create very small dataset
        features = pd.DataFrame(np.random.randn(50, 5))
        target = pd.Series(np.random.randint(0, 2, 50))

        validator = WalkForwardValidator(n_splits=10, expanding_window=True, gap=20)
        splits = validator.split(features, target)

        # Should handle gracefully (may have fewer splits)
        assert isinstance(splits, list)


class TestModelDriftDetector:
    """Test model drift detection."""

    def test_drift_detector_init(self):
        """Test drift detector initialization."""
        detector = ModelDriftDetector(enabled=True, threshold=0.05, window_size=100)

        assert detector.enabled is True
        assert detector.threshold == 0.05
        assert detector.window_size == 100
        assert detector.baseline_accuracy is None

    def test_set_baseline(self):
        """Test setting baseline accuracy."""
        detector = ModelDriftDetector(enabled=True)
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])

        detector.set_baseline(y_true, y_pred)

        assert detector.baseline_accuracy == 1.0  # Perfect accuracy

    def test_drift_detection_no_drift(self):
        """Test drift detection when no drift occurs."""
        detector = ModelDriftDetector(enabled=True, threshold=0.05, window_size=5)
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])

        detector.set_baseline(y_true, y_pred)

        # Add predictions with same performance
        for _ in range(5):
            detector.add_prediction(y_true, y_pred)

        drift_detected, current_accuracy = detector.check_drift()

        assert drift_detected is False
        assert current_accuracy == 1.0

    def test_drift_detection_with_drift(self):
        """Test drift detection when drift occurs."""
        detector = ModelDriftDetector(enabled=True, threshold=0.05, window_size=5)
        y_true_good = np.array([0, 1, 0, 1, 0])
        y_pred_good = np.array([0, 1, 0, 1, 0])

        detector.set_baseline(y_true_good, y_pred_good)

        # Add predictions with poor performance (all wrong)
        y_pred_bad = np.array([1, 0, 1, 0, 1])
        for _ in range(5):
            detector.add_prediction(y_true_good, y_pred_bad)

        drift_detected, current_accuracy = detector.check_drift()

        assert drift_detected is True
        assert current_accuracy < detector.baseline_accuracy

    def test_drift_detection_insufficient_data(self):
        """Test drift detection with insufficient data."""
        detector = ModelDriftDetector(enabled=True, threshold=0.05, window_size=10)
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])

        detector.set_baseline(y_true, y_pred)

        # Add only 2 predictions (less than window_size)
        detector.add_prediction(y_true, y_pred)
        detector.add_prediction(y_true, y_pred)

        drift_detected, current_accuracy = detector.check_drift()

        assert drift_detected is False  # Not enough data yet

    def test_drift_detection_disabled(self):
        """Test drift detection when disabled."""
        detector = ModelDriftDetector(enabled=False)

        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])

        detector.add_prediction(y_true, y_pred)
        drift_detected, current_accuracy = detector.check_drift()

        assert drift_detected is False
        assert current_accuracy is None


class TestModelVersionManager:
    """Test model versioning."""

    def test_version_manager_init(self):
        """Test version manager initialization."""
        manager = ModelVersionManager(enabled=True)

        assert manager.enabled is True
        assert manager.version == 1

    def test_get_version_string(self):
        """Test version string generation."""
        manager = ModelVersionManager(enabled=True)

        version_str = manager.get_version_string()

        assert version_str.startswith("rf_v1_")
        assert len(version_str) > 10  # Should have timestamp

    def test_version_increment(self):
        """Test version increment."""
        manager = ModelVersionManager(enabled=True)

        assert manager.version == 1
        manager.increment_version()
        assert manager.version == 2
        manager.increment_version()
        assert manager.version == 3

    def test_get_model_metadata(self):
        """Test model metadata generation."""
        manager = ModelVersionManager(enabled=True)

        metadata = manager.get_model_metadata()

        assert "version" in metadata
        assert "version_string" in metadata
        assert "timestamp" in metadata
        assert metadata["version"] == 1

    def test_version_manager_disabled(self):
        """Test version manager when disabled."""
        manager = ModelVersionManager(enabled=False)

        version_str = manager.get_version_string()

        assert version_str == "rf_latest"


class TestRetrainSchedule:
    """Test periodic retraining schedule."""

    def test_should_retrain_never_trained(self):
        """Test retrain check when model was never trained."""
        result = should_retrain_model(None)

        assert result is True  # Should train if never trained

    def test_should_retrain_recently_trained(self):
        """Test retrain check when model was recently trained."""
        recent_date = datetime.now() - timedelta(days=5)
        result = should_retrain_model(recent_date)

        # Assuming default retrain period is 30 days
        assert result is False  # Too recent

    def test_should_retrain_old_model(self):
        """Test retrain check when model is old."""
        old_date = datetime.now() - timedelta(days=35)
        result = should_retrain_model(old_date)

        # Assuming default retrain period is 30 days
        assert result is True  # Should retrain


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
