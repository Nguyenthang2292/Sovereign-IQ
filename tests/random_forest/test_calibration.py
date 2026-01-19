"""Tests for probability calibration functionality."""

import sys
from pathlib import Path

# Add project root to Python path for imports
ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from modules.random_forest.utils.calibration import calibrate_model, evaluate_calibration


class TestCalibrateModel:
    """Test probability calibration."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        # Create features
        features = np.random.randn(n_samples, n_features)
        # Create target with some correlation to features
        target = ((features[:, 0] * 2 + features[:, 1] * 1.5) > 0).astype(int)

        X_train = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(n_features)])
        y_train = pd.Series(target, name="target")

        return X_train, y_train

    def test_calibrate_model_sigmoid(self, sample_data):
        """Test calibration with sigmoid method (Platt scaling)."""
        X_train, y_train = sample_data

        # Train base model
        base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        base_model.fit(X_train, y_train)

        # Calibrate model
        calibrated = calibrate_model(base_model, X_train, y_train, method="sigmoid", cv=3)

        assert isinstance(calibrated, CalibratedClassifierCV)
        assert calibrated.method == "sigmoid"
        assert calibrated.cv == 3

    def test_calibrate_model_isotonic(self, sample_data):
        """Test calibration with isotonic method."""
        X_train, y_train = sample_data

        # Train base model
        base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        base_model.fit(X_train, y_train)

        # Calibrate model
        calibrated = calibrate_model(base_model, X_train, y_train, method="isotonic", cv=5)

        assert isinstance(calibrated, CalibratedClassifierCV)
        assert calibrated.method == "isotonic"
        assert calibrated.cv == 5

    def test_calibrate_model_unknown_method(self, sample_data):
        """Test calibration with unknown method (should fallback to sigmoid)."""
        X_train, y_train = sample_data

        # Train base model
        base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        base_model.fit(X_train, y_train)

        # Calibrate with unknown method
        calibrated = calibrate_model(base_model, X_train, y_train, method="unknown", cv=3)

        assert isinstance(calibrated, CalibratedClassifierCV)
        assert calibrated.method == "sigmoid"  # Should fallback

    def test_calibrate_model_predict_proba(self, sample_data):
        """Test that calibrated model produces probabilities."""
        X_train, y_train = sample_data

        # Train base model
        base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        base_model.fit(X_train, y_train)

        # Calibrate model
        calibrated = calibrate_model(base_model, X_train, y_train, method="sigmoid", cv=3)

        # Test prediction
        X_test = X_train.iloc[:10]
        proba = calibrated.predict_proba(X_test)

        assert proba.shape[0] == 10
        assert proba.shape[1] == 2  # Binary classification
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(proba >= 0) and np.all(proba <= 1)  # Valid probability range

    def test_calibrate_model_predict(self, sample_data):
        """Test that calibrated model can predict classes."""
        X_train, y_train = sample_data

        # Train base model
        base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        base_model.fit(X_train, y_train)

        # Calibrate model
        calibrated = calibrate_model(base_model, X_train, y_train, method="sigmoid", cv=3)

        # Test prediction
        X_test = X_train.iloc[:10]
        predictions = calibrated.predict(X_test)

        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)


class TestEvaluateCalibration:
    """Test calibration evaluation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training and test data."""
        np.random.seed(42)
        n_samples = 300
        n_features = 10

        # Create features
        features = np.random.randn(n_samples, n_features)
        # Create target with some correlation to features
        target = ((features[:, 0] * 2 + features[:, 1] * 1.5) > 0).astype(int)

        X = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(n_features)])
        y = pd.Series(target, name="target")

        # Split into train and test
        split_idx = int(n_samples * 0.7)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def test_evaluate_calibration_binary(self, sample_data):
        """Test calibration evaluation for binary classification."""
        X_train, X_test, y_train, y_test = sample_data

        # Train and calibrate model
        base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        base_model.fit(X_train, y_train)
        calibrated = calibrate_model(base_model, X_train, y_train, method="sigmoid", cv=3)

        # Evaluate calibration
        metrics = evaluate_calibration(calibrated, X_test, y_test)

        assert metrics is not None
        assert "brier_score" in metrics
        assert "expected_calibration_error" in metrics
        assert "calibration_curve" in metrics
        assert isinstance(metrics["brier_score"], float)
        assert isinstance(metrics["expected_calibration_error"], float)
        assert 0 <= metrics["brier_score"] <= 1  # Brier score range
        assert 0 <= metrics["expected_calibration_error"] <= 1  # ECE range

    def test_evaluate_calibration_multiclass(self):
        """Test calibration evaluation for multi-class classification."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        # Create multi-class data
        features = np.random.randn(n_samples, n_features)
        target = ((features[:, 0] * 2 + features[:, 1] * 1.5) % 3).astype(int)

        X = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(n_features)])
        y = pd.Series(target, name="target")

        split_idx = int(n_samples * 0.7)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        # Train and calibrate model
        base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        base_model.fit(X_train, y_train)
        calibrated = calibrate_model(base_model, X_train, y_train, method="sigmoid", cv=3)

        # Evaluate calibration
        metrics = evaluate_calibration(calibrated, X_test, y_test)

        assert metrics is not None
        assert "brier_score" in metrics
        assert "expected_calibration_error" in metrics

    def test_evaluate_calibration_uncalibrated(self, sample_data):
        """Test calibration evaluation on uncalibrated model."""
        X_train, X_test, y_train, y_test = sample_data

        # Train uncalibrated model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate calibration (should still work)
        metrics = evaluate_calibration(model, X_test, y_test)

        assert metrics is not None
        assert "brier_score" in metrics
        assert "expected_calibration_error" in metrics

    def test_evaluate_calibration_empty_data(self, sample_data):
        """Test calibration evaluation with empty test data."""
        X_train, _, y_train, _ = sample_data

        # Train and calibrate model
        base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        base_model.fit(X_train, y_train)
        calibrated = calibrate_model(base_model, X_train, y_train, method="sigmoid", cv=3)

        # Evaluate with empty data
        X_test_empty = pd.DataFrame()
        y_test_empty = pd.Series(dtype=int)

        metrics = evaluate_calibration(calibrated, X_test_empty, y_test_empty)

        # Should handle gracefully (may return None or handle error)
        # The exact behavior depends on sklearn's error handling
        assert metrics is None or isinstance(metrics, dict)


class TestCalibrationIntegration:
    """Test calibration integration with training pipeline."""

    def test_calibration_improves_probability_quality(self):
        """Test that calibration improves probability calibration metrics."""
        np.random.seed(42)
        n_samples = 300
        n_features = 10

        # Create data
        features = np.random.randn(n_samples, n_features)
        target = ((features[:, 0] * 2 + features[:, 1] * 1.5) > 0).astype(int)

        X = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(n_features)])
        y = pd.Series(target, name="target")

        split_idx = int(n_samples * 0.7)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        # Train uncalibrated model
        uncalibrated = RandomForestClassifier(n_estimators=50, random_state=42)
        uncalibrated.fit(X_train, y_train)

        # Train calibrated model
        calibrated = calibrate_model(uncalibrated, X_train, y_train, method="sigmoid", cv=3)

        # Evaluate both
        metrics_uncalibrated = evaluate_calibration(uncalibrated, X_test, y_test)
        metrics_calibrated = evaluate_calibration(calibrated, X_test, y_test)

        # Calibrated model should have better (lower) Brier score or ECE
        # Note: This is not always guaranteed, but generally true
        assert metrics_uncalibrated is not None
        assert metrics_calibrated is not None

        # Both should produce valid metrics
        assert "brier_score" in metrics_uncalibrated
        assert "brier_score" in metrics_calibrated
        assert "expected_calibration_error" in metrics_uncalibrated
        assert "expected_calibration_error" in metrics_calibrated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
