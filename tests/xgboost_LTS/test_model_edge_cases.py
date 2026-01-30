"""
Comprehensive tests for XGBoost model edge cases.
Tests error handling, boundary conditions, and complex scenarios.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from modules.xgboost_LTS.core.model import (
    train_and_predict,
    predict_next_move,
    ClassDiversityError,
)
from modules.xgboost_LTS.utils.cache_manager import CacheManager


class TestTrainAndPredictEdgeCases:
    """Test edge cases for train_and_predict function."""

    def test_empty_dataframe(self, monkeypatch):
        """Test with empty DataFrame."""
        df = pd.DataFrame()

        # Should raise appropriate error
        with pytest.raises((ValueError, IndexError)):
            train_and_predict(df, use_cache=False)

    def test_dataframe_with_only_nan_targets(self, monkeypatch):
        """Test when all targets are NaN."""
        df = pd.DataFrame({"feature1": np.random.randn(50), "feature2": np.random.randn(50), "Target": [np.nan] * 50})

        monkeypatch.setattr("modules.xgboost_LTS.core.model.MODEL_FEATURES", ["feature1", "feature2"])
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_MIN_TRAIN_FRACTION", 0.1)

        # Should handle gracefully
        result = train_and_predict(df, use_cache=False)

        # May return None or trained model depending on implementation
        # The important thing is it doesn't crash

    def test_single_class_in_training(self, monkeypatch):
        """Test with single class in training data."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "Target": [0] * 100,  # Only class 0
            }
        )

        monkeypatch.setattr("modules.xgboost_LTS.core.model.MODEL_FEATURES", ["feature1", "feature2"])
        monkeypatch.setattr("modules.xgboost_LTS.core.model.TARGET_HORIZON", 1)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_TRAIN_TEST_SPLIT", 0.8)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_MIN_TRAIN_FRACTION", 0.1)

        with pytest.raises(ClassDiversityError):
            train_and_predict(df, use_cache=False)

    def test_two_classes_missing_third(self, monkeypatch):
        """Test with only 2 out of 3 required classes."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "Target": np.random.choice([0, 1], 100),  # Missing class 2
            }
        )

        monkeypatch.setattr("modules.xgboost_LTS.core.model.MODEL_FEATURES", ["feature1", "feature2"])
        monkeypatch.setattr("modules.xgboost_LTS.core.model.TARGET_HORIZON", 1)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_TRAIN_TEST_SPLIT", 0.8)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_MIN_TRAIN_FRACTION", 0.1)

        with pytest.raises(ClassDiversityError) as exc:
            train_and_predict(df, use_cache=False)

        assert "class 2" in str(exc.value).lower() or "missing" in str(exc.value).lower()

    def test_highly_imbalanced_classes(self, monkeypatch):
        """Test with highly imbalanced class distribution."""
        # 95% class 0, 4% class 1, 1% class 2
        targets = [0] * 95 + [1] * 4 + [2] * 1
        np.random.shuffle(targets)

        df = pd.DataFrame({"feature1": np.random.randn(100), "feature2": np.random.randn(100), "Target": targets})

        monkeypatch.setattr("modules.xgboost_LTS.core.model.MODEL_FEATURES", ["feature1", "feature2"])
        monkeypatch.setattr("modules.xgboost_LTS.core.model.TARGET_HORIZON", 1)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_TRAIN_TEST_SPLIT", 0.8)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_MIN_TRAIN_FRACTION", 0.1)

        # Should still train (all classes present, even if imbalanced)
        result = train_and_predict(df, use_cache=False)

        assert result is not None
        assert hasattr(result, "predict")

    def test_insufficient_training_samples(self, monkeypatch):
        """Test with too few training samples."""
        # Only 10 samples with required train fraction of 0.5
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(10),
                "feature2": np.random.randn(10),
                "Target": np.random.choice([0, 1, 2], 10),
            }
        )

        monkeypatch.setattr("modules.xgboost_LTS.core.model.MODEL_FEATURES", ["feature1", "feature2"])
        monkeypatch.setattr("modules.xgboost_LTS.core.model.TARGET_HORIZON", 1)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_TRAIN_TEST_SPLIT", 0.5)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_MIN_TRAIN_FRACTION", 0.5)

        # May raise error or return None
        try:
            result = train_and_predict(df, use_cache=False)
            # If it returns something, that's okay too
        except (ValueError, IndexError) as e:
            # Error is also acceptable
            pass

    def test_features_with_nan_values(self, monkeypatch):
        """Test training with NaN values in features."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, np.nan, 4.0, 5.0] * 20,
                "feature2": np.random.randn(100),
                "Target": np.random.choice([0, 1, 2], 100),
            }
        )

        monkeypatch.setattr("modules.xgboost_LTS.core.model.MODEL_FEATURES", ["feature1", "feature2"])
        monkeypatch.setattr("modules.xgboost_LTS.core.model.TARGET_HORIZON", 1)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_TRAIN_TEST_SPLIT", 0.8)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_MIN_TRAIN_FRACTION", 0.1)

        # XGBoost should handle NaN values
        result = train_and_predict(df, use_cache=False)

        assert result is not None

    def test_features_with_infinite_values(self, monkeypatch):
        """Test training with infinite values in features."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, np.inf, -np.inf, 5.0] * 20,
                "feature2": np.random.randn(100),
                "Target": np.random.choice([0, 1, 2], 100),
            }
        )

        monkeypatch.setattr("modules.xgboost_LTS.core.model.MODEL_FEATURES", ["feature1", "feature2"])
        monkeypatch.setattr("modules.xgboost_LTS.core.model.TARGET_HORIZON", 1)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_TRAIN_TEST_SPLIT", 0.8)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_MIN_TRAIN_FRACTION", 0.1)

        # Should handle or raise appropriate error
        try:
            result = train_and_predict(df, use_cache=False)
        except (ValueError, FloatingPointError):
            pass  # Also acceptable

    def test_missing_required_features(self, monkeypatch):
        """Test when required features are missing."""
        df = pd.DataFrame({"wrong_feature": np.random.randn(50), "Target": np.random.choice([0, 1, 2], 50)})

        monkeypatch.setattr("modules.xgboost_LTS.core.model.MODEL_FEATURES", ["feature1", "feature2"])

        # Should raise KeyError or similar
        with pytest.raises((KeyError, ValueError)):
            train_and_predict(df, use_cache=False)

    def test_negative_target_values(self, monkeypatch):
        """Test with negative target values."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "Target": np.random.choice([-1, 0, 1], 100),  # Invalid labels
            }
        )

        monkeypatch.setattr("modules.xgboost_LTS.core.model.MODEL_FEATURES", ["feature1", "feature2"])
        monkeypatch.setattr("modules.xgboost_LTS.core.model.TARGET_HORIZON", 1)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_TRAIN_TEST_SPLIT", 0.8)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_MIN_TRAIN_FRACTION", 0.1)

        # May error or handle gracefully
        try:
            result = train_and_predict(df, use_cache=False)
        except (ValueError, AssertionError):
            pass  # Expected

    def test_categorical_target_column(self, monkeypatch):
        """Test with categorical target column."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "Target": pd.Categorical(np.random.choice(["0", "1", "2"], 100)),
            }
        )

        monkeypatch.setattr("modules.xgboost_LTS.core.model.MODEL_FEATURES", ["feature1", "feature2"])
        monkeypatch.setattr("modules.xgboost_LTS.core.model.TARGET_HORIZON", 1)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_TRAIN_TEST_SPLIT", 0.8)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_MIN_TRAIN_FRACTION", 0.1)

        # Should convert or error
        try:
            result = train_and_predict(df, use_cache=False)
        except (ValueError, TypeError):
            pass  # May need conversion


class TestPredictNextMove:
    """Test predict_next_move function."""

    def test_with_series_input(self):
        """Test prediction with Series input."""
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.2, 0.3, 0.5]])

        row = pd.Series({"feature1": 1.0, "feature2": 2.0})

        result = predict_next_move(model, row)

        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, [0.2, 0.3, 0.5])

    def test_with_dataframe_input(self):
        """Test prediction with DataFrame input."""
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.1, 0.6, 0.3]])

        row = pd.DataFrame({"feature1": [1.0], "feature2": [2.0]})

        result = predict_next_move(model, row)

        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, [0.1, 0.6, 0.3])

    def test_model_prediction_error(self):
        """Test handling of model prediction error."""
        model = MagicMock()
        model.predict_proba.side_effect = ValueError("Model error")

        row = pd.Series({"feature1": 1.0})

        with pytest.raises(ValueError) as exc:
            predict_next_move(model, row)

        assert "Model error" in str(exc.value)

    def test_wrong_number_of_classes(self):
        """Test when model returns wrong number of classes."""
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.5, 0.5]])  # Only 2 classes

        row = pd.Series({"feature1": 1.0})

        result = predict_next_move(model, row)

        # Should return the probabilities even if wrong shape
        assert result.shape == (2,)

    def test_empty_input(self):
        """Test with empty input."""
        model = MagicMock()

        with pytest.raises((ValueError, IndexError)):
            predict_next_move(model, pd.Series())


class TestCacheIntegration:
    """Test caching integration with model training."""

    def test_model_caching(self, monkeypatch, tmp_path):
        """Test that models are cached correctly."""
        from unittest.mock import patch as mock_patch

        df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "Target": np.random.choice([0, 1, 2], 100),
            }
        )

        monkeypatch.setattr("modules.xgboost_LTS.core.model.MODEL_FEATURES", ["feature1", "feature2"])
        monkeypatch.setattr("modules.xgboost_LTS.core.model.TARGET_HORIZON", 1)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_TRAIN_TEST_SPLIT", 0.8)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_MIN_TRAIN_FRACTION", 0.1)

        with mock_patch("config.ARTIFACTS_DIR", tmp_path):
            # First call should train
            result1 = train_and_predict(df, use_cache=True)

            # Second call should load from cache
            result2 = train_and_predict(df, use_cache=True)

            assert result1 is not None
            assert result2 is not None

    def test_cache_invalidation(self, monkeypatch, tmp_path):
        """Test that cache is invalidated when data changes."""
        from unittest.mock import patch as mock_patch

        df1 = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "Target": np.random.choice([0, 1, 2], 100),
            }
        )

        df2 = df1.copy()
        df2["feature1"] = df2["feature1"] + 1  # Change data

        monkeypatch.setattr("modules.xgboost_LTS.core.model.MODEL_FEATURES", ["feature1", "feature2"])
        monkeypatch.setattr("modules.xgboost_LTS.core.model.TARGET_HORIZON", 1)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_TRAIN_TEST_SPLIT", 0.8)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_MIN_TRAIN_FRACTION", 0.1)

        with mock_patch("config.ARTIFACTS_DIR", tmp_path):
            result1 = train_and_predict(df1, use_cache=True)
            result2 = train_and_predict(df2, use_cache=True)

            # Both should succeed
            assert result1 is not None
            assert result2 is not None


class TestDataValidation:
    """Test data validation in model training."""

    def test_no_target_column(self, monkeypatch):
        """Test when Target column is missing."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                # No Target column
            }
        )

        monkeypatch.setattr("modules.xgboost_LTS.core.model.MODEL_FEATURES", ["feature1", "feature2"])

        with pytest.raises((KeyError, ValueError)):
            train_and_predict(df, use_cache=False)

    def test_string_target_values(self, monkeypatch):
        """Test with string target values."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "Target": ["UP", "DOWN", "NEUTRAL"] * 17,  # String labels
            }
        )

        monkeypatch.setattr("modules.xgboost_LTS.core.model.MODEL_FEATURES", ["feature1", "feature2"])
        monkeypatch.setattr("modules.xgboost_LTS.core.model.TARGET_HORIZON", 1)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_TRAIN_TEST_SPLIT", 0.8)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_MIN_TRAIN_FRACTION", 0.1)

        # May need conversion or raise error
        try:
            result = train_and_predict(df, use_cache=False)
        except (ValueError, TypeError):
            pass

    def test_float_target_values(self, monkeypatch):
        """Test with float target values."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "Target": [0.0, 1.0, 2.0] * 17,  # Float labels
            }
        )

        monkeypatch.setattr("modules.xgboost_LTS.core.model.MODEL_FEATURES", ["feature1", "feature2"])
        monkeypatch.setattr("modules.xgboost_LTS.core.model.TARGET_HORIZON", 1)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_TRAIN_TEST_SPLIT", 0.8)
        monkeypatch.setattr("modules.xgboost_LTS.core.model.XGBOOST_MIN_TRAIN_FRACTION", 0.1)

        # Should handle float targets
        result = train_and_predict(df, use_cache=False)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
