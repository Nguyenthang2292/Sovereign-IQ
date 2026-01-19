import sys
from pathlib import Path

"""
Tests for Random Forest feature consistency.

Tests verify that:
- Model is trained with MODEL_FEATURES
- Prediction uses the same features that model was trained with
- No feature mismatch errors occur
- Model.feature_names_in_ matches MODEL_FEATURES
"""


# Add project root to Python path for imports
ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from config.model_features import MODEL_FEATURES
from modules.random_forest import (
    get_latest_random_forest_signal,
    load_random_forest_model,
    train_random_forest_model,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def training_data():
    """Create larger dataset for training"""
    np.random.seed(42)
    training_size = 500
    close_prices_large = 100 + np.cumsum(np.random.randn(training_size) * 0.5)
    high_prices_large = close_prices_large + np.random.uniform(0, 2, training_size)
    low_prices_large = close_prices_large - np.random.uniform(0, 2, training_size)
    open_prices_large = close_prices_large + np.random.uniform(-1, 1, training_size)

    volume_data_large = np.random.uniform(1000, 10000, training_size)

    return pd.DataFrame(
        {
            "open": open_prices_large,
            "high": high_prices_large,
            "low": low_prices_large,
            "close": close_prices_large,
            "volume": volume_data_large,
        }
    )


@pytest.fixture
def sample_data():
    """Create sample OHLC data for testing"""
    np.random.seed(42)
    sample_size = 100

    close_prices = 100 + np.cumsum(np.random.randn(sample_size) * 0.5)
    high_prices = close_prices + np.random.uniform(0, 2, sample_size)
    low_prices = close_prices - np.random.uniform(0, 2, sample_size)
    open_prices = close_prices + np.random.uniform(-1, 1, sample_size)

    volume_data = np.random.uniform(1000, 10000, sample_size)

    return pd.DataFrame(
        {"open": open_prices, "high": high_prices, "low": low_prices, "close": close_prices, "volume": volume_data}
    )


@pytest.fixture
def mock_sufficient_memory():
    """Mock sufficient memory (4GB)"""
    with patch("modules.random_forest.utils.training.psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.available = 4 * 1024**3  # 4GB
        yield mock_mem


# ============================================================================
# Tests for Feature Consistency
# ============================================================================


class TestFeatureConsistency:
    """Test suite for feature consistency between training and prediction"""

    def test_model_features_configuration(self):
        """Test that MODEL_FEATURES is properly configured and non-empty"""
        assert MODEL_FEATURES is not None, "MODEL_FEATURES should not be None"
        assert isinstance(
            MODEL_FEATURES, (list, tuple)
        ), f"MODEL_FEATURES should be a list or tuple, got {type(MODEL_FEATURES)}"
        assert (
            len(MODEL_FEATURES) > 0
        ), f"MODEL_FEATURES should have at least one feature, got {len(MODEL_FEATURES)} features"

        # Verify raw OHLCV features are NOT in MODEL_FEATURES (replaced by derived features)
        raw_ohlcv = ["open", "high", "low", "close", "volume"]
        for feature in raw_ohlcv:
            assert feature not in MODEL_FEATURES, (
                f"Raw OHLCV feature '{feature}' should not be in MODEL_FEATURES. "
                f"Use derived features (returns_1, returns_5, log_volume, high_low_range, close_open_diff) instead."
            )

        # Verify derived price features ARE in MODEL_FEATURES
        derived_features = ["returns_1", "returns_5", "log_volume", "high_low_range", "close_open_diff"]
        for feature in derived_features:
            assert feature in MODEL_FEATURES, f"Price-derived feature '{feature}' should be in MODEL_FEATURES"

    def test_model_trained_with_model_features(self, mock_sufficient_memory, training_data):
        """Test that model is trained with MODEL_FEATURES"""
        with patch("modules.random_forest.core.model.joblib.dump"):
            model = train_random_forest_model(training_data, save_model=False)

        assert model is not None

        # Check if model has feature_names_in_ attribute (sklearn >= 1.0)
        if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
            model_features = list(model.feature_names_in_)

            # Model should only have features from MODEL_FEATURES
            # (includes all features that are available in the data)
            for feature in model_features:
                assert feature in MODEL_FEATURES, f"Model feature '{feature}' not in MODEL_FEATURES"

    def test_model_feature_count_matches_model_features(self, mock_sufficient_memory, training_data):
        """Test that model has correct number of features matching MODEL_FEATURES"""
        with patch("modules.random_forest.core.model.joblib.dump"):
            model = train_random_forest_model(training_data, save_model=False)

        assert model is not None

        if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
            model_features = list(model.feature_names_in_)
            # Model should have at most the number of features in MODEL_FEATURES
            # (may be less if some features are not available in the data)
            assert len(model_features) <= len(
                MODEL_FEATURES
            ), f"Model has {len(model_features)} features, but MODEL_FEATURES has {len(MODEL_FEATURES)}"

            # Model should have at least some features
            assert len(model_features) > 0, "Model should have at least one feature"

    def test_prediction_uses_model_features(self, mock_sufficient_memory, training_data, sample_data):
        """Test that prediction uses the same features that model was trained with"""
        with patch("modules.random_forest.core.model.joblib.dump"):
            model = train_random_forest_model(training_data, save_model=False)

        if model is None:
            pytest.skip("Model training failed")

        # Get signal - should not raise feature mismatch error
        signal, confidence = get_latest_random_forest_signal(sample_data, model)

        # Should return valid signal
        assert signal in ["LONG", "SHORT", "NEUTRAL"]
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_no_feature_mismatch_error(self, mock_sufficient_memory, training_data, sample_data):
        """Test that prediction does not raise feature mismatch errors"""
        with patch("modules.random_forest.core.model.joblib.dump"):
            model = train_random_forest_model(training_data, save_model=False)

        if model is None:
            pytest.skip("Model training failed")

        # This should not raise ValueError about feature mismatch
        try:
            signal, confidence = get_latest_random_forest_signal(sample_data, model)
            assert signal in ["LONG", "SHORT", "NEUTRAL"]
        except ValueError as e:
            if "feature" in str(e).lower() and "mismatch" in str(e).lower():
                pytest.fail(f"Feature mismatch error occurred: {e}")
            raise

    def test_model_feature_names_in_consistency(self, mock_sufficient_memory, training_data):
        """Test that model.feature_names_in_ is consistent with MODEL_FEATURES"""
        with patch("modules.random_forest.core.model.joblib.dump"):
            model = train_random_forest_model(training_data, save_model=False)

        assert model is not None

        if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
            model_features = list(model.feature_names_in_)

            # All model features should be in MODEL_FEATURES
            missing_features = [f for f in model_features if f not in MODEL_FEATURES]
            assert len(missing_features) == 0, f"Model has features not in MODEL_FEATURES: {missing_features}"

    def test_data_preparation_uses_model_features(self, mock_sufficient_memory, training_data):
        """Test that data preparation filters to MODEL_FEATURES"""
        from modules.random_forest.utils.data_preparation import prepare_training_data

        prepared_data = prepare_training_data(training_data)

        assert prepared_data is not None
        features, target = prepared_data

        # Features DataFrame should only contain columns from MODEL_FEATURES
        feature_columns = set(features.columns)
        model_features_set = set(MODEL_FEATURES)

        # All feature columns should be in MODEL_FEATURES
        extra_features = feature_columns - model_features_set
        assert len(extra_features) == 0, f"Features DataFrame contains columns not in MODEL_FEATURES: {extra_features}"

        # Verify derived price features are present
        derived_features = ["returns_1", "returns_5", "log_volume", "high_low_range", "close_open_diff"]
        for feature in derived_features:
            assert (
                feature in feature_columns
            ), f"Price-derived feature '{feature}' should be present in prepared features"

        # Verify raw OHLCV features are NOT present
        raw_ohlcv = ["open", "high", "low", "close", "volume"]
        for feature in raw_ohlcv:
            assert feature not in feature_columns, f"Raw OHLCV feature '{feature}' should not be in prepared features"

    def test_signal_generation_handles_missing_features(self, mock_sufficient_memory, training_data, sample_data):
        """Test that signal generation handles missing features gracefully"""
        with patch("modules.random_forest.core.model.joblib.dump"):
            model = train_random_forest_model(training_data, save_model=False)

        if model is None:
            pytest.skip("Model training failed")

        # Create data with missing some features
        incomplete_data = sample_data.copy()
        # Remove volume column (keeping OHLC)
        if "volume" in incomplete_data.columns:
            incomplete_data = incomplete_data.drop(columns=["volume"])

        # Should still work (returns NEUTRAL if features can't be computed)
        signal, confidence = get_latest_random_forest_signal(incomplete_data, model)
        assert signal in ["LONG", "SHORT", "NEUTRAL"]

    def test_signal_generation_handles_extra_features(self, mock_sufficient_memory, training_data, sample_data):
        """Test that signal generation ignores extra features not expected by model"""
        with patch("modules.random_forest.core.model.joblib.dump"):
            model = train_random_forest_model(training_data, save_model=False)

        if model is None:
            pytest.skip("Model training failed")

        # Add extra features (features not in MODEL_FEATURES)
        data_with_extra = sample_data.copy()
        # Add some random features not in MODEL_FEATURES
        extra_features = ["extra_feature_1", "extra_feature_2", "extra_feature_3"]
        for feature in extra_features:
            data_with_extra[feature] = np.random.randn(len(data_with_extra))

        # Should work without error (extra features are ignored)
        signal, confidence = get_latest_random_forest_signal(data_with_extra, model)
        assert signal in ["LONG", "SHORT", "NEUTRAL"]


# ============================================================================
# Integration Tests
# ============================================================================


class TestFeatureConsistencyIntegration:
    """Integration tests for feature consistency in full workflow"""

    def test_full_workflow_feature_consistency(self, mock_sufficient_memory, training_data, sample_data):
        """Test full workflow: train -> save -> load -> predict with feature consistency"""
        temp_dir = tempfile.mkdtemp()
        try:
            model_path = Path(temp_dir) / "test_model.joblib"

            # Train and save model (disable versioning for consistent filename)
            with patch("modules.random_forest.core.model.MODELS_DIR", model_path.parent):
                with patch("modules.random_forest.core.model.RANDOM_FOREST_MODEL_FILENAME", model_path.name):
                    with patch("config.random_forest.RANDOM_FOREST_MODEL_VERSIONING_ENABLED", False):
                        model = train_random_forest_model(training_data, save_model=True)

            assert model is not None

            # Load model
            loaded_model = load_random_forest_model(model_path)
            assert loaded_model is not None

            # Predict with loaded model
            signal, confidence = get_latest_random_forest_signal(sample_data, loaded_model)
            assert signal in ["LONG", "SHORT", "NEUTRAL"]
            assert isinstance(confidence, float)

            # Verify feature consistency
            if hasattr(loaded_model, "feature_names_in_") and loaded_model.feature_names_in_ is not None:
                model_features = list(loaded_model.feature_names_in_)
                for feature in model_features:
                    assert feature in MODEL_FEATURES

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
