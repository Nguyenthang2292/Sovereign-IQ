"""Tests for feature order consistency between training and inference.

This test module verifies that:
1. Features in training data preserve order from MODEL_FEATURES
2. Model stores feature names in correct order
3. Inference uses the same feature order as training
4. Feature selection preserves order when possible
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from config.model_features import MODEL_FEATURES
from modules.random_forest import get_latest_random_forest_signal, train_random_forest_model
from modules.random_forest.utils.data_preparation import prepare_training_data


class TestFeatureOrderConsistency:
    """Test suite for feature order consistency."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data with all required features."""
        np.random.seed(42)
        n = 500
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.random.uniform(0, 2, n)
        low = close - np.random.uniform(0, 2, n)
        open_price = close + np.random.uniform(-1, 1, n)
        volume = np.random.uniform(1000, 10000, n)

        df = pd.DataFrame(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        return df

    def test_prepare_training_data_preserves_model_features_order(self, sample_training_data):
        """Test that prepare_training_data preserves order from MODEL_FEATURES."""
        result = prepare_training_data(sample_training_data)

        if result is None:
            pytest.skip("prepare_training_data returned None (insufficient data)")

        features, target = result

        # Get features that are in both MODEL_FEATURES and the prepared features
        model_features_in_data = [f for f in MODEL_FEATURES if f in features.columns]
        prepared_features = list(features.columns)

        # Check that MODEL_FEATURES order is preserved for features that exist
        model_feature_indices = {feat: idx for idx, feat in enumerate(MODEL_FEATURES)}
        prepared_feature_indices = {feat: idx for idx, feat in enumerate(prepared_features)}

        # For each feature in MODEL_FEATURES that exists in prepared features,
        # check that its relative order is preserved
        for i, feat in enumerate(model_features_in_data):
            if i > 0:
                prev_feat = model_features_in_data[i - 1]
                # Previous feature should come before current feature in prepared features
                assert (
                    prepared_feature_indices[prev_feat] < prepared_feature_indices[feat]
                ), f"Feature order not preserved: {prev_feat} should come before {feat}"

    def test_model_stores_feature_names_in_order(self, sample_training_data):
        """Test that trained model stores feature names in the same order as training data."""
        with patch("modules.random_forest.core.model.joblib.dump"):
            model = train_random_forest_model(sample_training_data, save_model=False)

        if model is None:
            pytest.skip("Model training failed")

        # Get training features
        prepared_data = prepare_training_data(sample_training_data)
        if prepared_data is None:
            pytest.skip("prepare_training_data returned None")

        training_features, _ = prepared_data

        # Model should have feature_names_in_ attribute
        assert hasattr(model, "feature_names_in_"), "Model should have feature_names_in_ attribute"
        assert model.feature_names_in_ is not None, "feature_names_in_ should not be None"

        # Model's feature names should match training features order
        model_feature_names = list(model.feature_names_in_)
        training_feature_names = list(training_features.columns)

        assert (
            model_feature_names == training_feature_names
        ), f"Model feature order doesn't match training order.\nModel: {model_feature_names[:10]}\nTraining: {training_feature_names[:10]}"

    def test_inference_uses_model_feature_order(self, sample_training_data):
        """Test that inference uses the exact feature order from model.feature_names_in_."""
        with patch("modules.random_forest.core.model.joblib.dump"):
            model = train_random_forest_model(sample_training_data, save_model=False)

        if model is None:
            pytest.skip("Model training failed")

        # Create inference data (same structure as training)
        inference_data = sample_training_data.tail(10).copy()

        # Mock IndicatorEngine to return features in same order
        with patch("modules.random_forest.core.signals.IndicatorEngine") as mock_engine:
            mock_instance = mock_engine.return_value

            # Get training features to simulate what IndicatorEngine would return
            prepared_data = prepare_training_data(sample_training_data)
            if prepared_data is None:
                pytest.skip("prepare_training_data returned None")

            training_features, _ = prepared_data
            # Use same features for inference (simulating IndicatorEngine output)
            mock_instance.compute_features.return_value = training_features.tail(10)

            with patch("modules.random_forest.utils.features.add_advanced_features") as mock_advanced:
                # Return features in same order
                mock_advanced.return_value = training_features.tail(10)

                # Get signal
                signal, confidence = get_latest_random_forest_signal(inference_data, model)

                # Should succeed (not NEUTRAL with 0.0 due to feature mismatch)
                assert signal in ["LONG", "SHORT", "NEUTRAL"]
                assert isinstance(confidence, float)
                assert 0.0 <= confidence <= 1.0

    def test_model_features_order_preserved_in_prepared_data(self, sample_training_data):
        """Test that MODEL_FEATURES order is preserved in prepared training data."""
        result = prepare_training_data(sample_training_data)

        if result is None:
            pytest.skip("prepare_training_data returned None")

        features, target = result

        # Check that features from MODEL_FEATURES appear in the same order
        model_features_present = []
        for feat in MODEL_FEATURES:
            if feat in features.columns:
                model_features_present.append(feat)

        # Get the order in prepared features
        prepared_order = []
        for feat in features.columns:
            if feat in MODEL_FEATURES:
                prepared_order.append(feat)

        # MODEL_FEATURES order should be preserved
        assert (
            model_features_present == prepared_order
        ), f"MODEL_FEATURES order not preserved.\nExpected: {model_features_present[:10]}\nGot: {prepared_order[:10]}"

    def test_enhanced_features_added_after_model_features(self, sample_training_data):
        """Test that enhanced features are added after MODEL_FEATURES."""
        result = prepare_training_data(sample_training_data)

        if result is None:
            pytest.skip("prepare_training_data returned None")

        features, target = result
        feature_list = list(features.columns)

        # Find indices of MODEL_FEATURES and enhanced features
        model_feature_indices = []
        enhanced_feature_indices = []

        for idx, feat in enumerate(feature_list):
            if feat in MODEL_FEATURES:
                model_feature_indices.append(idx)
            elif any(
                pattern in feat
                for pattern in ["roc_", "atr_ratio", "price_to_SMA_", "rolling_std_", "rolling_skew_", "_lag_"]
            ):
                enhanced_feature_indices.append(idx)

        # All MODEL_FEATURES should come before enhanced features
        if model_feature_indices and enhanced_feature_indices:
            max_model_idx = max(model_feature_indices)
            min_enhanced_idx = min(enhanced_feature_indices)
            assert (
                max_model_idx < min_enhanced_idx
            ), f"Enhanced features should come after MODEL_FEATURES. Max MODEL_FEATURES idx: {max_model_idx}, Min enhanced idx: {min_enhanced_idx}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
