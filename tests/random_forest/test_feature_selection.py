"""Tests for feature selection functionality."""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest

from modules.random_forest.utils.feature_selection import (
    select_features,
    select_features_mutual_info,
    select_features_rf_importance,
)


class TestMutualInfoFeatureSelection:
    """Test mutual information feature selection."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with informative and noise features."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        # Create informative features (correlated with target)
        informative_features = np.random.randn(n_samples, 3)
        target = (informative_features[:, 0] > 0).astype(int)

        # Create noise features (uncorrelated with target)
        noise_features = np.random.randn(n_samples, n_features - 3)

        # Combine features
        all_features = np.hstack([informative_features, noise_features])
        feature_names = [f"feature_{i}" for i in range(n_features)]

        features = pd.DataFrame(all_features, columns=feature_names)
        target_series = pd.Series(target, name="target")

        return features, target_series

    def test_select_features_mutual_info_basic(self, sample_data):
        """Test basic mutual information feature selection."""
        features, target = sample_data
        k = 5

        features_selected, selected_names, selector = select_features_mutual_info(features, target, k=k)

        assert len(features_selected.columns) == k
        assert len(selected_names) == k
        assert isinstance(selector, SelectKBest)
        assert all(name in features.columns for name in selected_names)

    def test_select_features_mutual_info_k_too_large(self, sample_data):
        """Test mutual info selection when k exceeds available features."""
        features, target = sample_data
        k = 1000  # Much larger than number of features

        features_selected, selected_names, selector = select_features_mutual_info(features, target, k=k)

        # Should use all available features
        assert len(features_selected.columns) == len(features.columns)
        assert len(selected_names) == len(features.columns)

    def test_select_features_mutual_info_k_equals_features(self, sample_data):
        """Test mutual info selection when k equals number of features."""
        features, target = sample_data
        k = len(features.columns)

        features_selected, selected_names, selector = select_features_mutual_info(features, target, k=k)

        # Should use all features
        assert len(features_selected.columns) == len(features.columns)


class TestRFImportanceFeatureSelection:
    """Test Random Forest importance-based feature selection."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with varying feature importance."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        # Create features with different importances
        features = np.random.randn(n_samples, n_features)
        # Make first 3 features more important
        target = (
            (features[:, 0] * 2 + features[:, 1] * 1.5 + features[:, 2] * 1.0) > 0
        ).astype(int)

        feature_names = [f"feature_{i}" for i in range(n_features)]
        features_df = pd.DataFrame(features, columns=feature_names)
        target_series = pd.Series(target, name="target")

        return features_df, target_series

    def test_select_features_rf_importance_basic(self, sample_data):
        """Test basic RF importance feature selection."""
        features, target = sample_data
        threshold = 0.01

        features_selected, selected_names, model = select_features_rf_importance(
            features, target, threshold=threshold
        )

        assert len(selected_names) > 0
        assert isinstance(model, RandomForestClassifier)
        assert all(name in features.columns for name in selected_names)
        assert len(features_selected.columns) == len(selected_names)

    def test_select_features_rf_importance_high_threshold(self, sample_data):
        """Test RF importance selection with high threshold."""
        features, target = sample_data
        threshold = 0.5  # Very high threshold

        features_selected, selected_names, model = select_features_rf_importance(
            features, target, threshold=threshold
        )

        # Should fallback to top 10 if no features above threshold
        assert len(selected_names) > 0
        assert len(selected_names) <= 10

    def test_select_features_rf_importance_with_pretrained_model(self, sample_data):
        """Test RF importance selection with pre-trained model."""
        features, target = sample_data

        # Pre-train a model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(features, target)

        threshold = 0.01

        features_selected, selected_names, returned_model = select_features_rf_importance(
            features, target, model=model, threshold=threshold
        )

        assert returned_model == model  # Should return the same model
        assert len(selected_names) > 0


class TestFeatureSelectionIntegration:
    """Test integrated feature selection function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_samples = 200
        features = pd.DataFrame(np.random.randn(n_samples, 10), columns=[f"feature_{i}" for i in range(10)])
        target = pd.Series(np.random.randint(0, 2, n_samples), name="target")
        return features, target

    def test_select_features_mutual_info_method(self, sample_data):
        """Test select_features with mutual_info method."""
        features, target = sample_data

        features_selected, selected_names, selector = select_features(features, target, method="mutual_info")

        assert len(selected_names) > 0
        # Selector may be None if k >= number of features
        if selector is not None:
            assert isinstance(selector, SelectKBest)

    def test_select_features_rf_importance_method(self, sample_data):
        """Test select_features with rf_importance method."""
        features, target = sample_data

        features_selected, selected_names, model = select_features(features, target, method="rf_importance")

        assert len(selected_names) > 0
        assert isinstance(model, RandomForestClassifier)

    def test_select_features_unknown_method(self, sample_data):
        """Test select_features with unknown method."""
        features, target = sample_data

        features_selected, selected_names, selector = select_features(features, target, method="unknown_method")

        # Should return all features
        assert len(selected_names) == len(features.columns)
        assert selector is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
