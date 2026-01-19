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

from modules.random_forest.utils.training import apply_sampling, create_model_and_weights


class TestApplySampling:
    """Test suite for apply_sampling function in training utils"""

    @pytest.fixture
    def sample_data(self):
        """Create sample features and target"""
        np.random.seed(42)
        features = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
        target = pd.Series(np.random.randint(0, 2, 100), name="target")
        return features, target

    @pytest.mark.parametrize("strategy", ["SMOTE", "ADASYN", "BorderlineSMOTE"])
    def test_apply_sampling_success(self, sample_data, strategy):
        """Test successful sampling application for various strategies"""
        features, target = sample_data

        # Patch the strategy in config
        with patch("modules.random_forest.utils.training.RANDOM_FOREST_SAMPLING_STRATEGY", strategy):
            # Mock the appropriate sampler
            with patch(f"modules.random_forest.utils.training.{strategy}") as mock_sampler_cls:
                mock_sampler = mock_sampler_cls.return_value
                resampled_features = pd.DataFrame(np.random.rand(150, 5), columns=features.columns)
                resampled_target = pd.Series(np.random.randint(0, 2, 150), name="target")
                mock_sampler.fit_resample.return_value = (resampled_features, resampled_target)

                result_features, result_target, applied = apply_sampling(features, target)

                assert len(result_features) == 150
                assert applied is True
                assert mock_sampler_cls.called

    def test_apply_sampling_none(self, sample_data):
        """Test sampling skipped when strategy is NONE"""
        features, target = sample_data
        with patch("modules.random_forest.utils.training.RANDOM_FOREST_SAMPLING_STRATEGY", "NONE"):
            result_features, result_target, applied = apply_sampling(features, target)
            assert len(result_features) == 100
            assert applied is False

    def test_apply_sampling_balanced_rf(self, sample_data):
        """Test sampling skipped when using BALANCED_RF (handled in model creation)"""
        features, target = sample_data
        with patch("modules.random_forest.utils.training.RANDOM_FOREST_SAMPLING_STRATEGY", "BALANCED_RF"):
            result_features, result_target, applied = apply_sampling(features, target)
            assert len(result_features) == 100
            assert applied is False

    def test_create_model_balanced_rf(self, sample_data):
        """Test that BalancedRandomForestClassifier is created when strategy is BALANCED_RF"""
        _, target = sample_data
        with (
            patch("modules.random_forest.utils.training.RANDOM_FOREST_SAMPLING_STRATEGY", "BALANCED_RF"),
            patch("modules.random_forest.utils.training.BALANCED_RF_AVAILABLE", True),
            patch("modules.random_forest.utils.training.BalancedRandomForestClassifier") as mock_brf,
        ):
            create_model_and_weights(target, sampling_applied=False)
            assert mock_brf.called
            # Ensure class_weight is NOT in params
            _, kwargs = mock_brf.call_args
            assert "class_weight" not in kwargs

    def test_create_model_default_rf(self, sample_data):
        """Test that standard RandomForestClassifier is created with class weights when no sampling applied"""
        _, target = sample_data
        with patch("modules.random_forest.utils.training.RANDOM_FOREST_SAMPLING_STRATEGY", "NONE"):
            with patch("modules.random_forest.utils.training.RandomForestClassifier") as mock_rf:
                create_model_and_weights(target, sampling_applied=False)
                assert mock_rf.called
                _, kwargs = mock_rf.call_args
                assert kwargs["class_weight"] is not None

    def test_apply_sampling_low_memory(self, sample_data):
        """Test sampling skipped when memory is low"""
        features, target = sample_data
        with patch("modules.random_forest.utils.training.psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.available = 100 * 1024 * 1024
            with patch("modules.random_forest.utils.training.log_warn") as mock_warn:
                _, _, applied = apply_sampling(features, target)
                assert applied is False
                assert mock_warn.called
                assert "Low memory" in mock_warn.call_args[0][0]

    def test_apply_sampling_runtime_error(self, sample_data):
        """Test fallback when SMOTE raises RuntimeError"""
        features, target = sample_data

        with patch("modules.random_forest.utils.training.SMOTE") as mock_sampler_cls:
            mock_sampler = mock_sampler_cls.return_value
            mock_sampler.fit_resample.side_effect = RuntimeError("SMOTE failure simulation")

            with patch("modules.random_forest.utils.training.log_warn") as mock_warn:
                _, _, applied = apply_sampling(features, target)

                # Should return original data
                assert applied is False
                assert mock_warn.called
                assert "SMOTE application failed" in mock_warn.call_args[0][0]

    def test_apply_sampling_unexpected_error(self, sample_data):
        """Test fallback when SMOTE raises unexpected exception"""
        features, target = sample_data

        with patch("modules.random_forest.utils.training.SMOTE") as mock_sampler_cls:
            mock_sampler = mock_sampler_cls.return_value
            mock_sampler.fit_resample.side_effect = Exception("Total Surprise")

            with patch("modules.random_forest.utils.training.log_error") as mock_err:
                _, _, applied = apply_sampling(features, target)

                # Should return original data
                assert applied is False
                assert mock_err.called
                assert "Unexpected error" in mock_err.call_args[0][0]

    def test_create_model_and_weights_without_sampling(self):
        """Test that class_weight is set when sampling is NOT applied"""
        # Create imbalanced target (before SMOTE)
        target_imbalanced = pd.Series([0, 0, 0, 0, 0, 1], name="target")

        model = create_model_and_weights(target_imbalanced, sampling_applied=False)

        # Without SMOTE, class_weight should be a dict
        assert model.class_weight is not None
        assert isinstance(model.class_weight, dict)
        assert 0 in model.class_weight
        assert 1 in model.class_weight

    def test_create_model_and_weights_with_sampling(self):
        """Test that class_weight is None when sampling is applied"""
        target = pd.Series([0, 1])
        with patch("modules.random_forest.utils.training.RandomForestClassifier") as mock_rf:
            create_model_and_weights(target, sampling_applied=True)
            assert mock_rf.called
            _, kwargs = mock_rf.call_args
            assert kwargs["class_weight"] is None
