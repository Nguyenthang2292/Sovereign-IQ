"""Tests for expanded hyperparameter search space in optimization.

This module tests that the hyperparameter search space includes all
the new parameters: criterion, min_impurity_decrease, and bootstrap.
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
import optuna
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from modules.random_forest.optimization import HyperparameterTuner
from modules.random_forest.utils.training import create_model_and_weights


class TestExpandedSearchSpace:
    """Test expanded hyperparameter search space."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        features = pd.DataFrame(np.random.randn(n_samples, 10), columns=[f"feature_{i}" for i in range(10)])
        target = pd.Series(np.random.randint(0, 3, n_samples), name="target")
        return features, target

    def test_all_hyperparameters_suggested(self, sample_data, tmp_path):
        """Test that all hyperparameters are suggested in the objective function."""
        tuner = HyperparameterTuner("BTCUSDT", "1h", storage_dir=str(tmp_path))
        features, target = sample_data

        study = optuna.create_study()
        trial = study.ask()

        # Track all suggest calls
        suggest_calls = {}

        original_suggest_int = trial.suggest_int
        original_suggest_categorical = trial.suggest_categorical
        original_suggest_float = trial.suggest_float

        def tracked_suggest_int(name, low, high, step=None, **kwargs):
            if step is not None:
                result = original_suggest_int(name, low=low, high=high, step=step, **kwargs)
            else:
                result = original_suggest_int(name, low=low, high=high, **kwargs)
            suggest_calls[name] = {"type": "int", "low": low, "high": high, "step": step, "value": result}
            return result

        def tracked_suggest_categorical(name, choices, **kwargs):
            result = original_suggest_categorical(name, choices=choices, **kwargs)
            suggest_calls[name] = {"type": "categorical", "choices": choices, "value": result}
            return result

        def tracked_suggest_float(name, low, high, **kwargs):
            result = original_suggest_float(name, low=low, high=high, **kwargs)
            suggest_calls[name] = {"type": "float", "low": low, "high": high, "value": result}
            return result

        trial.suggest_int = tracked_suggest_int
        trial.suggest_categorical = tracked_suggest_categorical
        trial.suggest_float = tracked_suggest_float

        # Call objective
        score = tuner._objective(trial, features, target, n_splits=3)

        # Verify all required hyperparameters are suggested
        required_params = [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "criterion",
            "min_impurity_decrease",
            "bootstrap",
        ]

        for param in required_params:
            assert param in suggest_calls, f"Parameter {param} was not suggested"

        # Verify specific ranges/choices
        assert suggest_calls["n_estimators"]["low"] == 100
        assert suggest_calls["n_estimators"]["high"] == 1000
        assert suggest_calls["n_estimators"]["step"] == 100

        assert suggest_calls["max_depth"]["low"] == 0
        assert suggest_calls["max_depth"]["high"] == 50

        assert suggest_calls["min_samples_split"]["low"] == 2
        assert suggest_calls["min_samples_split"]["high"] == 50

        assert suggest_calls["min_samples_leaf"]["low"] == 1
        assert suggest_calls["min_samples_leaf"]["high"] == 20

        assert set(suggest_calls["max_features"]["choices"]) == {"sqrt", "log2", 0.5, 0.7, None}
        assert set(suggest_calls["criterion"]["choices"]) == {"gini", "entropy", "log_loss"}
        assert suggest_calls["min_impurity_decrease"]["low"] == 0.0
        assert suggest_calls["min_impurity_decrease"]["high"] == 0.1
        assert set(suggest_calls["bootstrap"]["choices"]) == {True, False}

        # Score should be valid
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_model_creation_with_new_hyperparameters(self):
        """Test that create_model_and_weights supports new hyperparameters."""
        target = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

        # Test with all new hyperparameters
        custom_params = {
            "criterion": "entropy",
            "min_impurity_decrease": 0.01,
            "bootstrap": False,
            "n_estimators": 50,
            "max_depth": 10,
        }

        model = create_model_and_weights(target, custom_params=custom_params, sampling_applied=False)

        assert isinstance(model, RandomForestClassifier)
        assert model.criterion == "entropy"
        assert model.min_impurity_decrease == 0.01
        assert model.bootstrap is False
        assert model.n_estimators == 50
        assert model.max_depth == 10

    def test_model_creation_with_criterion_log_loss(self):
        """Test model creation with log_loss criterion."""
        target = pd.Series([0, 1, 0, 1, 0, 1])

        custom_params = {"criterion": "log_loss"}

        model = create_model_and_weights(target, custom_params=custom_params, sampling_applied=False)

        assert isinstance(model, RandomForestClassifier)
        assert model.criterion == "log_loss"

    def test_model_creation_with_bootstrap_false(self):
        """Test model creation with bootstrap=False."""
        target = pd.Series([0, 1, 0, 1, 0, 1])

        custom_params = {"bootstrap": False}

        model = create_model_and_weights(target, custom_params=custom_params, sampling_applied=False)

        assert isinstance(model, RandomForestClassifier)
        assert model.bootstrap is False

    def test_model_creation_with_min_impurity_decrease(self):
        """Test model creation with min_impurity_decrease."""
        target = pd.Series([0, 1, 0, 1, 0, 1])

        custom_params = {"min_impurity_decrease": 0.05}

        model = create_model_and_weights(target, custom_params=custom_params, sampling_applied=False)

        assert isinstance(model, RandomForestClassifier)
        assert model.min_impurity_decrease == 0.05

    def test_model_creation_with_max_features_float(self):
        """Test model creation with float max_features values."""
        target = pd.Series([0, 1, 0, 1, 0, 1])

        # Test with 0.5
        custom_params_05 = {"max_features": 0.5}
        model_05 = create_model_and_weights(target, custom_params=custom_params_05, sampling_applied=False)
        assert model_05.max_features == 0.5

        # Test with 0.7
        custom_params_07 = {"max_features": 0.7}
        model_07 = create_model_and_weights(target, custom_params=custom_params_07, sampling_applied=False)
        assert model_07.max_features == 0.7


class TestEndToEndOptimization:
    """Test end-to-end optimization with expanded search space."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for optimization."""
        np.random.seed(42)
        n_samples = 300

        # Create features
        features = np.random.randn(n_samples, 10)
        # Create target with some correlation
        target = ((features[:, 0] * 2 + features[:, 1] * 1.5) > 0).astype(int)

        features_df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(10)])
        target_series = pd.Series(target, name="target")

        return features_df, target_series

    def test_optimization_with_expanded_search_space(self, sample_data, tmp_path):
        """Test that optimization works with expanded search space."""
        features, target = sample_data

        tuner = HyperparameterTuner("BTCUSDT", "1h", storage_dir=str(tmp_path))

        # Create mock prepare_training_data
        with patch("modules.random_forest.optimization.prepare_training_data") as mock_prepare:
            mock_prepare.return_value = (features, target)

            # Create minimal DataFrame with OHLCV columns
            df = pd.DataFrame(
                {
                    "open": np.random.randn(len(features)),
                    "high": np.random.randn(len(features)),
                    "low": np.random.randn(len(features)),
                    "close": np.random.randn(len(features)),
                    "volume": np.random.randn(len(features)),
                }
            )

            # Run optimization with very few trials for speed
            best_params = tuner.optimize(df, n_trials=2, load_existing=False)

            # Verify best_params contains all expected keys
            assert isinstance(best_params, dict)
            assert "n_estimators" in best_params
            assert "random_state" in best_params

            # Verify new hyperparameters can be in best_params (if suggested by Optuna)
            # Note: They may or may not be present depending on Optuna's suggestions
            # But the model should be able to use them if present
            if "criterion" in best_params:
                assert best_params["criterion"] in ["gini", "entropy", "log_loss"]
            if "bootstrap" in best_params:
                assert isinstance(best_params["bootstrap"], bool)
            if "min_impurity_decrease" in best_params:
                assert 0.0 <= best_params["min_impurity_decrease"] <= 0.1

    def test_model_training_with_optimized_params(self, sample_data):
        """Test that model can be trained with optimized parameters."""
        features, target = sample_data

        # Simulate optimized parameters (including new ones)
        optimized_params = {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": 0.7,
            "criterion": "entropy",
            "min_impurity_decrease": 0.01,
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1,
        }

        # Create model with optimized params
        model = RandomForestClassifier(**optimized_params)

        # Train model
        model.fit(features, target)

        # Verify model was trained successfully
        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == features.shape[1]

        # Verify predictions work
        predictions = model.predict(features[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
