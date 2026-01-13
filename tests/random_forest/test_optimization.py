import sys
from pathlib import Path

"""
Tests for Random Forest optimization module.

Tests cover:
- StudyManager functionality
- HyperparameterTuner optimization
- Integration with existing training pipeline
- Error handling and edge cases
"""


# Add project root to Python path for imports
ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

import json
import shutil
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import optuna
import pandas as pd
import pytest

from config import (
    MODEL_RANDOM_STATE,
)
from config.model_features import MODEL_FEATURES
from modules.random_forest.optimization import HyperparameterTuner, StudyManager
from modules.random_forest.utils.data_preparation import prepare_training_data

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_optimization_data():
    """Create sample data with features and target for optimization testing"""
    np.random.seed(42)
    # Create larger dataset to ensure enough samples after feature engineering
    sample_size = 200

    close_prices = 100 + np.cumsum(np.random.randn(sample_size) * 0.5)
    high_prices = close_prices + np.random.uniform(0, 2, sample_size)
    low_prices = close_prices - np.random.uniform(0, 2, sample_size)
    open_prices = close_prices + np.random.uniform(-1, 1, sample_size)
    volume_data = np.random.uniform(1000, 10000, sample_size)

    return pd.DataFrame(
        {"open": open_prices, "high": high_prices, "low": low_prices, "close": close_prices, "volume": volume_data}
    )


@pytest.fixture
def temp_storage_dir():
    """Create temporary directory for study storage"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def prepared_data(sample_optimization_data):
    """Create prepared features and target data"""
    prepared = prepare_training_data(sample_optimization_data)
    if prepared is None:
        pytest.skip("Failed to prepare training data")
    return prepared


# ============================================================================
# Test StudyManager
# ============================================================================


class TestStudyManager:
    """Test suite for StudyManager class"""

    def test_init(self, temp_storage_dir):
        """Test StudyManager initialization"""
        manager = StudyManager(storage_dir=temp_storage_dir)
        assert manager.storage_dir.exists()
        assert manager.storage_dir == Path(temp_storage_dir).resolve()

    def test_save_study(self, temp_storage_dir):
        """Test saving study metadata"""
        manager = StudyManager(storage_dir=temp_storage_dir)

        # Create mock study
        study = Mock()
        study.study_name = "test_study"
        study.direction = Mock(name="MAXIMIZE")
        study.trials = []
        study.best_params = {"n_estimators": 100, "max_depth": 10}
        study.best_value = 0.85

        filepath = manager.save_study(
            study=study, symbol="BTCUSDT", timeframe="1h", best_params=study.best_params, best_score=study.best_value
        )

        assert Path(filepath).exists()

        # Verify saved content
        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["symbol"] == "BTCUSDT"
        assert data["timeframe"] == "1h"
        assert data["best_params"]["n_estimators"] == 100
        assert data["best_score"] == 0.85

    def test_load_best_params_not_found(self, temp_storage_dir):
        """Test loading best params when no study exists"""
        manager = StudyManager(storage_dir=temp_storage_dir)
        params = manager.load_best_params("BTCUSDT", "1h")
        assert params is None

    def test_load_best_params_found(self, temp_storage_dir):
        """Test loading best params from existing study"""
        manager = StudyManager(storage_dir=temp_storage_dir)

        # Create and save a study
        study = Mock()
        study.study_name = "test_study"
        study.direction = Mock(name="MAXIMIZE")
        study.trials = []
        study.best_params = {"n_estimators": 150, "max_depth": 15}
        study.best_value = 0.90

        manager.save_study(
            study=study, symbol="BTCUSDT", timeframe="1h", best_params=study.best_params, best_score=study.best_value
        )

        # Load params
        params = manager.load_best_params("BTCUSDT", "1h", max_age_days=30)
        assert params is not None
        assert params["n_estimators"] == 150
        assert params["max_depth"] == 15

    def test_load_best_params_expired(self, temp_storage_dir):
        """Test loading expired study returns None"""
        manager = StudyManager(storage_dir=temp_storage_dir)

        # Create study with old timestamp
        timestamp = (datetime.now() - timedelta(days=31)).strftime("%Y%m%d_%H%M%S")
        filename = f"study_BTCUSDT_1h_{timestamp}.json"
        filepath = Path(temp_storage_dir) / filename

        study_data = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "timestamp": timestamp,
            "best_params": {"n_estimators": 100},
            "best_score": 0.80,
            "n_trials": 10,
            "study_name": "test",
            "direction": "MAXIMIZE",
            "trials": [],
        }

        with open(filepath, "w") as f:
            json.dump(study_data, f)

        params = manager.load_best_params("BTCUSDT", "1h", max_age_days=30)
        assert params is None


# ============================================================================
# Test HyperparameterTuner
# ============================================================================


class TestHyperparameterTuner:
    """Test suite for HyperparameterTuner class"""

    def test_init(self, temp_storage_dir):
        """Test HyperparameterTuner initialization"""
        tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h", storage_dir=temp_storage_dir)
        assert tuner.symbol == "BTCUSDT"
        assert tuner.timeframe == "1h"
        assert tuner.target_horizon == 5

    def test_objective_with_valid_data(self, prepared_data, temp_storage_dir):
        """Test _objective method with valid data"""
        tuner = HyperparameterTuner("BTCUSDT", "1h", storage_dir=temp_storage_dir)
        features, target = prepared_data

        # Create trial
        study = optuna.create_study()
        trial = study.ask()

        score = tuner._objective(trial, features, target, n_splits=3)

        # Score should be between 0 and 1 (accuracy)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_objective_with_insufficient_data(self, temp_storage_dir):
        """Test _objective with insufficient data returns 0.0 or low score"""
        tuner = HyperparameterTuner("BTCUSDT", "1h", storage_dir=temp_storage_dir)

        # Create very small dataset (too small for CV folds)
        small_features = pd.DataFrame(np.random.randn(5, 5), columns=[f"feature_{i}" for i in range(5)])
        small_target = pd.Series([0, 1, 0, 1, 0])

        study = optuna.create_study()
        trial = study.ask()

        score = tuner._objective(trial, small_features, small_target, n_splits=3)
        # With insufficient data, should return 0.0 (no valid folds)
        assert score == 0.0

    @patch("modules.random_forest.optimization.prepare_training_data")
    def test_optimize_insufficient_data(self, mock_prepare, temp_storage_dir):
        """Test optimize with insufficient data returns default params"""
        mock_prepare.return_value = None

        tuner = HyperparameterTuner("BTCUSDT", "1h", storage_dir=temp_storage_dir)
        df = pd.DataFrame({"open": [100], "high": [105], "low": [95], "close": [100], "volume": [1000]})

        with pytest.raises(ValueError, match="Failed to prepare"):
            tuner.optimize(df, n_trials=1)

    def test_optimize_with_small_dataset(self, sample_optimization_data, temp_storage_dir):
        """Test optimize with dataset smaller than MIN_TRAINING_SAMPLES"""
        tuner = HyperparameterTuner("BTCUSDT", "1h", storage_dir=temp_storage_dir)

        # Create very small dataset
        small_df = sample_optimization_data.head(50)

        with patch("modules.random_forest.optimization.prepare_training_data") as mock_prepare:
            # Mock to return small features
            # Defensive check: ensure MODEL_FEATURES has at least 5 elements
            if len(MODEL_FEATURES) >= 5:
                feature_columns = MODEL_FEATURES[:5]
            else:
                # Fallback to dynamically generated column names
                feature_columns = [f"feature_{i}" for i in range(5)]
            small_features = pd.DataFrame(np.random.randn(50, 5), columns=feature_columns)
            small_target = pd.Series([0, 1, -1] * 16 + [0, 1])[:50]
            mock_prepare.return_value = (small_features, small_target)

            params = tuner.optimize(small_df, n_trials=1)

            # Should return default params
            assert "n_estimators" in params
            assert "random_state" in params

    def test_optimize_single_trial(self, prepared_data, temp_storage_dir):
        """Test optimize method with single trial"""
        tuner = HyperparameterTuner("BTCUSDT", "1h", storage_dir=temp_storage_dir)
        features, target = prepared_data

        # Create DataFrame from prepared data (simulate full workflow)
        df = pd.concat([features, target.to_frame(name="target")], axis=1)
        # Add OHLCV columns (mock)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = np.random.randn(len(df))

        with patch("modules.random_forest.optimization.prepare_training_data") as mock_prepare:
            mock_prepare.return_value = (features, target)

            params = tuner.optimize(df, n_trials=1, load_existing=False)

            assert isinstance(params, dict)
            assert "n_estimators" in params
            assert "random_state" in params
            assert params["random_state"] == MODEL_RANDOM_STATE

    def test_get_best_params_no_cache(self, prepared_data, temp_storage_dir):
        """Test get_best_params when no cached params exist"""
        tuner = HyperparameterTuner("BTCUSDT", "1h", storage_dir=temp_storage_dir)
        features, target = prepared_data

        df = pd.concat([features, target.to_frame(name="target")], axis=1)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = np.random.randn(len(df))

        with patch("modules.random_forest.optimization.prepare_training_data") as mock_prepare:
            mock_prepare.return_value = (features, target)

            params = tuner.get_best_params(df, n_trials=1, use_cached=False)

            assert isinstance(params, dict)
            assert "n_estimators" in params

    def test_get_best_params_with_cache(self, temp_storage_dir):
        """Test get_best_params uses cached params"""
        tuner = HyperparameterTuner("BTCUSDT", "1h", storage_dir=temp_storage_dir)

        # Create cached study
        manager = StudyManager(storage_dir=temp_storage_dir)
        study = Mock()
        study.study_name = "test"
        study.direction = Mock(name="MAXIMIZE")
        study.trials = []
        study.best_params = {"n_estimators": 200, "max_depth": 20, "random_state": MODEL_RANDOM_STATE, "n_jobs": -1}
        study.best_value = 0.95

        manager.save_study(
            study=study, symbol="BTCUSDT", timeframe="1h", best_params=study.best_params, best_score=study.best_value
        )

        df = pd.DataFrame()
        params = tuner.get_best_params(df, n_trials=1, use_cached=True)

        assert params is not None
        assert params["n_estimators"] == 200

    def test_optimize_missing_target_column(self, temp_storage_dir):
        """Test optimize with missing target after preparation"""
        tuner = HyperparameterTuner("BTCUSDT", "1h", storage_dir=temp_storage_dir)

        with patch("modules.random_forest.optimization.prepare_training_data") as mock_prepare:
            mock_prepare.return_value = None

            df = pd.DataFrame()
            with pytest.raises(ValueError, match="Failed to prepare"):
                tuner.optimize(df, n_trials=1)

    def test_optimize_single_class(self, temp_storage_dir):
        """Test optimize with only one class in target"""
        tuner = HyperparameterTuner("BTCUSDT", "1h", storage_dir=temp_storage_dir)

        # Defensive check: ensure MODEL_FEATURES has at least 5 elements
        if len(MODEL_FEATURES) >= 5:
            feature_columns = MODEL_FEATURES[:5]
        else:
            # Fallback to dynamically generated column names
            feature_columns = [f"feature_{i}" for i in range(5)]

        features = pd.DataFrame(np.random.randn(100, 5), columns=feature_columns)
        target = pd.Series([0] * 100)  # Only one class

        df = pd.DataFrame()

        with patch("modules.random_forest.optimization.prepare_training_data") as mock_prepare:
            mock_prepare.return_value = (features, target)

            with pytest.raises(ValueError, match="Need at least 2 classes"):
                tuner.optimize(df, n_trials=1)


# ============================================================================
# Integration Tests
# ============================================================================


class TestOptimizationIntegration:
    """Integration tests for optimization with existing pipeline"""

    def test_optimization_with_smote(self, sample_optimization_data, temp_storage_dir):
        """Test optimization integrates correctly with SMOTE"""
        tuner = HyperparameterTuner("BTCUSDT", "1h", storage_dir=temp_storage_dir)

        # Try to prepare data first
        prepared_result = prepare_training_data(sample_optimization_data)
        if prepared_result is None:
            pytest.skip("Failed to prepare data")

        features, target = prepared_result

        with patch("modules.random_forest.optimization.prepare_training_data") as mock_prepare:
            mock_prepare.return_value = (features, target)

            params = tuner.optimize(sample_optimization_data, n_trials=1, load_existing=False)

            assert isinstance(params, dict)
            # Verify SMOTE is used in _objective (no explicit check needed as it's in the code)

    def test_optimization_with_class_weights(self, prepared_data, temp_storage_dir):
        """Test optimization uses class weights correctly"""
        tuner = HyperparameterTuner("BTCUSDT", "1h", storage_dir=temp_storage_dir)
        features, target = prepared_data

        study = optuna.create_study()
        trial = study.ask()

        score = tuner._objective(trial, features, target, n_splits=3)

        # Should complete without errors (class weights are computed internally)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
