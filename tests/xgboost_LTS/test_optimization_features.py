"""
Tests for XGBoost LTS optimization features (Task 1) using pytest.
"""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import TimeSeriesSplit

# Ensure modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from modules.xgboost_LTS.utils.gpu_utils import detect_cuda_available, get_gpu_info
from modules.xgboost_LTS.utils.cv_parallel import run_parallel_cv
from modules.xgboost_LTS.core.optimization import HyperparameterTuner
from config import TARGET_LABELS


@pytest.fixture(autouse=True)
def clear_gpu_cache():
    """Clear GPU cache before each test."""
    detect_cuda_available.cache_clear()
    get_gpu_info.cache_clear()


@patch("modules.xgboost_LTS.utils.gpu_utils.subprocess.run")
def test_detect_cuda_available_success(mock_run):
    """Test that CUDA detection returns True when nvidia-smi succeeds."""
    # Mock successful subprocess call
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_run.return_value = mock_process

    assert detect_cuda_available() is True
    mock_run.assert_called_once()


@patch("modules.xgboost_LTS.utils.gpu_utils.subprocess.run")
def test_detect_cuda_available_caching(mock_run):
    """Test that result is cached after first call."""
    # Mock successful subprocess call
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_run.return_value = mock_process

    # First call
    result1 = detect_cuda_available()
    assert result1 is True

    # Second call should not trigger subprocess again
    result2 = detect_cuda_available()
    assert result2 is True

    # Verify subprocess was only called once
    mock_run.assert_called_once()


@patch("modules.xgboost_LTS.utils.gpu_utils.subprocess.run")
def test_get_gpu_info(mock_run):
    """Test GPU info retrieval."""
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "Tesla T4"
    mock_run.return_value = mock_process

    info = get_gpu_info()
    assert info == "Tesla T4"


@pytest.fixture
def cv_data():
    """Fixture for Cross Validation data."""
    n_samples = 100
    X = pd.DataFrame({"feature1": np.random.randn(n_samples), "feature2": np.random.randn(n_samples)})
    y = pd.Series(np.random.randint(0, len(TARGET_LABELS), n_samples))
    tscv = TimeSeriesSplit(n_splits=3)
    params = {"n_estimators": 2, "max_depth": 2, "num_class": len(TARGET_LABELS)}
    return X, y, tscv, params


def test_run_parallel_cv_integration(cv_data):
    """Test actual execution of parallel CV (integration test)."""
    X, y, tscv, params = cv_data
    # This runs a real parallel execution with a small dataset
    cv_scores, all_y_true, all_y_pred = run_parallel_cv(X, y, tscv, params, max_workers=2)

    # Check if we got results (might be empty if gaps consume all data,
    # but with 100 samples and default configs it usually produces something
    # unless TARGET_HORIZON is very large)
    # Even if empty, it should not raise an exception.
    assert isinstance(cv_scores, list)
    assert isinstance(all_y_true, list)
    assert isinstance(all_y_pred, list)


@patch("modules.xgboost_LTS.utils.cv_parallel.ProcessPoolExecutor")
def test_run_parallel_cv_mocked(MockExecutor, cv_data):
    """Test parallel CV with mocked executor to verify plumbing."""
    X, y, tscv, params = cv_data

    # Setup mock future result
    mock_future = MagicMock()
    mock_future.result.return_value = (1, 0.85, [0, 1], [0, 1], "Success")

    # Setup mock executor context manager
    mock_executor_instance = MockExecutor.return_value.__enter__.return_value
    mock_executor_instance.submit.return_value = mock_future

    # We need to mock as_completed as well since it iterates over futures
    with patch("modules.xgboost_LTS.utils.cv_parallel.as_completed") as mock_as_completed:
        mock_as_completed.return_value = [mock_future]

        run_parallel_cv(X, y, tscv, params, max_workers=2)

        # Verify submit was called (number of splits times)
        assert mock_executor_instance.submit.call_count == 3


@patch("modules.xgboost_LTS.core.optimization.optuna.create_study")
@patch("modules.xgboost_LTS.core.optimization.file_lock")
def test_optimize_parallel_config(mock_lock, mock_create_study):
    """Test that n_jobs=-1 is passed to study.optimize."""
    tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h")
    tuner.study_manager = MagicMock()  # Mock storage interaction

    # Mock dataframe not strictly needed here since we bypass validation via patch

    mock_study = MagicMock()
    mock_study.best_value = 0.85  # Set a float value for formatting
    mock_study.best_params = {"n_estimators": 100}
    mock_create_study.return_value = mock_study

    # We need to bypass the data validation checks if possible, or provide valid data
    # Let's patch the validation logic or providing a DF that passes `MODEL_FEATURES` check
    # Instead, let's just patch `MODEL_FEATURES` to match our dummy DF
    with patch("modules.xgboost_LTS.core.optimization.MODEL_FEATURES", ["close"]):
        # Re-create df with just 'close' and 'Target'
        df_simple = pd.DataFrame({"close": np.random.randn(500), "Target": np.random.randint(0, 3, 500)})

        tuner.optimize(df_simple, n_trials=1, load_existing=False)

        # Verify optimize was called with n_jobs=-1
        args, kwargs = mock_study.optimize.call_args
        assert kwargs.get("n_jobs") == -1
        assert kwargs.get("gc_after_trial") is True
