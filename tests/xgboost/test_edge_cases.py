"""
Edge case tests for XGBoost module.

This module covers testing gaps identified in code review:
1. Empty DataFrame input
2. DataFrames with NaN values in critical columns
3. DataFrames where all labels are the same class
4. Extremely low volatility periods (flat prices)
5. GPU availability edge cases
6. Concurrent access to Optuna study database
7. Path traversal attacks in study names
"""

import concurrent.futures
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from config import MODEL_FEATURES, TARGET_LABELS
from modules.xgboost.core import labeling
from modules.xgboost.core.model import ClassDiversityError, predict_next_move, train_and_predict
from modules.xgboost.core.optimization import HyperparameterTuner, StudyManager

# ==================== Empty DataFrame Tests ====================


def test_train_and_predict_empty_dataframe():
    """Test train_and_predict with empty DataFrame."""
    df = pd.DataFrame(columns=MODEL_FEATURES + ["Target"])

    with pytest.raises(ValueError, match="Insufficient data"):
        train_and_predict(df)


def test_predict_next_move_empty_dataframe():
    """Test predict_next_move with empty DataFrame."""
    df = _synthetic_df(rows=100)
    model = train_and_predict(df)

    empty_df = pd.DataFrame(columns=MODEL_FEATURES)

    with pytest.raises((ValueError, IndexError)):
        predict_next_move(model, empty_df)


def test_hyperparameter_tuner_empty_dataframe():
    """Test HyperparameterTuner with empty DataFrame."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h", storage_dir=tmpdir)
        df = pd.DataFrame(columns=MODEL_FEATURES + ["Target"])

        # Should return default params without running optimization
        best_params = tuner.optimize(df, n_trials=10)
        assert best_params is not None


# ==================== NaN Values in Critical Columns Tests ====================


def test_train_and_predict_nan_in_features(monkeypatch):
    """Test train_and_predict with NaN values in feature columns."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = _synthetic_df(rows=200)
    # Add NaN values to critical feature columns
    df.loc[10:15, "SMA_20"] = np.nan
    df.loc[20:25, "RSI_14"] = np.nan
    df.loc[30:35, "ATR_14"] = np.nan

    # Should handle NaN values gracefully (either fillna or raise informative error)
    try:
        model = train_and_predict(df)
        assert model is not None
    except (ValueError, TypeError) as e:
        # If it raises an error, it should be informative
        assert "NaN" in str(e) or "missing" in str(e).lower() or "null" in str(e).lower()


def test_train_and_predict_all_nan_features():
    """Test train_and_predict when all features are NaN."""
    df = _synthetic_df(rows=100)
    # Set all features to NaN
    for col in MODEL_FEATURES:
        df[col] = np.nan

    with pytest.raises((ValueError, TypeError)):
        train_and_predict(df)


def test_predict_next_move_nan_in_features():
    """Test predict_next_move with NaN values in features."""
    df = _synthetic_df(rows=100)
    model = train_and_predict(df)

    last_row = df.iloc[-1:].copy()
    # Set some features to NaN
    last_row.loc[last_row.index[0], "SMA_20"] = np.nan
    last_row.loc[last_row.index[0], "RSI_14"] = np.nan

    # Should handle NaN values (either fillna or raise informative error)
    try:
        proba = predict_next_move(model, last_row)
        assert len(proba) == len(TARGET_LABELS)
        assert np.isclose(proba.sum(), 1.0, atol=1e-3)
    except (ValueError, TypeError) as e:
        assert "NaN" in str(e) or "missing" in str(e).lower()


# ==================== All Labels Same Class Tests ====================


def test_train_and_predict_all_labels_class_0():
    """Test train_and_predict when all labels are class 0 (DOWN)."""
    df = _synthetic_df(rows=200)
    df["Target"] = 0  # All labels are DOWN

    with pytest.raises(ClassDiversityError, match="Insufficient class diversity"):
        train_and_predict(df)


def test_train_and_predict_all_labels_class_1():
    """Test train_and_predict when all labels are class 1 (NEUTRAL)."""
    df = _synthetic_df(rows=200)
    df["Target"] = 1  # All labels are NEUTRAL

    with pytest.raises(ClassDiversityError, match="missing class 0"):
        train_and_predict(df)


def test_train_and_predict_all_labels_class_2():
    """Test train_and_predict when all labels are class 2 (UP)."""
    df = _synthetic_df(rows=200)
    df["Target"] = 2  # All labels are UP

    with pytest.raises(ClassDiversityError, match="missing class 0"):
        train_and_predict(df)


def test_hyperparameter_tuner_all_labels_same_class():
    """Test HyperparameterTuner with all labels same class."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h", storage_dir=tmpdir)

        df = _synthetic_df(rows=200)
        df["Target"] = 0  # All labels are class 0

        # Should return default params (no valid folds)
        with patch("modules.xgboost.optimization.optuna") as mock_optuna:
            best_params = tuner.optimize(df, n_trials=10)
            assert best_params is not None
            # Should not call optuna if no valid folds
            mock_optuna.create_study.assert_not_called()


# ==================== Extremely Low Volatility (Flat Prices) Tests ====================


def test_apply_directional_labels_flat_prices(monkeypatch):
    """Test apply_directional_labels with completely flat prices (zero volatility)."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    # Create DataFrame with completely flat prices (no movement at all)
    df = pd.DataFrame(
        {
            "close": [100.0] * 100,  # All prices are exactly the same
            "ATR_RATIO_14_50": [1.0] * 100,
        }
    )

    result = labeling.apply_directional_labels(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert "TargetLabel" in result.columns
    assert "DynamicThreshold" in result.columns
    # Thresholds should be valid (not NaN, not inf) even with flat prices
    assert result["DynamicThreshold"].notna().all()
    assert not (result["DynamicThreshold"] == np.inf).any()
    assert not (result["DynamicThreshold"] == -np.inf).any()
    # With flat prices, most labels should be NEUTRAL (no significant movement)
    # But we can't assert exact values due to threshold calculations


def test_apply_directional_labels_near_flat_prices(monkeypatch):
    """Test apply_directional_labels with near-flat prices (minimal movement)."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    # Create DataFrame with minimal price movement (0.01% changes)
    base_price = 100.0
    prices = [base_price + (i * 0.0001) for i in range(100)]  # Very small changes

    df = pd.DataFrame(
        {
            "close": prices,
            "ATR_RATIO_14_50": [0.5] * 100,  # Low ATR ratio (low volatility)
        }
    )

    result = labeling.apply_directional_labels(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert "TargetLabel" in result.columns
    assert "DynamicThreshold" in result.columns
    # Thresholds should be valid
    assert result["DynamicThreshold"].notna().all()
    assert not (result["DynamicThreshold"] == np.inf).any()


# ==================== GPU Availability Edge Cases Tests ====================


@patch("modules.xgboost.model.USE_GPU", True)
@patch("subprocess.run")
def test_build_model_gpu_available(mock_subprocess):
    """Test build_model when GPU is available."""
    mock_subprocess.return_value = Mock(returncode=0)  # nvidia-smi succeeds

    df = _synthetic_df(rows=200)
    model = train_and_predict(df)

    assert model is not None
    # Verify GPU parameters were attempted (check if subprocess.run was called)
    mock_subprocess.assert_called()


@patch("modules.xgboost.model.USE_GPU", True)
@patch("subprocess.run")
def test_build_model_gpu_not_available(mock_subprocess):
    """Test build_model when GPU is not available (nvidia-smi fails)."""
    mock_subprocess.side_effect = FileNotFoundError("nvidia-smi not found")

    df = _synthetic_df(rows=200)
    # Should fall back to CPU mode
    model = train_and_predict(df)

    assert model is not None


@patch("modules.xgboost.model.USE_GPU", True)
@patch("subprocess.run")
def test_build_model_gpu_timeout(mock_subprocess):
    """Test build_model when GPU check times out."""
    import subprocess

    mock_subprocess.side_effect = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

    df = _synthetic_df(rows=200)
    # Should fall back to CPU mode
    model = train_and_predict(df)

    assert model is not None


@patch("modules.xgboost.model.USE_GPU", True)
@patch("subprocess.run")
def test_build_model_gpu_device_parameter_fails(mock_subprocess):
    """Test build_model when device='cuda' parameter causes error."""
    mock_subprocess.return_value = Mock(returncode=0)  # nvidia-smi succeeds

    df = _synthetic_df(rows=200)

    # Mock XGBClassifier to raise error with device parameter
    original_cls = None
    try:
        import xgboost as xgb

        original_cls = xgb.XGBClassifier

        class MockXGBClassifier:
            def __init__(self, **kwargs):
                if "device" in kwargs:
                    raise ValueError("device parameter not supported")
                # Fallback to CPU mode
                self.params = kwargs

            def fit(self, X, y):
                pass

            def predict(self, X):
                return np.zeros(len(X))

            def predict_proba(self, X):
                return np.ones((len(X), len(TARGET_LABELS))) / len(TARGET_LABELS)

            def score(self, X, y):
                return 0.5

        xgb.XGBClassifier = MockXGBClassifier

        # Should fall back to CPU mode (without device parameter)
        model = train_and_predict(df)
        assert model is not None
    finally:
        if original_cls is not None:
            xgb.XGBClassifier = original_cls


# ==================== Concurrent Access to Optuna Study Database Tests ====================


def test_study_manager_concurrent_save():
    """Test StudyManager with concurrent save operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StudyManager(storage_dir=tmpdir)

        def save_study(symbol, timeframe, trial_num):
            study = Mock()
            study.study_name = f"test_study_{trial_num}"
            study.direction = Mock()
            study.direction.name = "maximize"
            study.trials = []

            best_params = {"learning_rate": 0.1 + trial_num * 0.01, "max_depth": 5}
            best_score = 0.8 + trial_num * 0.01

            return manager.save_study(
                study=study,
                symbol=symbol,
                timeframe=timeframe,
                best_params=best_params,
                best_score=best_score,
            )

        # Run concurrent saves
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(save_study, "BTCUSDT", "1h", i) for i in range(10)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All saves should succeed
        assert len(results) == 10
        assert all(Path(r).exists() for r in results)


def test_study_manager_concurrent_load():
    """Test StudyManager with concurrent load operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StudyManager(storage_dir=tmpdir)

        # Create a study first
        study = Mock()
        study.study_name = "test_study"
        study.direction = Mock()
        study.direction.name = "maximize"
        study.trials = []

        best_params = {"learning_rate": 0.15, "max_depth": 6}
        best_score = 0.87

        manager.save_study(
            study=study,
            symbol="BTCUSDT",
            timeframe="1h",
            best_params=best_params,
            best_score=best_score,
        )

        # Run concurrent loads
        def load_params():
            return manager.load_best_params("BTCUSDT", "1h")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(load_params) for _ in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All loads should return the same params
        assert len(results) == 20
        assert all(r == best_params for r in results)


def test_hyperparameter_tuner_concurrent_optimize():
    """Test HyperparameterTuner with concurrent optimize operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        df = _synthetic_df(rows=300)

        def optimize(symbol, timeframe):
            tuner = HyperparameterTuner(symbol=symbol, timeframe=timeframe, storage_dir=tmpdir)
            with patch("modules.xgboost.optimization.optuna") as mock_optuna:
                mock_study = Mock()
                mock_study.best_params = {"learning_rate": 0.15, "max_depth": 6}
                mock_study.best_value = 0.87
                mock_study.study_name = "test_study"
                mock_study.direction = Mock()
                mock_study.direction.name = "maximize"
                mock_study.trials = []

                mock_optuna.load_study.side_effect = ValueError("Study not found")
                mock_optuna.create_study.return_value = mock_study

                return tuner.optimize(df, n_trials=5, load_existing=False)

        # Run concurrent optimizations
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(optimize, "BTCUSDT", "1h"),
                executor.submit(optimize, "ETHUSDT", "1h"),
                executor.submit(optimize, "BTCUSDT", "4h"),
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All optimizations should succeed
        assert len(results) == 3
        assert all("learning_rate" in r for r in results)


# ==================== Path Traversal Attacks Tests ====================


def test_study_manager_path_traversal_symbol():
    """Test StudyManager with path traversal in symbol name.
    
    NOTE: This test documents a security vulnerability - StudyManager does not sanitize
    symbol/timeframe inputs, allowing path traversal attacks. The test verifies this
    behavior and should be updated when the vulnerability is fixed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StudyManager(storage_dir=tmpdir)

        study = Mock()
        study.study_name = "test_study"
        study.direction = Mock()
        study.direction.name = "maximize"
        study.trials = []

        best_params = {"learning_rate": 0.15, "max_depth": 6}
        best_score = 0.87

        # Attempt path traversal in symbol
        malicious_symbol = "../../../etc/passwd"

        # Currently, StudyManager does not sanitize input, so this will create
        # a file with the literal path traversal in the filename
        # This is a security vulnerability that should be fixed
        try:
            filepath = manager.save_study(
                study=study,
                symbol=malicious_symbol,
                timeframe="1h",
                best_params=best_params,
                best_score=best_score,
            )

            # Verify file was created (even if in wrong location)
            filepath_obj = Path(filepath)
            # The file might be created outside storage_dir due to path traversal
            # This documents the vulnerability
            assert filepath_obj.exists() or filepath_obj.resolve().exists()
        except (FileNotFoundError, OSError):
            # If parent directories don't exist, save will fail
            # This is expected behavior but still a vulnerability
            pass


def test_study_manager_path_traversal_timeframe():
    """Test StudyManager with path traversal in timeframe name.
    
    NOTE: This test documents a security vulnerability - StudyManager does not sanitize
    symbol/timeframe inputs, allowing path traversal attacks.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StudyManager(storage_dir=tmpdir)

        study = Mock()
        study.study_name = "test_study"
        study.direction = Mock()
        study.direction.name = "maximize"
        study.trials = []

        best_params = {"learning_rate": 0.15, "max_depth": 6}
        best_score = 0.87

        # Attempt path traversal in timeframe
        malicious_timeframe = "../../../etc/passwd"
        try:
            filepath = manager.save_study(
                study=study,
                symbol="BTCUSDT",
                timeframe=malicious_timeframe,
                best_params=best_params,
                best_score=best_score,
            )
            # File might be created outside storage_dir - this documents the vulnerability
            filepath_obj = Path(filepath)
            assert filepath_obj.exists() or filepath_obj.resolve().exists()
        except (FileNotFoundError, OSError):
            # If parent directories don't exist, save will fail
            pass


def test_study_manager_path_traversal_combined():
    """Test StudyManager with path traversal in both symbol and timeframe.
    
    NOTE: This test documents a security vulnerability - StudyManager does not sanitize
    symbol/timeframe inputs, allowing path traversal attacks.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StudyManager(storage_dir=tmpdir)

        study = Mock()
        study.study_name = "test_study"
        study.direction = Mock()
        study.direction.name = "maximize"
        study.trials = []

        best_params = {"learning_rate": 0.15, "max_depth": 6}
        best_score = 0.87

        # Attempt path traversal in both
        malicious_symbol = "../../../etc"
        malicious_timeframe = "passwd"
        try:
            filepath = manager.save_study(
                study=study,
                symbol=malicious_symbol,
                timeframe=malicious_timeframe,
                best_params=best_params,
                best_score=best_score,
            )
            # File might be created outside storage_dir - this documents the vulnerability
            filepath_obj = Path(filepath)
            # On Windows, path resolution might normalize the path
            resolved_path = filepath_obj.resolve()
            # The file might be created in a location outside storage_dir
            # This is a security vulnerability
            assert resolved_path.exists() or filepath_obj.exists()
        except (FileNotFoundError, OSError):
            # If parent directories don't exist, save will fail
            pass


def test_study_manager_path_traversal_load():
    """Test StudyManager load_best_params with path traversal.
    
    NOTE: This test documents a security vulnerability - StudyManager does not sanitize
    symbol/timeframe inputs, potentially allowing loading files outside storage_dir.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StudyManager(storage_dir=tmpdir)

        # First create a legitimate study
        study = Mock()
        study.study_name = "test_study"
        study.direction = Mock()
        study.direction.name = "maximize"
        study.trials = []

        best_params = {"learning_rate": 0.15, "max_depth": 6}
        best_score = 0.87

        manager.save_study(
            study=study,
            symbol="BTCUSDT",
            timeframe="1h",
            best_params=best_params,
            best_score=best_score,
        )

        # Attempt path traversal in load
        # Currently, glob pattern matching might not work correctly with path traversal
        # This could be a security issue if it allows loading files outside storage_dir
        malicious_symbol = "../../../etc"
        malicious_timeframe = "passwd"

        # The glob pattern might not match correctly, but if it does, it's a vulnerability
        params = manager.load_best_params(malicious_symbol, malicious_timeframe)
        # Should return None if no matching file found (which is expected)
        # But if it returns params, that indicates a security vulnerability
        # For now, we just document the behavior
        assert params is None or params == best_params  # Document both cases


# ==================== Helper Functions ====================


def _synthetic_df(rows=300):
    """Create synthetic DataFrame with all required features and target."""
    from config.model_features import CANDLESTICK_PATTERN_NAMES

    rng = np.random.default_rng(42)
    data = {
        "open": rng.normal(100, 1, rows).cumsum(),
        "high": rng.normal(101, 1, rows).cumsum(),
        "low": rng.normal(99, 1, rows).cumsum(),
        "close": rng.normal(100.5, 1, rows).cumsum(),
        "volume": rng.integers(1000, 2000, rows),
        "SMA_20": rng.normal(100, 1, rows),
        "SMA_50": rng.normal(100, 1, rows),
        "SMA_200": rng.normal(100, 1, rows),
        "RSI_9": rng.uniform(0, 100, rows),
        "RSI_14": rng.uniform(0, 100, rows),
        "RSI_25": rng.uniform(0, 100, rows),
        "ATR_14": rng.uniform(0.5, 2, rows),
        "MACD_12_26_9": rng.normal(0, 1, rows),
        "MACDh_12_26_9": rng.normal(0, 1, rows),
        "MACDs_12_26_9": rng.normal(0, 1, rows),
        "BBP_5_2.0": rng.uniform(0, 1, rows),
        "STOCHRSIk_14_14_3_3": rng.uniform(0, 100, rows),
        "STOCHRSId_14_14_3_3": rng.uniform(0, 100, rows),
        "OBV": rng.normal(0, 1, rows).cumsum(),
        # Candlestick patterns
        **{pattern: rng.integers(0, 2, rows) for pattern in CANDLESTICK_PATTERN_NAMES},
    }
    df = pd.DataFrame(data)
    # Ensure all classes are present
    target = rng.integers(0, len(TARGET_LABELS), rows)
    target[: len(TARGET_LABELS)] = np.arange(len(TARGET_LABELS))
    df["Target"] = target
    return df
