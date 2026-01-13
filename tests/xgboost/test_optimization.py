"""
Tests for XGBoost hyperparameter optimization module.

Tests cover:
- StudyManager: save/load studies, caching
- HyperparameterTuner: optimization, objective function, edge cases
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import optuna
import pandas as pd

from config import MODEL_FEATURES, TARGET_HORIZON, TARGET_LABELS, XGBOOST_PARAMS
from modules.xgboost.optimization import HyperparameterTuner, StudyManager


def _synthetic_df(rows=300):
    """Tạo synthetic DataFrame với đầy đủ features và target."""
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
        "DOJI": rng.integers(0, 2, rows),
        "HAMMER": rng.integers(0, 2, rows),
        "INVERTED_HAMMER": rng.integers(0, 2, rows),
        "SHOOTING_STAR": rng.integers(0, 2, rows),
        "MARUBOZU_BULL": rng.integers(0, 2, rows),
        "MARUBOZU_BEAR": rng.integers(0, 2, rows),
        "SPINNING_TOP": rng.integers(0, 2, rows),
        "DRAGONFLY_DOJI": rng.integers(0, 2, rows),
        "GRAVESTONE_DOJI": rng.integers(0, 2, rows),
        "BULLISH_ENGULFING": rng.integers(0, 2, rows),
        "BEARISH_ENGULFING": rng.integers(0, 2, rows),
        "BULLISH_HARAMI": rng.integers(0, 2, rows),
        "BEARISH_HARAMI": rng.integers(0, 2, rows),
        "HARAMI_CROSS_BULL": rng.integers(0, 2, rows),
        "HARAMI_CROSS_BEAR": rng.integers(0, 2, rows),
        "MORNING_STAR": rng.integers(0, 2, rows),
        "EVENING_STAR": rng.integers(0, 2, rows),
        "PIERCING": rng.integers(0, 2, rows),
        "DARK_CLOUD": rng.integers(0, 2, rows),
        "THREE_WHITE_SOLDIERS": rng.integers(0, 2, rows),
        "THREE_BLACK_CROWS": rng.integers(0, 2, rows),
        "THREE_INSIDE_UP": rng.integers(0, 2, rows),
        "THREE_INSIDE_DOWN": rng.integers(0, 2, rows),
        "TWEEZER_TOP": rng.integers(0, 2, rows),
        "TWEEZER_BOTTOM": rng.integers(0, 2, rows),
        "RISING_WINDOW": rng.integers(0, 2, rows),
        "FALLING_WINDOW": rng.integers(0, 2, rows),
        "TASUKI_GAP_BULL": rng.integers(0, 2, rows),
        "TASUKI_GAP_BEAR": rng.integers(0, 2, rows),
        "MAT_HOLD_BULL": rng.integers(0, 2, rows),
        "MAT_HOLD_BEAR": rng.integers(0, 2, rows),
        "ADVANCE_BLOCK": rng.integers(0, 2, rows),
        "STALLED_PATTERN": rng.integers(0, 2, rows),
        "BELT_HOLD_BULL": rng.integers(0, 2, rows),
        "BELT_HOLD_BEAR": rng.integers(0, 2, rows),
        "KICKER_BULL": rng.integers(0, 2, rows),
        "KICKER_BEAR": rng.integers(0, 2, rows),
        "HANGING_MAN": rng.integers(0, 2, rows),
    }
    df = pd.DataFrame(data)
    # Đảm bảo có đủ tất cả classes
    target = rng.integers(0, len(TARGET_LABELS), rows)
    # Đảm bảo mỗi class xuất hiện ít nhất một lần
    target[: len(TARGET_LABELS)] = np.arange(len(TARGET_LABELS))
    df["Target"] = target
    return df


# ==================== StudyManager Tests ====================


def test_study_manager_init():
    """Test StudyManager initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StudyManager(storage_dir=tmpdir)
        assert manager.storage_dir.exists()
        assert manager.storage_dir.is_dir()


def test_study_manager_save_study():
    """Test saving study metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StudyManager(storage_dir=tmpdir)

        # Tạo mock study
        study = Mock()
        study.study_name = "test_study"
        study.direction = Mock()
        study.direction.name = "maximize"
        study.trials = [
            Mock(
                number=0,
                value=0.85,
                params={"learning_rate": 0.1, "max_depth": 5},
                state=optuna.trial.TrialState.COMPLETE,
            ),
            Mock(
                number=1,
                value=0.87,
                params={"learning_rate": 0.15, "max_depth": 6},
                state=optuna.trial.TrialState.COMPLETE,
            ),
        ]

        best_params = {"learning_rate": 0.15, "max_depth": 6}
        best_score = 0.87

        filepath = manager.save_study(
            study=study,
            symbol="BTCUSDT",
            timeframe="1h",
            best_params=best_params,
            best_score=best_score,
        )

        # Kiểm tra file được tạo
        assert Path(filepath).exists()

        # Kiểm tra nội dung
        with open(filepath, "r", encoding="utf-8") as f:
            study_data = json.load(f)

        assert study_data["symbol"] == "BTCUSDT"
        assert study_data["timeframe"] == "1h"
        assert study_data["best_score"] == 0.87
        assert study_data["best_params"] == best_params
        assert study_data["n_trials"] == 2
        assert len(study_data["trials"]) == 2


def test_study_manager_load_best_params_not_found():
    """Test loading best params when no study exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StudyManager(storage_dir=tmpdir)
        params = manager.load_best_params("BTCUSDT", "1h")
        assert params is None


def test_study_manager_load_best_params():
    """Test loading best params from existing study."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StudyManager(storage_dir=tmpdir)

        # Tạo mock study và save
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

        # Load params
        loaded_params = manager.load_best_params("BTCUSDT", "1h", max_age_days=30)
        assert loaded_params == best_params


def test_study_manager_load_best_params_old_study():
    """Test loading best params from old study (should return None)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StudyManager(storage_dir=tmpdir)

        # Tạo mock study với timestamp cũ
        study = Mock()
        study.study_name = "test_study"
        study.direction = Mock()
        study.direction.name = "maximize"
        study.trials = []

        best_params = {"learning_rate": 0.15, "max_depth": 6}
        best_score = 0.87

        # Save study
        filepath = manager.save_study(
            study=study,
            symbol="BTCUSDT",
            timeframe="1h",
            best_params=best_params,
            best_score=best_score,
        )

        # Sửa timestamp trong file để làm cho nó cũ
        with open(filepath, "r", encoding="utf-8") as f:
            study_data = json.load(f)

        old_timestamp = (datetime.now() - timedelta(days=35)).strftime("%Y%m%d_%H%M%S")
        study_data["timestamp"] = old_timestamp

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(study_data, f, indent=2)

        # Load params với max_age_days=30
        loaded_params = manager.load_best_params("BTCUSDT", "1h", max_age_days=30)
        assert loaded_params is None


# ==================== HyperparameterTuner Tests ====================


def test_hyperparameter_tuner_init():
    """Test HyperparameterTuner initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h", storage_dir=tmpdir)
        assert tuner.symbol == "BTCUSDT"
        assert tuner.timeframe == "1h"
        assert tuner.study_manager.storage_dir.exists()


def test_hyperparameter_tuner_objective():
    """Test _objective function với mock trial."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h", storage_dir=tmpdir)

        df = _synthetic_df(rows=500)  # Đủ data để có valid folds
        X = df[MODEL_FEATURES]
        y = df["Target"].astype(int)

        # Tạo mock trial
        trial = Mock()
        trial.suggest_int = Mock(
            side_effect=lambda name, low, high, **kwargs: {
                "n_estimators": 100,
                "max_depth": 5,
                "min_child_weight": 3,
            }.get(name, (low + high) // 2)
        )
        trial.suggest_float = Mock(
            side_effect=lambda name, low, high, **kwargs: {
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0.1,
            }.get(name, (low + high) / 2)
        )

        # Chạy objective
        score = tuner._objective(trial, X, y, n_splits=3)

        # Kiểm tra score hợp lệ
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_hyperparameter_tuner_objective_insufficient_data():
    """Test _objective với data không đủ (ít hơn TARGET_HORIZON)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h", storage_dir=tmpdir)

        # Tạo data ít hơn TARGET_HORIZON
        df = _synthetic_df(rows=TARGET_HORIZON - 1)
        X = df[MODEL_FEATURES]
        y = df["Target"].astype(int)

        trial = Mock()
        trial.suggest_int = Mock(return_value=100)
        trial.suggest_float = Mock(return_value=0.1)

        score = tuner._objective(trial, X, y, n_splits=2)

        # Nên return 0.0 vì không có valid folds
        assert score == 0.0


def test_hyperparameter_tuner_objective_no_valid_folds():
    """Test _objective khi không có valid folds (thiếu classes)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h", storage_dir=tmpdir)

        df = _synthetic_df(rows=200)
        # Chỉ có 1 class (sẽ không có valid folds)
        df["Target"] = 0
        X = df[MODEL_FEATURES]
        y = df["Target"].astype(int)

        trial = Mock()
        trial.suggest_int = Mock(return_value=100)
        trial.suggest_float = Mock(return_value=0.1)

        score = tuner._objective(trial, X, y, n_splits=2)

        # Nên return 0.0 vì không có valid folds
        assert score == 0.0


def test_hyperparameter_tuner_optimize_insufficient_data():
    """Test optimize với data không đủ (< 100 samples)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h", storage_dir=tmpdir)

        df = _synthetic_df(rows=50)  # Ít hơn 100

        # Mock optuna để tránh tạo study thực sự
        with patch("modules.xgboost.optimization.optuna") as mock_optuna:
            best_params = tuner.optimize(df, n_trials=10)

            # Nên return default params
            assert best_params == XGBOOST_PARAMS.copy()
            # Không nên gọi optuna
            mock_optuna.create_study.assert_not_called()


@patch("modules.xgboost.optimization.optuna")
def test_hyperparameter_tuner_optimize(mock_optuna):
    """Test optimize với đủ data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h", storage_dir=tmpdir)

        df = _synthetic_df(rows=300)

        # Mock study
        mock_study = Mock()
        mock_study.best_params = {"learning_rate": 0.15, "max_depth": 6}
        mock_study.best_value = 0.87
        mock_study.study_name = "test_study"
        mock_study.direction = Mock()
        mock_study.direction.name = "maximize"
        mock_study.trials = []

        mock_optuna.load_study.side_effect = ValueError("Study not found")
        mock_optuna.create_study.return_value = mock_study

        best_params = tuner.optimize(df, n_trials=5, load_existing=False)

        # Kiểm tra best_params có các fields cần thiết
        assert "learning_rate" in best_params
        assert "max_depth" in best_params
        assert "random_state" in best_params
        assert "objective" in best_params
        assert "num_class" in best_params

        # Kiểm tra study.optimize được gọi
        mock_study.optimize.assert_called_once()


@patch("modules.xgboost.optimization.optuna")
def test_hyperparameter_tuner_optimize_load_existing(mock_optuna):
    """Test optimize với load existing study."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h", storage_dir=tmpdir)

        df = _synthetic_df(rows=300)

        # Mock existing study
        mock_study = Mock()
        mock_study.best_params = {"learning_rate": 0.15, "max_depth": 6}
        mock_study.best_value = 0.87
        mock_study.study_name = "test_study"
        mock_study.direction = Mock()
        mock_study.direction.name = "maximize"
        mock_study.trials = []

        mock_optuna.load_study.return_value = mock_study

        best_params = tuner.optimize(df, n_trials=5, load_existing=True)

        # Kiểm tra load_study được gọi
        mock_optuna.load_study.assert_called_once()
        # Không nên tạo study mới
        mock_optuna.create_study.assert_not_called()

        assert "learning_rate" in best_params
        assert "max_depth" in best_params


def test_hyperparameter_tuner_get_best_params_cached():
    """Test get_best_params với cached params."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h", storage_dir=tmpdir)

        df = _synthetic_df(rows=300)

        # Tạo và save study trước
        study = Mock()
        study.study_name = "test_study"
        study.direction = Mock()
        study.direction.name = "maximize"
        study.trials = []

        cached_params = {"learning_rate": 0.15, "max_depth": 6}
        tuner.study_manager.save_study(
            study=study,
            symbol="BTCUSDT",
            timeframe="1h",
            best_params=cached_params,
            best_score=0.87,
        )

        # Get best params với use_cached=True
        best_params = tuner.get_best_params(df, n_trials=10, use_cached=True)

        assert best_params == cached_params


def test_hyperparameter_tuner_get_best_params_no_cache():
    """Test get_best_params khi không có cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h", storage_dir=tmpdir)

        df = _synthetic_df(rows=300)

        # Mock optimize để tránh chạy thực sự
        with patch.object(tuner, "optimize") as mock_optimize:
            mock_optimize.return_value = {"learning_rate": 0.15, "max_depth": 6}

            best_params = tuner.get_best_params(df, n_trials=10, use_cached=True)

            # Nên gọi optimize
            mock_optimize.assert_called_once_with(df, n_trials=10)
            assert best_params == {"learning_rate": 0.15, "max_depth": 6}


def test_hyperparameter_tuner_get_best_params_use_cached_false():
    """Test get_best_params với use_cached=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h", storage_dir=tmpdir)

        df = _synthetic_df(rows=300)

        # Tạo cached params
        study = Mock()
        study.study_name = "test_study"
        study.direction = Mock()
        study.direction.name = "maximize"
        study.trials = []

        cached_params = {"learning_rate": 0.15, "max_depth": 6}
        tuner.study_manager.save_study(
            study=study,
            symbol="BTCUSDT",
            timeframe="1h",
            best_params=cached_params,
            best_score=0.87,
        )

        # Mock optimize
        with patch.object(tuner, "optimize") as mock_optimize:
            mock_optimize.return_value = {"learning_rate": 0.2, "max_depth": 7}

            best_params = tuner.get_best_params(df, n_trials=10, use_cached=False)

            # Nên gọi optimize thay vì dùng cache
            mock_optimize.assert_called_once_with(df, n_trials=10)
            assert best_params == {"learning_rate": 0.2, "max_depth": 7}


def test_hyperparameter_tuner_objective_with_whitelist():
    """Test _objective với classifier có whitelist (fallback compatibility)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tuner = HyperparameterTuner(symbol="BTCUSDT", timeframe="1h", storage_dir=tmpdir)

        df = _synthetic_df(rows=500)
        X = df[MODEL_FEATURES]
        y = df["Target"].astype(int)

        # Mock classifier với whitelist và predict method
        original_cls = tuner.classifier_cls

        # Tạo mock instance với predict trả về array hợp lệ
        mock_instance = Mock()

        # Mock predict để trả về array có cùng length với y_test
        def mock_predict(X_test):
            # Trả về array với cùng length, giá trị random trong range [0, len(TARGET_LABELS))
            return np.random.randint(0, len(TARGET_LABELS), size=len(X_test))

        mock_instance.predict = Mock(side_effect=mock_predict)
        mock_instance.fit = Mock(return_value=None)

        # Mock class với whitelist
        mock_cls = Mock(return_value=mock_instance)
        mock_cls.XGB_PARAM_WHITELIST = {"learning_rate", "max_depth", "random_state"}
        tuner.classifier_cls = mock_cls

        trial = Mock()
        trial.suggest_int = Mock(return_value=5)
        trial.suggest_float = Mock(return_value=0.1)

        try:
            score = tuner._objective(trial, X, y, n_splits=3)
            # Nên chạy được (có thể return 0.0 nếu không có valid folds)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
        finally:
            tuner.classifier_cls = original_cls
