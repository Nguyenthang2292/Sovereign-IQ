from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from modules.auto_trade.core.atc_scanner import SignalResult
from modules.auto_trade.core.xgboost_filter import XGBoostFilter


@pytest.fixture
def mock_data_fetcher():
    return MagicMock()


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict_proba = MagicMock(return_value=np.array([[0.1, 0.8, 0.1]]))
    model.n_classes_ = 3
    return model


@pytest.fixture
def mock_joblib_load(mock_model):
    """Patch joblib.load where the filter uses it so the filter gets the mock model."""
    with patch("modules.auto_trade.core.xgboost_filter.joblib.load", return_value=mock_model) as mock:
        yield mock


# Path.exists() in _load_model: path.exists(), then path_native.exists() (x3).
# .joblib exists, .json not -> loader uses joblib.load.
_PATH_EXISTS_JOBLIB_ONLY = [True, False, False, False]


@pytest.fixture
def mock_predict_next_move():
    with patch("modules.auto_trade.core.xgboost_filter.predict_next_move") as mock:
        yield mock


@pytest.fixture
def mock_indicator_engine():
    with patch("modules.auto_trade.core.xgboost_filter.IndicatorEngine") as mock_engine_cls:
        engine = MagicMock()
        mock_engine_cls.return_value = engine
        engine.compute_features.side_effect = lambda df: df  # No-op
        yield engine


@pytest.fixture
def mock_add_advanced_features():
    with patch("modules.auto_trade.core.xgboost_filter.add_advanced_features") as mock:
        mock.side_effect = lambda df: df  # No-op
        yield mock


def test_init_load_model(mock_data_fetcher, mock_joblib_load):
    with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
        filter = XGBoostFilter(mock_data_fetcher, "model.joblib")
        mock_joblib_load.assert_called_once()
        assert filter.model is not None


def test_init_model_not_found(mock_data_fetcher, mock_joblib_load):
    with patch("pathlib.Path.exists", return_value=False):
        filter = XGBoostFilter(
            mock_data_fetcher, "missing.joblib", config={"require_model": False}
        )
        mock_joblib_load.assert_not_called()
        assert filter.model is None


def test_filter_pass_long(
    mock_data_fetcher, mock_joblib_load, mock_predict_next_move, mock_indicator_engine, mock_add_advanced_features
):
    with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
        filter = XGBoostFilter(mock_data_fetcher, "model.joblib", config={"min_confidence": 0.6, "min_required_candles": 100})

        # Signals input (symbol, score, signal_type, details, strengths)
        signals = [
            SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0})
        ]

        # Mock Data Fetcher
        mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 100})

        # Mock Prediction: 0.1 DOWN, 0.1 NEUTRAL, 0.8 UP
        mock_predict_next_move.return_value = np.array([0.1, 0.1, 0.8])

        filtered = filter.filter_signals(signals)

        assert len(filtered) == 1
        assert filtered[0].symbol == "BTCUSDT"
        assert filtered[0].details["xgboost_conf"] == 0.8
        assert filtered[0].details["xgboost_dir"] == "UP"


def test_filter_reject_contradiction(
    mock_data_fetcher, mock_joblib_load, mock_predict_next_move, mock_indicator_engine, mock_add_advanced_features
):
    with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
        filter = XGBoostFilter(
            mock_data_fetcher, "model.joblib", config={"min_confidence": 0.6, "min_required_candles": 100}
        )
        # Signals input: LONG (symbol, score, signal_type, details, strengths)
        signals = [
            SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0})
        ]

        # Mock Data Fetcher
        mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 100})

        # Mock Prediction: 0.8 DOWN (Contradiction)
        mock_predict_next_move.return_value = np.array([0.8, 0.1, 0.1])

        filtered = filter.filter_signals(signals)

        assert len(filtered) == 0


def test_filter_reject_low_confidence(
    mock_data_fetcher, mock_joblib_load, mock_predict_next_move, mock_indicator_engine, mock_add_advanced_features
):
    with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
        filter = XGBoostFilter(
            mock_data_fetcher, "model.joblib", config={"min_confidence": 0.8, "min_required_candles": 100}
        )
        # Signals input: LONG (symbol, score, signal_type, details, strengths)
        signals = [
            SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0})
        ]

        # Mock Data Fetcher
        mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 100})

        # Mock Prediction: 0.7 UP (Matches direction, but < 0.8 min_confidence)
        mock_predict_next_move.return_value = np.array([0.1, 0.2, 0.7])

        filtered = filter.filter_signals(signals)

        assert len(filtered) == 0


def test_no_model_returns_all(mock_data_fetcher, mock_joblib_load):
    with patch("pathlib.Path.exists", return_value=False):
        filter = XGBoostFilter(
            mock_data_fetcher, "missing.joblib", config={"require_model": False}
        )

        signals = [SignalResult("BTCUSDT", 1.0, "LONG", {}, {})]
        # Should return signals as-is if model missing
        filtered = filter.filter_signals(signals)
        assert len(filtered) == 1
        assert filtered[0].symbol == "BTCUSDT"


# ============================================================================
# Initialization & Validation Tests (7 tests total)
# ============================================================================


class TestXGBoostFilterInitialization:
    """Tests for XGBoostFilter initialization and validation."""

    def test_init_with_default_config(self, mock_data_fetcher, mock_joblib_load):
        """Test initialization with default configuration."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib")

            assert filter.min_confidence == 0.3  # from config XGBOOST_FILTER_DEFAULTS
            assert filter.history_limit == 1500
            assert filter.prediction_timeframe == "5m"
            assert filter.on_error == "drop"
            assert filter.min_required_candles == 250

    def test_init_with_custom_config(self, mock_data_fetcher, mock_joblib_load):
        """Test initialization with custom configuration."""
        config = {
            "min_confidence": 0.7,
            "history_limit": 2000,
            "prediction_timeframe": "15m",
            "on_error": "pass",
            "min_required_candles": 300,
        }

        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib", config=config)

            assert filter.min_confidence == 0.7
            assert filter.history_limit == 2000
            assert filter.prediction_timeframe == "15m"
            assert filter.on_error == "pass"
            assert filter.min_required_candles == 300

    def test_init_invalid_min_confidence(self, mock_data_fetcher, mock_joblib_load):
        """Test that invalid min_confidence raises ValueError."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            # Too high
            with pytest.raises(ValueError, match="min_confidence must be between 0 and 1"):
                XGBoostFilter(mock_data_fetcher, "model.joblib", config={"min_confidence": 1.5})

            # Too low
            with pytest.raises(ValueError, match="min_confidence must be between 0 and 1"):
                XGBoostFilter(mock_data_fetcher, "model.joblib", config={"min_confidence": -0.1})

    def test_init_invalid_history_limit(self, mock_data_fetcher, mock_joblib_load):
        """Test that invalid history_limit raises ValueError."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            with pytest.raises(ValueError, match="history_limit must be positive"):
                XGBoostFilter(mock_data_fetcher, "model.joblib", config={"history_limit": 0})

            with pytest.raises(ValueError, match="history_limit must be positive"):
                XGBoostFilter(mock_data_fetcher, "model.joblib", config={"history_limit": -100})

    def test_init_invalid_prediction_timeframe(self, mock_data_fetcher, mock_joblib_load):
        """Test that invalid prediction_timeframe raises ValueError."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            with pytest.raises(ValueError, match="Invalid prediction_timeframe"):
                XGBoostFilter(mock_data_fetcher, "model.joblib", config={"prediction_timeframe": "3m"})

    def test_init_invalid_on_error(self, mock_data_fetcher, mock_joblib_load):
        """Test that invalid on_error raises ValueError."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            with pytest.raises(ValueError, match="on_error must be 'drop', 'pass', or 'neutral'"):
                XGBoostFilter(mock_data_fetcher, "model.joblib", config={"on_error": "invalid"})


# ============================================================================
# Model Loading & Security Tests (6 tests total)
# ============================================================================


class TestXGBoostFilterModelLoading:
    """Tests for model loading and security."""

    def test_load_model_success(self, mock_data_fetcher, mock_joblib_load):
        """Test successful model loading."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib")
            assert filter.model is not None
            mock_joblib_load.assert_called_once()

    def test_load_model_file_not_found(self, mock_data_fetcher, mock_joblib_load):
        """Test handling of missing model file."""
        with patch("pathlib.Path.exists", return_value=False):
            filter = XGBoostFilter(
                mock_data_fetcher, "missing.joblib", config={"require_model": False}
            )
            assert filter.model is None
            mock_joblib_load.assert_not_called()

    def test_model_integrity_check_passes(self, mock_data_fetcher):
        """Test model integrity check with matching hash."""
        model = MagicMock()
        model.predict_proba = lambda x: np.array([[0.1, 0.1, 0.8]])
        model.n_classes_ = 3

        expected_hash = "abc123"
        config = {"model_hash": expected_hash}

        with (
            patch("modules.auto_trade.core.xgboost_filter.joblib.load", return_value=model),
            patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY),
            patch("hashlib.sha256") as mock_sha256,
            patch("builtins.open", MagicMock()),  # Mock file open for integrity check
        ):
            mock_hash = MagicMock()
            mock_hash.hexdigest.return_value = expected_hash
            mock_sha256.return_value = mock_hash

            filter = XGBoostFilter(mock_data_fetcher, "model.joblib", config=config)
            assert filter.model is not None

    def test_model_integrity_check_fails(self, mock_data_fetcher):
        """Test model integrity check with mismatching hash."""
        config = {"model_hash": "expected_hash", "require_model": False}

        with (
            patch("modules.auto_trade.core.xgboost_filter.joblib.load") as mock_load,
            patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY),
            patch("hashlib.sha256") as mock_sha256,
            patch("builtins.open", MagicMock()),
        ):
            mock_hash = MagicMock()
            mock_hash.hexdigest.return_value = "different_hash"
            mock_sha256.return_value = mock_hash

            filter = XGBoostFilter(mock_data_fetcher, "model.joblib", config=config)
            assert filter.model is None
            mock_load.assert_not_called()

    def test_load_model_no_hash_configured(self, mock_data_fetcher):
        """Test model loading without hash (should warn but load)."""
        model = MagicMock()
        model.predict_proba = lambda x: np.array([[0.1, 0.1, 0.8]])
        model.n_classes_ = 3

        with (
            patch("modules.auto_trade.core.xgboost_filter.joblib.load", return_value=model),
            patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY),
            patch("modules.auto_trade.core.xgboost_filter.log_warn") as mock_warn,
        ):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib")

            assert filter.model is not None
            mock_warn.assert_called()
            first_call = str(mock_warn.call_args_list[0])
            assert "model_hash" in first_call


# ============================================================================
# Signal Filtering Logic Tests (10 tests total)
# ============================================================================


class TestXGBoostFilterSignalFiltering:
    """Tests for signal filtering logic."""

    def test_filter_pass_long(
        self,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test filtering passes when model confirms LONG signal."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(
                mock_data_fetcher, "model.joblib", config={"min_confidence": 0.6, "min_required_candles": 100}
            )

            signals = [SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0})]

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 100})
            # Model predicts UP (confirms LONG)
            mock_predict_next_move.side_effect = None  # Reset any previous side_effect
            mock_predict_next_move.return_value = np.array([0.1, 0.1, 0.8])

            filtered = filter.filter_signals(signals)

            assert len(filtered) == 1
            assert filtered[0].symbol == "BTCUSDT"
            assert filtered[0].signal_type == "LONG"
            assert filtered[0].details["xgboost_conf"] == 0.8
            assert filtered[0].details["xgboost_dir"] == "UP"
            assert filtered[0].details["xgboost_validated"] is True

    def test_filter_pass_short(
        self,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test filtering passes when model confirms SHORT signal."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(
                mock_data_fetcher, "model.joblib", config={"min_confidence": 0.6, "min_required_candles": 100}
            )

            signals = [SignalResult("BTCUSDT", 1.0, "SHORT", {"1h": "SHORT"}, {"1h": -1.0})]

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 100})
            # Model predicts DOWN (confirms SHORT)
            mock_predict_next_move.return_value = np.array([0.8, 0.1, 0.1])

            filtered = filter.filter_signals(signals)

            assert len(filtered) == 1
            assert filtered[0].signal_type == "SHORT"
            assert filtered[0].details["xgboost_dir"] == "DOWN"

    def test_filter_reject_contradiction(
        self,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test filtering rejects when model contradicts signal."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(
                mock_data_fetcher, "model.joblib", config={"min_confidence": 0.6, "min_required_candles": 100}
            )

            signals = [SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0})]

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 100})
            # Model predicts DOWN (contradicts LONG)
            mock_predict_next_move.return_value = np.array([0.8, 0.1, 0.1])

            filtered = filter.filter_signals(signals)

            assert len(filtered) == 0

    def test_filter_reject_low_confidence(
        self,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test filtering rejects when confidence is below threshold."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(
                mock_data_fetcher, "model.joblib", config={"min_confidence": 0.8, "min_required_candles": 100}
            )

            signals = [SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0})]

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 100})
            # Model predicts UP but confidence 0.7 < 0.8
            mock_predict_next_move.return_value = np.array([0.1, 0.2, 0.7])

            filtered = filter.filter_signals(signals)

            assert len(filtered) == 0

    def test_filter_neutral_prediction(
        self,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test filtering rejects when model predicts NEUTRAL."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(
                mock_data_fetcher, "model.joblib", config={"min_confidence": 0.6, "min_required_candles": 100}
            )

            signals = [SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0})]

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 100})
            # Model predicts NEUTRAL
            mock_predict_next_move.return_value = np.array([0.2, 0.6, 0.2])

            filtered = filter.filter_signals(signals)

            assert len(filtered) == 0

    def test_filter_multiple_signals_mixed(
        self,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test filtering multiple signals with mixed results."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(
                mock_data_fetcher, "model.joblib", config={"min_confidence": 0.6, "min_required_candles": 100}
            )

            signals = [
                SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0}),
                SignalResult("ETHUSDT", 1.0, "SHORT", {"1h": "SHORT"}, {"1h": -1.0}),
                SignalResult("BNBUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0}),
            ]

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 100})

            # Use call count to return different predictions for each symbol
            call_count = 0

            def mock_predict(model, df):
                nonlocal call_count
                call_count += 1
                # First call (BTCUSDT): UP (confirms LONG)
                if call_count == 1:
                    return np.array([0.1, 0.1, 0.8])
                # Second call (ETHUSDT): DOWN (confirms SHORT)
                elif call_count == 2:
                    return np.array([0.8, 0.1, 0.1])
                # Third call (BNBUSDT): NEUTRAL (rejects LONG)
                else:
                    return np.array([0.2, 0.6, 0.2])

            mock_predict_next_move.side_effect = mock_predict

            filtered = filter.filter_signals(signals)

            # Should pass BTC and ETH, reject BNB
            assert len(filtered) == 2
            symbols_passed = [s.symbol for s in filtered]
            assert "BTCUSDT" in symbols_passed
            assert "ETHUSDT" in symbols_passed
            assert "BNBUSDT" not in symbols_passed

    def test_filter_empty_signal_list(self, mock_data_fetcher, mock_joblib_load):
        """Test filtering with empty signal list."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib")

            filtered = filter.filter_signals([])

            assert len(filtered) == 0

    def test_filter_preserves_original_details(
        self,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test that filtering preserves original signal details."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(
                mock_data_fetcher, "model.joblib", config={"min_confidence": 0.6, "min_required_candles": 100}
            )

            original_details = {"1h": "LONG", "15m": "LONG", "5m": "NEUTRAL"}
            signals = [SignalResult("BTCUSDT", 1.0, "LONG", original_details, {})]

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 100})
            mock_predict_next_move.return_value = np.array([0.1, 0.1, 0.8])

            filtered = filter.filter_signals(signals)

            # Original details should be preserved
            assert filtered[0].details["1h"] == "LONG"
            assert filtered[0].details["15m"] == "LONG"
            assert filtered[0].details["5m"] == "NEUTRAL"
            # Plus new XGBoost details
            assert "xgboost_conf" in filtered[0].details
            assert "xgboost_dir" in filtered[0].details

    def test_no_model_returns_all_signals(self, mock_data_fetcher, mock_joblib_load):
        """Test that all signals pass when model is not loaded."""
        with patch("pathlib.Path.exists", return_value=False):
            filter = XGBoostFilter(
                mock_data_fetcher, "missing.joblib", config={"require_model": False}
            )

            signals = [
                SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0}),
                SignalResult("ETHUSDT", 1.0, "SHORT", {"1h": "SHORT"}, {"1h": -1.0}),
            ]

            filtered = filter.filter_signals(signals)

            # Should return all signals unchanged
            assert len(filtered) == 2
            assert filtered[0].symbol == "BTCUSDT"
            assert filtered[1].symbol == "ETHUSDT"


# ============================================================================
# Error Handling Policy Tests (3 tests total)
# ============================================================================


class TestXGBoostFilterErrorHandling:
    """Tests for error handling policies."""

    @patch("modules.auto_trade.core.xgboost_filter.log_error")
    def test_on_error_drop(
        self,
        mock_log_error,
        mock_data_fetcher,
        mock_joblib_load,
    ):
        """Test 'drop' policy - errors cause signal to be dropped."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(
                mock_data_fetcher, "model.joblib", config={"on_error": "drop", "min_required_candles": 100}
            )

            signals = [SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0})]

            # Mock _predict_signal to raise an exception
            with patch.object(filter, "_predict_signal", side_effect=Exception("Prediction error")):
                filtered = filter.filter_signals(signals)

            # Signal should be dropped
            assert len(filtered) == 0
            mock_log_error.assert_called()

    @patch("modules.auto_trade.core.xgboost_filter.log_error")
    def test_on_error_pass(
        self,
        mock_log_error,
        mock_data_fetcher,
        mock_joblib_load,
    ):
        """Test 'pass' policy - errors pass original signal through."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(
                mock_data_fetcher, "model.joblib", config={"on_error": "pass", "min_required_candles": 100}
            )

            signals = [SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0})]

            # Mock _predict_signal to raise an exception
            with patch.object(filter, "_predict_signal", side_effect=Exception("Prediction error")):
                filtered = filter.filter_signals(signals)

            # Original signal should pass through
            assert len(filtered) == 1
            assert filtered[0].symbol == "BTCUSDT"
            assert filtered[0].signal_type == "LONG"
            mock_log_error.assert_called()

    @patch("modules.auto_trade.core.xgboost_filter.log_error")
    def test_on_error_neutral(
        self,
        mock_log_error,
        mock_data_fetcher,
        mock_joblib_load,
    ):
        """Test 'neutral' policy - errors convert signal to NEUTRAL."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(
                mock_data_fetcher, "model.joblib", config={"on_error": "neutral", "min_required_candles": 100}
            )

            signals = [SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0})]

            # Mock _predict_signal to raise an exception
            with patch.object(filter, "_predict_signal", side_effect=Exception("Prediction error")):
                filtered = filter.filter_signals(signals)

            # Should return NEUTRAL signal
            assert len(filtered) == 1
            assert filtered[0].symbol == "BTCUSDT"
            assert filtered[0].signal_type == "NEUTRAL"
            assert filtered[0].score == 0.0
            assert "xgboost_error" in filtered[0].details
            mock_log_error.assert_called()


# ============================================================================
# Prediction Functionality Tests (9 tests total)
# ============================================================================


class TestXGBoostFilterPrediction:
    """Tests for prediction functionality."""

    def test_predict_signal_up(
        self,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test prediction returns UP direction."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib")

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 300})
            mock_predict_next_move.return_value = np.array([0.1, 0.1, 0.8])

            confidence, direction = filter._predict_signal("BTCUSDT")

            assert confidence == 0.8
            assert direction == "UP"

    def test_predict_signal_down(
        self,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test prediction returns DOWN direction."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib")

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 300})
            mock_predict_next_move.return_value = np.array([0.8, 0.1, 0.1])

            confidence, direction = filter._predict_signal("BTCUSDT")

            assert confidence == 0.8
            assert direction == "DOWN"

    def test_predict_signal_neutral(
        self,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test prediction returns NEUTRAL direction."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib")

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 300})
            mock_predict_next_move.return_value = np.array([0.2, 0.6, 0.2])

            confidence, direction = filter._predict_signal("BTCUSDT")

            assert confidence == 0.6
            assert direction == "NEUTRAL"

    @patch("modules.auto_trade.core.xgboost_filter.log_warn")
    def test_predict_no_data(self, mock_log_warn, mock_data_fetcher, mock_joblib_load):
        """Test prediction with no available data."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib")

            # Return None or empty DataFrame
            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame()

            confidence, direction = filter._predict_signal("BTCUSDT")

            assert confidence == 0.0
            assert direction == "NEUTRAL"
            mock_log_warn.assert_called()

    @patch("modules.auto_trade.core.xgboost_filter.log_warn")
    def test_predict_insufficient_data(
        self, mock_log_warn, mock_data_fetcher, mock_joblib_load, mock_indicator_engine, mock_add_advanced_features
    ):
        """Test prediction with insufficient candles."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib", config={"min_required_candles": 250})

            # Only 100 candles (< 250 required)
            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 100})

            confidence, direction = filter._predict_signal("BTCUSDT")

            assert confidence == 0.0
            assert direction == "NEUTRAL"
            mock_log_warn.assert_called()
            assert "Insufficient data" in str(mock_log_warn.call_args)

    @patch("modules.auto_trade.core.xgboost_filter.log_warn")
    def test_predict_invalid_format(
        self,
        mock_log_warn,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test prediction validation with invalid format."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib")

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 300})
            # Return wrong format (4 values instead of 3)
            mock_predict_next_move.return_value = np.array([0.1, 0.2, 0.3, 0.4])

            confidence, direction = filter._predict_signal("BTCUSDT")

            assert confidence == 0.0
            assert direction == "NEUTRAL"

    @patch("modules.auto_trade.core.xgboost_filter.log_warn")
    def test_predict_probabilities_dont_sum_to_one(
        self,
        mock_log_warn,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test prediction validation with invalid probability sum."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib")

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 300})
            # Probabilities don't sum to ~1.0 (sum = 0.5) but UP is highest
            mock_predict_next_move.return_value = np.array([0.1, 0.1, 0.3])

            confidence, direction = filter._predict_signal("BTCUSDT")

            # Should still predict (just warn)
            assert direction == "UP"
            mock_log_warn.assert_called()

    @patch("modules.auto_trade.core.xgboost_filter.log_error")
    def test_predict_feature_computation_error(
        self, mock_log_error, mock_data_fetcher, mock_joblib_load, mock_predict_next_move, mock_add_advanced_features
    ):
        """Test prediction error handling in feature computation."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib")

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 300})
            # Simulate error in feature computation
            mock_predict_next_move.side_effect = Exception("Feature error")

            confidence, direction = filter._predict_signal("BTCUSDT")

            assert confidence == 0.0
            assert direction == "NEUTRAL"
            mock_log_error.assert_called()


# ============================================================================
# Caching Behavior Tests (2 tests total)
# ============================================================================


class TestXGBoostFilterCaching:
    """Tests for prediction caching behavior."""

    def test_prediction_cache_hit(
        self,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test that cached predictions are reused."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib", config={"min_confidence": 0.6})

            signals = [
                SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0}),
                SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0}),  # Duplicate
            ]

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 300})
            mock_predict_next_move.return_value = np.array([0.1, 0.1, 0.8])

            filtered = filter.filter_signals(signals)

            # Should pass both (same prediction reused)
            assert len(filtered) == 2
            # predict_next_move should only be called once due to caching
            assert mock_predict_next_move.call_count == 1

    def test_clear_cache(
        self,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test that cache can be cleared."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib", config={"min_confidence": 0.6})

            signals = [SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0})]

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 300})
            mock_predict_next_move.return_value = np.array([0.1, 0.1, 0.8])

            # First filter - prediction is cached
            filter.filter_signals(signals)
            first_call_count = mock_predict_next_move.call_count

            # Clear cache
            filter.clear_cache()

            # Second filter - prediction is recomputed
            filter.filter_signals(signals)
            second_call_count = mock_predict_next_move.call_count

            # Should have called predict_next_move twice
            assert second_call_count == first_call_count + 1


# ============================================================================
# Edge Cases Tests (4 tests total)
# ============================================================================


class TestXGBoostFilterEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @patch("modules.auto_trade.core.xgboost_filter.log_warn")
    def test_all_signals_rejected(
        self,
        mock_log_warn,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test when all signals are rejected."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib", config={"min_confidence": 0.9})

            signals = [
                SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0}),
                SignalResult("ETHUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0}),
            ]

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 300})
            # All predictions below confidence threshold
            mock_predict_next_move.return_value = np.array([0.1, 0.1, 0.7])

            filtered = filter.filter_signals(signals)

            assert len(filtered) == 0

    @patch("modules.auto_trade.core.xgboost_filter.log_warn")
    def test_all_signals_pass(
        self,
        mock_log_warn,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test when all signals pass."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib", config={"min_confidence": 0.5})

            signals = [
                SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0}),
                SignalResult("ETHUSDT", 1.0, "SHORT", {"1h": "SHORT"}, {"1h": -1.0}),
            ]

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 300})

            # Use call count to return appropriate predictions
            call_count = 0

            def mock_predict(model, df):
                nonlocal call_count
                call_count += 1
                # First call (BTCUSDT LONG): UP
                if call_count == 1:
                    return np.array([0.1, 0.1, 0.8])
                # Second call (ETHUSDT SHORT): DOWN
                else:
                    return np.array([0.8, 0.1, 0.1])

            mock_predict_next_move.side_effect = mock_predict

            filtered = filter.filter_signals(signals)

            assert len(filtered) == 2

    @patch("modules.auto_trade.core.xgboost_filter.log_error")
    def test_multiple_signals_with_errors(
        self,
        mock_log_error,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test multiple signals with some having errors."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(mock_data_fetcher, "model.joblib", config={"on_error": "drop"})

            signals = [
                SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0}),
                SignalResult("ETHUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0}),
                SignalResult("BNBUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0}),
            ]

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 300})

            call_count = 0

            def mock_predict_side_effect(model, df):
                nonlocal call_count
                call_count += 1
                # Second signal fails
                if call_count == 2:
                    raise Exception("ETHUSDT error")
                return np.array([0.1, 0.1, 0.8])

            mock_predict_next_move.side_effect = mock_predict_side_effect

            filtered = filter.filter_signals(signals)

            # Should pass BTC and BNB (drop ETH due to error)
            assert len(filtered) == 2
            symbols_passed = [s.symbol for s in filtered]
            assert "BTCUSDT" in symbols_passed
            assert "BNBUSDT" in symbols_passed
            assert "ETHUSDT" not in symbols_passed

    def test_custom_prediction_timeframe(
        self,
        mock_data_fetcher,
        mock_joblib_load,
        mock_predict_next_move,
        mock_indicator_engine,
        mock_add_advanced_features,
    ):
        """Test with custom prediction timeframe."""
        with patch("pathlib.Path.exists", side_effect=_PATH_EXISTS_JOBLIB_ONLY):
            filter = XGBoostFilter(
                mock_data_fetcher, "model.joblib", config={"prediction_timeframe": "1h", "min_confidence": 0.6}
            )

            signals = [SignalResult("BTCUSDT", 1.0, "LONG", {"1h": "LONG"}, {"1h": 1.0})]

            mock_data_fetcher.fetch_ohlcv.return_value = pd.DataFrame({"close": [100.0] * 300})
            mock_predict_next_move.return_value = np.array([0.1, 0.1, 0.8])

            filtered = filter.filter_signals(signals)

            assert len(filtered) == 1

            # Verify correct timeframe was used
            mock_data_fetcher.fetch_ohlcv.assert_called_once()
            call_kwargs = mock_data_fetcher.fetch_ohlcv.call_args[1]
            assert call_kwargs["timeframe"] == "1h"
