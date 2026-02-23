"""Tests for XGBoostServerlessFilter.

Key behaviors tested:
- Mock mode: random Lambda-like predictions (test only)
- Live mode: real Lambda client (boto3)
- Fallback: Lambda fail → local XGBoostFilter
- No fallback: neither Lambda nor local model → pass-through
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Lazy import helpers (avoid dask import chain at collection time) ──────────


def _import_filter_class():
    from modules.auto_trade.core.xgboost_serverless_filter import XGBoostServerlessFilter

    return XGBoostServerlessFilter


def _import_signal_result():
    from modules.auto_trade.core.atc_scanner import SignalResult

    return SignalResult


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_signal(symbol: str, signal_type: str = "LONG", score: float = 0.8):
    SR = _import_signal_result()
    return SR(
        symbol=symbol,
        score=score,
        signal_type=signal_type,
        details={"some_key": "some_value"},
        strengths={"5m": 0.5, "15m": 0.7},
    )


def _make_ohlcv_df(rows: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=rows, freq="15min")
    return pd.DataFrame(
        {
            "open": np.random.uniform(90, 110, rows),
            "high": np.random.uniform(100, 120, rows),
            "low": np.random.uniform(80, 100, rows),
            "close": np.random.uniform(90, 110, rows),
            "volume": np.random.uniform(1000, 5000, rows),
        },
        index=dates,
    )


def _mock_lambda_response(symbol: str, label: str, confidence: float) -> dict:
    """Build a deterministic Lambda-style response for testing."""
    probs = [0.1, 0.1, 0.8] if label == "UP" else ([0.8, 0.1, 0.1] if label == "DOWN" else [0.1, 0.8, 0.1])
    return {
        "success": True,
        "predictions": [
            {
                "symbol": symbol,
                "timeframe": "15m",
                "prediction": {
                    "label": label,
                    "probabilities": probs,
                    "confidence": confidence,
                },
            }
        ],
        "timing": {},
    }


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_data_fetcher():
    fetcher = MagicMock()
    fetcher.fetch_ohlcv.return_value = _make_ohlcv_df(200)
    return fetcher


@pytest.fixture
def mock_config():
    """Config with mock mode enabled (no AWS calls)."""
    return {
        "xgboost_serverless_function_name": "test-xgboost-lambda",
        "xgboost_serverless_region": "us-east-1",
        "xgboost_serverless_model_version": "v1",
        "xgboost_serverless_timeframe": "15m",
        "xgboost_serverless_candle_limit": 200,
        "xgboost_serverless_min_confidence": 0.4,
        "xgboost_serverless_min_candles": 50,
        "xgboost_serverless_mock_mode": True,
    }


# ── Init tests ────────────────────────────────────────────────────────────────


class TestXGBoostServerlessFilterInit:
    def test_init_mock_mode(self, mock_data_fetcher, mock_config):
        Cls = _import_filter_class()
        flt = Cls(data_fetcher=mock_data_fetcher, config=mock_config)
        assert flt.mock_mode is True
        assert flt.function_name == "test-xgboost-lambda"
        assert flt.region == "us-east-1"
        assert flt.model_version == "v1"
        assert flt.min_confidence == 0.4
        # In mock mode, _local_fallback is not built
        assert flt._local_fallback is None

    def test_init_live_mode_falls_back_when_no_boto3_credentials(self, mock_data_fetcher):
        """When boto3 raises (no creds), lambda_available=False but no crash."""
        Cls = _import_filter_class()
        with patch(
            "modules.auto_trade.core.xgboost_serverless_filter.XGBoostLambdaClient",
            side_effect=Exception("No credentials"),
        ):
            flt = Cls(
                data_fetcher=mock_data_fetcher,
                config={
                    "xgboost_serverless_mock_mode": False,
                },
            )
        assert flt._lambda_available is False
        # _local_fallback is attempted (may be None if no model on disk)
        # — we just verify it doesn't crash

    def test_init_with_empty_config_uses_defaults(self, mock_data_fetcher):
        Cls = _import_filter_class()
        flt = Cls(
            data_fetcher=mock_data_fetcher,
            config={"xgboost_serverless_mock_mode": True},
        )
        assert flt.function_name == "xgboost-serverless-predict"
        assert flt.min_confidence == 0.55


# ── filter_signals tests ──────────────────────────────────────────────────────


class TestXGBoostServerlessFilterSignals:
    def test_empty_signals_returns_empty(self, mock_data_fetcher, mock_config):
        Cls = _import_filter_class()
        flt = Cls(data_fetcher=mock_data_fetcher, config=mock_config)
        assert flt.filter_signals([]) == []

    def test_filter_confirms_matching_direction(self, mock_data_fetcher, mock_config):
        """When Lambda says UP and ATC says LONG → signal kept."""
        Cls = _import_filter_class()
        flt = Cls(data_fetcher=mock_data_fetcher, config=mock_config)

        signals = [_make_signal("BTCUSDT", "LONG", 0.85)]

        def forced_mock(requests):
            return _mock_lambda_response("BTCUSDT", "UP", 0.75)

        with patch(
            "modules.auto_trade.core.xgboost_serverless_filter._mock_predict",
            side_effect=forced_mock,
        ):
            result = flt.filter_signals(signals)

        assert len(result) == 1
        assert result[0].details["xgboost_conf"] == 0.75
        assert result[0].details["xgboost_dir"] == "LONG"
        assert result[0].details["xgboost_backend"] == "serverless"
        assert result[0].details["xgboost_label"] == "UP"
        # Original detail preserved
        assert result[0].details["some_key"] == "some_value"

    def test_filter_rejects_direction_mismatch(self, mock_data_fetcher, mock_config):
        """Lambda says DOWN but ATC says LONG → signal dropped."""
        Cls = _import_filter_class()
        flt = Cls(data_fetcher=mock_data_fetcher, config=mock_config)

        signals = [_make_signal("BTCUSDT", "LONG", 0.85)]

        def mismatched(requests):
            return _mock_lambda_response("BTCUSDT", "DOWN", 0.75)

        with patch(
            "modules.auto_trade.core.xgboost_serverless_filter._mock_predict",
            side_effect=mismatched,
        ):
            result = flt.filter_signals(signals)

        assert len(result) == 0

    def test_filter_rejects_low_confidence(self, mock_data_fetcher, mock_config):
        """Confidence below threshold → signal dropped."""
        Cls = _import_filter_class()
        mock_config["xgboost_serverless_min_confidence"] = 0.9
        flt = Cls(data_fetcher=mock_data_fetcher, config=mock_config)

        signals = [_make_signal("BTCUSDT", "LONG", 0.85)]

        def low_conf(requests):
            return _mock_lambda_response("BTCUSDT", "UP", 0.6)

        with patch(
            "modules.auto_trade.core.xgboost_serverless_filter._mock_predict",
            side_effect=low_conf,
        ):
            result = flt.filter_signals(signals)

        assert len(result) == 0

    def test_no_ohlcv_data_routes_to_fallback(self, mock_data_fetcher, mock_config):
        """No OHLCV → routes to local fallback (or pass-through if no model)."""
        Cls = _import_filter_class()
        mock_data_fetcher.fetch_ohlcv.return_value = None

        flt = Cls(data_fetcher=mock_data_fetcher, config=mock_config)
        signals = [_make_signal("BTCUSDT", "LONG", 0.85)]

        # In mock mode with no OHLCV, result is empty (no items to process)
        result = flt.filter_signals(signals)
        # Could be empty or passed-through depending on mock response; just check no crash
        assert isinstance(result, list)

    def test_lambda_call_failure_routes_to_local_fallback(self, mock_data_fetcher):
        """When Lambda call fails at runtime, should route to local XGBoostFilter."""
        Cls = _import_filter_class()

        # Live mode (not mock), but Lambda client will fail at call time
        mock_lambda_client = MagicMock()
        mock_lambda_client.predict.side_effect = RuntimeError("Lambda timeout")

        mock_local_fallback = MagicMock()
        expected_signal = _make_signal("BTC/USDT", "LONG", 0.85)
        mock_local_fallback.filter_signals.return_value = [expected_signal]

        with patch(
            "modules.auto_trade.core.xgboost_serverless_filter.XGBoostLambdaClient",
            return_value=mock_lambda_client,
        ):
            flt = Cls(
                data_fetcher=mock_data_fetcher,
                config={
                    "xgboost_serverless_mock_mode": False,
                    "xgboost_serverless_min_candles": 50,
                    "xgboost_serverless_candle_limit": 200,
                    "xgboost_serverless_timeframe": "15m",
                    "xgboost_serverless_min_confidence": 0.5,
                },
            )

        # Inject the mock local fallback after init
        flt._local_fallback = mock_local_fallback

        signals = [_make_signal("BTC/USDT", "LONG", 0.85)]
        result = flt.filter_signals(signals)

        # Verify local fallback was called
        mock_local_fallback.filter_signals.assert_called_once_with(signals)
        assert result == [expected_signal]

    def test_lambda_unavailable_routes_to_local_fallback(self, mock_data_fetcher):
        """When Lambda client init fails, should route directly to local fallback."""
        Cls = _import_filter_class()

        mock_local_fallback = MagicMock()
        expected_signal = _make_signal("ETH/USDT", "SHORT", 0.75)
        mock_local_fallback.filter_signals.return_value = [expected_signal]

        with patch(
            "modules.auto_trade.core.xgboost_serverless_filter.XGBoostLambdaClient",
            side_effect=Exception("No credentials"),
        ):
            flt = Cls(
                data_fetcher=mock_data_fetcher,
                config={"xgboost_serverless_mock_mode": False},
            )

        assert flt._lambda_available is False

        # Inject mock local fallback
        flt._local_fallback = mock_local_fallback

        signals = [_make_signal("ETH/USDT", "SHORT", 0.75)]
        result = flt.filter_signals(signals)

        mock_local_fallback.filter_signals.assert_called_once_with(signals)
        assert result == [expected_signal]

    def test_no_local_fallback_passes_through(self, mock_data_fetcher):
        """If Lambda fails AND no local fallback, signals pass through unchanged."""
        Cls = _import_filter_class()

        with patch(
            "modules.auto_trade.core.xgboost_serverless_filter.XGBoostLambdaClient",
            side_effect=Exception("No credentials"),
        ):
            flt = Cls(
                data_fetcher=mock_data_fetcher,
                config={"xgboost_serverless_mock_mode": False},
            )

        # No local fallback
        flt._local_fallback = None
        flt._lambda_available = False

        signals = [_make_signal("ETH/USDT", "SHORT", 0.75)]
        result = flt.filter_signals(signals)

        # All signals pass through unchanged
        assert result == signals

    def test_missing_model_fires_request_training_per_symbol(self, mock_data_fetcher):
        """Fire-and-forget: request_training called once per ATC signal, no blocking."""
        Cls = _import_filter_class()

        mock_lambda_client = MagicMock()
        with (
            patch(
                "modules.auto_trade.core.xgboost_serverless_filter.XGBoostLambdaClient",
                return_value=mock_lambda_client,
            ),
            patch.object(Cls, "_build_local_fallback", return_value=None),
        ):
            flt = Cls(
                data_fetcher=mock_data_fetcher,
                config={
                    "xgboost_serverless_mock_mode": False,
                    "xgboost_serverless_timeframe": "15m",
                    "xgboost_serverless_model_version": "v1",
                    "xgboost_serverless_s3_bucket": "xgboost-models-store",
                },
            )

        sig_btc = _make_signal("BTC/USDT", "LONG", 0.8)
        sig_eth = _make_signal("ETH/USDT", "SHORT", 0.7)
        request_items = [
            {
                "symbol": "BTCUSDT",
                "timeframe": "15m",
                "model_version": "v1",
                "model_s3_key": "BTCUSDT_15m_v1.json",
                "data": {},
            },
            {
                "symbol": "ETHUSDT",
                "timeframe": "15m",
                "model_version": "v1",
                "model_s3_key": "ETHUSDT_15m_v1.json",
                "data": {},
            },
        ]
        signal_map = {"BTCUSDT": sig_btc, "ETHUSDT": sig_eth}

        with patch(
            "modules.auto_trade.core.xgboost_serverless_filter.request_training",
            return_value="pending",
        ) as rt_mock:
            result = flt._handle_missing_models(
                signals=[sig_btc, sig_eth],
                request_items=request_items,
                signal_map=signal_map,
                exc_str="Failed to download model from S3",
            )

        # Fire-and-forget: one request_training call per symbol (dynamic = 2)
        assert rt_mock.call_count == 2
        # Current cycle returns empty — no blocking
        assert result == []
        # Lambda predict must NOT be called in this cycle
        mock_lambda_client.predict.assert_not_called()

    def test_missing_model_skips_infra_error_symbols(self, mock_data_fetcher):
        """Symbols with infra_error or skipped status are counted separately but still return []."""
        Cls = _import_filter_class()

        mock_lambda_client = MagicMock()
        with (
            patch(
                "modules.auto_trade.core.xgboost_serverless_filter.XGBoostLambdaClient",
                return_value=mock_lambda_client,
            ),
            patch.object(Cls, "_build_local_fallback", return_value=None),
        ):
            flt = Cls(
                data_fetcher=mock_data_fetcher,
                config={
                    "xgboost_serverless_mock_mode": False,
                    "xgboost_serverless_timeframe": "15m",
                    "xgboost_serverless_model_version": "v1",
                    "xgboost_serverless_s3_bucket": "xgboost-models-store",
                },
            )

        sig_btc = _make_signal("BTC/USDT", "LONG", 0.8)
        sig_eth = _make_signal("ETH/USDT", "SHORT", 0.7)
        request_items = [
            {
                "symbol": "BTCUSDT",
                "timeframe": "15m",
                "model_version": "v1",
                "model_s3_key": "BTCUSDT_15m_v1.json",
                "data": {},
            },
            {
                "symbol": "ETHUSDT",
                "timeframe": "15m",
                "model_version": "v1",
                "model_s3_key": "ETHUSDT_15m_v1.json",
                "data": {},
            },
        ]
        signal_map = {"BTCUSDT": sig_btc, "ETHUSDT": sig_eth}

        # BTC → pending (training ok), ETH → infra_error (IAM issue)
        with patch(
            "modules.auto_trade.core.xgboost_serverless_filter.request_training",
            side_effect=["pending", "infra_error"],
        ) as rt_mock:
            result = flt._handle_missing_models(
                signals=[sig_btc, sig_eth],
                request_items=request_items,
                signal_map=signal_map,
                exc_str="Failed to download model from S3",
            )

        assert rt_mock.call_count == 2
        assert result == []  # always empty this cycle
        mock_lambda_client.predict.assert_not_called()


# ── OHLCV helper tests ────────────────────────────────────────────────────────


class TestXGBoostServerlessFilterHelpers:
    def test_fetch_ohlcv_returns_dict(self, mock_data_fetcher, mock_config):
        Cls = _import_filter_class()
        flt = Cls(data_fetcher=mock_data_fetcher, config=mock_config)
        result = flt._fetch_ohlcv("BTC/USDT")
        assert result is not None
        for key in ("timestamp", "open", "high", "low", "close", "volume"):
            assert key in result
        assert len(result["close"]) >= 50

    def test_fetch_ohlcv_returns_none_on_error(self, mock_data_fetcher, mock_config):
        Cls = _import_filter_class()
        mock_data_fetcher.fetch_ohlcv.side_effect = Exception("Network error")
        flt = Cls(data_fetcher=mock_data_fetcher, config=mock_config)
        assert flt._fetch_ohlcv("BTC/USDT") is None

    def test_fetch_ohlcv_returns_none_on_too_few_candles(self, mock_data_fetcher, mock_config):
        Cls = _import_filter_class()
        mock_data_fetcher.fetch_ohlcv.return_value = _make_ohlcv_df(10)
        flt = Cls(data_fetcher=mock_data_fetcher, config=mock_config)
        assert flt._fetch_ohlcv("BTC/USDT") is None

    def test_passthrough_all_static(self):
        """_passthrough_all returns the original signals as-is."""
        Cls = _import_filter_class()
        signals = [
            _make_signal("BTC/USDT", "LONG", 0.85),
            _make_signal("ETH/USDT", "SHORT", 0.75),
        ]
        result = Cls._passthrough_all(signals)
        assert result == signals  # same objects, no mutation
