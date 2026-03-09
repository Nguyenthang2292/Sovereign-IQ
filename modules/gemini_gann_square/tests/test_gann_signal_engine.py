"""
Smoke tests for GannSignalEngine.

All external I/O dependencies (OHLCV fetcher, chart generator, Gemini AI)
are mocked so these tests run without network access or API keys.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from modules.gemini_gann_square.core.gann_calculator import (
    GannSquareResult,
    GannZone,
)
from modules.gemini_gann_square.core.gann_signal_engine import (
    GannAnalysisResult,
    GannSignalEngine,
)
from modules.gemini_gann_square.core.swing_detector import SwingPoint

# ──────────────────────────────────────────────
# Helpers / fixtures
# ──────────────────────────────────────────────


def _make_ohlcv_df(n: int = 50) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame large enough for swing detection."""
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(42)
    close = 100.0 + rng.standard_normal(n).cumsum()
    high = close + 1.0
    low = close - 1.0
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1000.0] * n,
        },
        index=timestamps,
    )


_VALID_GEMINI_JSON = """{
  "zone_confirmed": 1,
  "trend_confirmed": "DOWN",
  "override_reason": "",
  "signal": "SHORT",
  "entry_price": 95.0,
  "stop_loss": 102.0,
  "take_profit_1": 88.0,
  "take_profit_2": 80.0,
  "confidence_pct": 75,
  "reasoning": "Price in Zone 1 of a down-trend. Short setup confirmed."
}"""


# ──────────────────────────────────────────────
# Engine instantiation
# ──────────────────────────────────────────────


class TestGannSignalEngineInit:
    def test_default_instantiation(self):
        engine = GannSignalEngine()
        assert engine.swing_detector.lookback == 5

    def test_custom_lookback(self):
        engine = GannSignalEngine(lookback=3)
        assert engine.swing_detector.lookback == 3


# ──────────────────────────────────────────────
# Happy-path analyze() with mocked dependencies
# ──────────────────────────────────────────────


class TestGannSignalEngineAnalyze:
    """Full pipeline smoke tests with all external I/O mocked.

    DataFetcher and ExchangeManager are imported inline inside analyze() to
    avoid circular imports.  We patch them at their source modules so the
    deferred ``from ... import ...`` inside the function body picks up the mock.
    """

    @pytest.fixture(autouse=True)
    def _silence_logging(self):
        """Suppress log_info/log_warn/log_error during analyze() calls.

        SwingPoint.__repr__ contains ▼/▲ Unicode arrows which cause a
        UnicodeEncodeError on Windows cp1252 terminals when printed by log_info.
        """
        with (
            patch("modules.gemini_gann_square.core.gann_signal_engine.log_info"),
            patch("modules.gemini_gann_square.core.gann_signal_engine.log_warn"),
            patch("modules.gemini_gann_square.core.gann_signal_engine.log_error"),
        ):
            yield

    @patch("modules.common.core.exchange_manager.ExchangeManager")
    @patch("modules.common.core.data_fetcher.DataFetcher")
    def test_analyze_returns_gann_analysis_result(self, MockFetcher, MockEM):
        df = _make_ohlcv_df(60)
        MockFetcher.return_value.fetch_ohlcv_with_fallback_exchange.return_value = (df, "binance")

        engine = GannSignalEngine(lookback=3)
        engine.chart_generator = MagicMock()
        engine.chart_generator.create_chart.return_value = "/tmp/chart.png"
        engine.gemini_analyzer = MagicMock()
        engine.gemini_analyzer.analyze_chart.return_value = _VALID_GEMINI_JSON

        result = engine.analyze(symbol="BTC/USDT", timeframe="4h")

        assert isinstance(result, GannAnalysisResult)
        assert result.symbol == "BTC/USDT"
        assert result.timeframe == "4h"

    @patch("modules.common.core.exchange_manager.ExchangeManager")
    @patch("modules.common.core.data_fetcher.DataFetcher")
    def test_analyze_parses_gemini_signal_correctly(self, MockFetcher, MockEM):
        df = _make_ohlcv_df(60)
        MockFetcher.return_value.fetch_ohlcv_with_fallback_exchange.return_value = (df, "binance")

        engine = GannSignalEngine(lookback=3)
        engine.chart_generator = MagicMock()
        engine.chart_generator.create_chart.return_value = "/tmp/chart.png"
        engine.gemini_analyzer = MagicMock()
        engine.gemini_analyzer.analyze_chart.return_value = _VALID_GEMINI_JSON

        result = engine.analyze(symbol="BTC/USDT", timeframe="4h")

        assert result.signal == "SHORT"
        assert result.entry_price == pytest.approx(95.0)
        assert result.stop_loss == pytest.approx(102.0)
        assert result.confidence_pct == 75
        assert result.gemini_parse_error == ""

    @patch("modules.common.core.exchange_manager.ExchangeManager")
    @patch("modules.common.core.data_fetcher.DataFetcher")
    def test_analyze_falls_back_when_gemini_signal_is_invalid(self, MockFetcher, MockEM):
        df = _make_ohlcv_df(60)
        MockFetcher.return_value.fetch_ohlcv_with_fallback_exchange.return_value = (df, "binance")

        invalid_signal_json = """{
            "zone_confirmed": 1,
            "trend_confirmed": "DOWN",
            "override_reason": "",
            "signal": "HOLD",
            "entry_price": 95.0,
            "stop_loss": 102.0,
            "take_profit_1": 88.0,
            "take_profit_2": 80.0,
            "confidence_pct": 75,
            "reasoning": "Invalid signal token should be rejected."
        }"""

        engine = GannSignalEngine(lookback=3)
        engine.chart_generator = MagicMock()
        engine.chart_generator.create_chart.return_value = "/tmp/chart.png"
        engine.gemini_analyzer = MagicMock()
        engine.gemini_analyzer.analyze_chart.return_value = invalid_signal_json

        result = engine.analyze(symbol="BTC/USDT", timeframe="4h")

        assert result.signal == result.gann_result.signal_code

    @patch("modules.common.core.exchange_manager.ExchangeManager")
    @patch("modules.common.core.data_fetcher.DataFetcher")
    def test_analyze_sanitizes_invalid_numeric_values(self, MockFetcher, MockEM):
        df = _make_ohlcv_df(60)
        MockFetcher.return_value.fetch_ohlcv_with_fallback_exchange.return_value = (df, "binance")

        invalid_numeric_json = """{
            "zone_confirmed": 99,
            "trend_confirmed": "DOWN",
            "override_reason": "",
            "signal": "SHORT",
            "entry_price": -1,
            "stop_loss": "NaN",
            "take_profit_1": "bad",
            "take_profit_2": 80.0,
            "confidence_pct": 150,
            "reasoning": "Sanitize invalid values"
        }"""

        engine = GannSignalEngine(lookback=3)
        engine.chart_generator = MagicMock()
        engine.chart_generator.create_chart.return_value = "/tmp/chart.png"
        engine.gemini_analyzer = MagicMock()
        engine.gemini_analyzer.analyze_chart.return_value = invalid_numeric_json

        result = engine.analyze(symbol="BTC/USDT", timeframe="4h")

        assert 1 <= result.zone_confirmed <= 4
        assert result.zone_confirmed == result.gann_result.current_zone
        assert result.entry_price == pytest.approx(0.0)
        assert result.stop_loss == pytest.approx(0.0)
        assert result.take_profit_1 == pytest.approx(0.0)
        assert result.take_profit_2 == pytest.approx(80.0)
        assert result.confidence_pct == 100

    @patch("modules.common.core.exchange_manager.ExchangeManager")
    @patch("modules.common.core.data_fetcher.DataFetcher")
    def test_analyze_zeros_prices_for_skip_signal(self, MockFetcher, MockEM):
        df = _make_ohlcv_df(60)
        MockFetcher.return_value.fetch_ohlcv_with_fallback_exchange.return_value = (df, "binance")

        skip_signal_json = """{
            "zone_confirmed": 3,
            "trend_confirmed": "DOWN",
            "override_reason": "",
            "signal": "SKIP",
            "entry_price": 95.0,
            "stop_loss": 102.0,
            "take_profit_1": 88.0,
            "take_profit_2": 80.0,
            "confidence_pct": 45,
            "reasoning": "Skip signal should not include tradable levels"
        }"""

        engine = GannSignalEngine(lookback=3)
        engine.chart_generator = MagicMock()
        engine.chart_generator.create_chart.return_value = "/tmp/chart.png"
        engine.gemini_analyzer = MagicMock()
        engine.gemini_analyzer.analyze_chart.return_value = skip_signal_json

        result = engine.analyze(symbol="BTC/USDT", timeframe="4h")

        assert result.signal == "SKIP"
        assert result.entry_price == pytest.approx(0.0)
        assert result.stop_loss == pytest.approx(0.0)
        assert result.take_profit_1 == pytest.approx(0.0)
        assert result.take_profit_2 == pytest.approx(0.0)

    @patch("modules.common.core.exchange_manager.ExchangeManager")
    @patch("modules.common.core.data_fetcher.DataFetcher")
    def test_analyze_falls_back_on_malformed_gemini_response(self, MockFetcher, MockEM):
        df = _make_ohlcv_df(60)
        MockFetcher.return_value.fetch_ohlcv_with_fallback_exchange.return_value = (df, "binance")

        engine = GannSignalEngine(lookback=3)
        engine.chart_generator = MagicMock()
        engine.chart_generator.create_chart.return_value = "/tmp/chart.png"
        engine.gemini_analyzer = MagicMock()
        engine.gemini_analyzer.analyze_chart.return_value = "No JSON here."

        result = engine.analyze(symbol="BTC/USDT", timeframe="4h")

        # Fallback values
        assert result.signal == result.gann_result.signal_code
        assert result.entry_price == pytest.approx(0.0)
        assert result.gemini_parse_error != ""

    @patch("modules.common.core.exchange_manager.ExchangeManager")
    @patch("modules.common.core.data_fetcher.DataFetcher")
    def test_analyze_raises_when_no_ohlcv_data(self, MockFetcher, MockEM):
        MockFetcher.return_value.fetch_ohlcv_with_fallback_exchange.return_value = (None, "binance")

        engine = GannSignalEngine(lookback=3)
        with pytest.raises(ValueError, match="No OHLCV data"):
            engine.analyze(symbol="BTC/USDT", timeframe="4h")

    @patch("modules.common.core.exchange_manager.ExchangeManager")
    @patch("modules.common.core.data_fetcher.DataFetcher")
    def test_analyze_raises_when_df_empty(self, MockFetcher, MockEM):
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        MockFetcher.return_value.fetch_ohlcv_with_fallback_exchange.return_value = (empty_df, "binance")

        engine = GannSignalEngine(lookback=3)
        with pytest.raises(ValueError, match="No OHLCV data"):
            engine.analyze(symbol="BTC/USDT", timeframe="4h")

    @patch("modules.common.core.exchange_manager.ExchangeManager")
    @patch("modules.common.core.data_fetcher.DataFetcher")
    def test_chart_generator_called_once(self, MockFetcher, MockEM):
        df = _make_ohlcv_df(60)
        MockFetcher.return_value.fetch_ohlcv_with_fallback_exchange.return_value = (df, "binance")

        engine = GannSignalEngine(lookback=3)
        engine.chart_generator = MagicMock()
        engine.chart_generator.create_chart.return_value = "/tmp/chart.png"
        engine.gemini_analyzer = MagicMock()
        engine.gemini_analyzer.analyze_chart.return_value = _VALID_GEMINI_JSON

        engine.analyze(symbol="ETH/USDT", timeframe="1h")

        engine.chart_generator.create_chart.assert_called_once()

    @patch("modules.common.core.exchange_manager.ExchangeManager")
    @patch("modules.common.core.data_fetcher.DataFetcher")
    def test_gemini_called_with_chart_path(self, MockFetcher, MockEM):
        df = _make_ohlcv_df(60)
        MockFetcher.return_value.fetch_ohlcv_with_fallback_exchange.return_value = (df, "binance")

        engine = GannSignalEngine(lookback=3)
        engine.chart_generator = MagicMock()
        engine.chart_generator.create_chart.return_value = "/tmp/specific_chart.png"
        engine.gemini_analyzer = MagicMock()
        engine.gemini_analyzer.analyze_chart.return_value = _VALID_GEMINI_JSON

        engine.analyze(symbol="BTC/USDT", timeframe="4h")

        call_kwargs = engine.gemini_analyzer.analyze_chart.call_args
        assert call_kwargs.kwargs["image_path"] == "/tmp/specific_chart.png"


# ──────────────────────────────────────────────
# is_tradeable and display smoke tests
# ──────────────────────────────────────────────


class TestGannAnalysisResultHelpers:
    def _make_result(self, signal: str) -> GannAnalysisResult:
        """Build a minimal GannAnalysisResult for display/helper tests."""
        zone = GannZone(
            zone_number=1,
            pivot_index=5,
            pivot_price=100.0,
            slope=-0.5,
            _upper_slope=-0.5,
            _lower_slope=-1.0,
            label="Zone 1 (SHORT)",
            is_tradeable=True,
            signal="SHORT",  # type: ignore[arg-type]
        )
        sp_high = SwingPoint(index=5, timestamp=pd.Timestamp("2024-01-05"), price=100.0, kind="high")  # type: ignore[arg-type]
        sp_low = SwingPoint(index=50, timestamp=pd.Timestamp("2024-03-01"), price=60.0, kind="low")  # type: ignore[arg-type]
        gann_r = GannSquareResult(
            trend="DOWN",
            swing_high=sp_high,
            swing_low=sp_low,
            price_range=40.0,
            zones=[zone, zone, zone, zone],
            current_zone=1,
            signal_code="SHORT",  # type: ignore[arg-type]
            current_index=25,
        )
        return GannAnalysisResult(
            symbol="BTC/USDT",
            timeframe="4h",
            gann_result=gann_r,
            chart_path="/tmp/chart.png",
            zone_confirmed=1,
            trend_confirmed="DOWN",
            override_reason="",
            signal=signal,  # type: ignore[arg-type]
            entry_price=95.0,
            stop_loss=102.0,
            take_profit_1=88.0,
            take_profit_2=80.0,
            confidence_pct=70,
            reasoning="Test reasoning.",
        )

    def test_is_tradeable_long(self):
        assert self._make_result("LONG").is_tradeable() is True

    def test_is_tradeable_short(self):
        assert self._make_result("SHORT").is_tradeable() is True

    def test_is_tradeable_skip(self):
        assert self._make_result("SKIP").is_tradeable() is False

    def test_display_contains_symbol(self):
        output = self._make_result("SHORT").display()
        assert "BTC/USDT" in output

    def test_display_contains_signal(self):
        output = self._make_result("SHORT").display()
        assert "SHORT" in output

    def test_display_contains_entry_for_tradeable(self):
        output = self._make_result("SHORT").display()
        assert "Entry" in output

    def test_display_no_entry_for_skip(self):
        output = self._make_result("SKIP").display()
        assert "Entry" not in output


class TestGannSignalEnginePromptBuilder:
    def test_build_prompt_replaces_placeholders(self, tmp_path):
        prompt_file = tmp_path / "gann_analysis.txt"
        prompt_file.write_text(
            "SYMBOL={SYMBOL}; TF={TIMEFRAME}; Z1={ZONE1_SIGNAL}; CUR={CURRENT_ZONE}",
            encoding="utf-8",
        )

        engine = GannSignalEngine(lookback=3)
        engine._PROMPTS_DIR = tmp_path

        zone = GannZone(
            zone_number=1,
            pivot_index=5,
            pivot_price=100.0,
            slope=-0.5,
            _upper_slope=-0.5,
            _lower_slope=-1.0,
            label="Zone 1 (SHORT)",
            is_tradeable=True,
            signal="SHORT",  # type: ignore[arg-type]
        )
        sp_high = SwingPoint(index=5, timestamp=pd.Timestamp("2024-01-05"), price=100.0, kind="high")  # type: ignore[arg-type]
        sp_low = SwingPoint(index=50, timestamp=pd.Timestamp("2024-03-01"), price=60.0, kind="low")  # type: ignore[arg-type]
        gann_r = GannSquareResult(
            trend="DOWN",
            swing_high=sp_high,
            swing_low=sp_low,
            price_range=40.0,
            zones=[zone, zone, zone, zone],
            current_zone=1,
            signal_code="SHORT",  # type: ignore[arg-type]
        )

        prompt = engine._build_prompt(
            symbol="BTC/USDT",
            timeframe="4h",
            current_price=95.0,
            gann=gann_r,
        )

        assert "{SYMBOL}" not in prompt
        assert "{TIMEFRAME}" not in prompt
        assert "{ZONE1_SIGNAL}" not in prompt
        assert "{CURRENT_ZONE}" not in prompt
        assert "BTC/USDT" in prompt
        assert "4h" in prompt
