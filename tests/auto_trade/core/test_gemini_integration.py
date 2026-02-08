import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pandas as pd
import pytest

from modules.auto_trade.core.atc_scanner import SignalResult
from modules.auto_trade.core.gemini_integration import GeminiIntegration, GeminiSignal


@pytest.fixture
def mock_data_fetcher():
    df = MagicMock()
    df.fetch_ohlcv_with_fallback_exchange.return_value = (pd.DataFrame({"close": [100.0] * 200}), "binance")
    return df


@pytest.fixture
def mock_chart_generator():
    with mock_patch("modules.auto_trade.core.gemini_integration.ChartGenerator") as mock:
        generator = MagicMock()
        mock.return_value = generator
        generator.create_chart.return_value = "temp_chart.png"
        yield generator


@pytest.fixture
def mock_analyzer():
    with mock_patch("modules.auto_trade.core.gemini_integration.GeminiChartAnalyzer") as mock:
        analyzer = MagicMock()
        mock.return_value = analyzer
        yield analyzer


@pytest.fixture
def sample_gemini_json():
    return json.dumps(
        {
            "trend": "Uptrend",
            "patterns": ["Bullish Flag", "Support Bounce"],
            "support_resistance": {"support": [90.0], "resistance": [110.0]},
            "signal": "LONG",
            "confidence": 0.85,
            "entry": 102.5,
            "stop_loss": 98.0,
            "take_profit": 115.0,
        }
    )


def test_analyze_candidate_success(mock_data_fetcher, mock_chart_generator, mock_analyzer, sample_gemini_json):
    """Test successful full flow analysis."""
    mock_analyzer.analyze_chart.return_value = f"```json\n{sample_gemini_json}\n```"

    integration = GeminiIntegration(mock_data_fetcher)
    signal = SignalResult("BTCUSDT", 1.0, "LONG", {}, strengths={})

    with mock_patch("os.remove"), mock_patch("os.path.exists", return_value=True):
        result = integration.analyze_candidate(signal)

    assert result is not None
    assert isinstance(result, GeminiSignal)
    assert result.signal == "LONG"
    assert result.confidence == 0.85
    assert result.trend == "Uptrend"
    assert result.entry == 102.5
    assert "Bullish Flag" in result.reasoning

    # Interaction checks
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.assert_called_with(
        symbol="BTCUSDT", timeframe="1h", limit=200, check_freshness=False
    )
    mock_chart_generator.create_chart.assert_called_once()
    mock_analyzer.analyze_chart.assert_called_once()


def test_analyze_candidate_parsing_error(mock_data_fetcher, mock_chart_generator, mock_analyzer):
    """Test handling of invalid JSON from Gemini."""
    mock_analyzer.analyze_chart.return_value = "I think it is a LONG but I won't give JSON."

    integration = GeminiIntegration(mock_data_fetcher)
    signal = SignalResult("BTCUSDT", 1.0, "LONG", {}, strengths={})

    with mock_patch("os.remove"), mock_patch("os.path.exists", return_value=True):
        result = integration.analyze_candidate(signal)

    assert result is None


def test_no_data_abort(mock_data_fetcher, mock_chart_generator, mock_analyzer):
    """Test abort if data fetching fails."""
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (None, None)

    integration = GeminiIntegration(mock_data_fetcher)
    signal = SignalResult("BTCUSDT", 1.0, "LONG", {}, strengths={})

    result = integration.analyze_candidate(signal)

    assert result is None
    mock_chart_generator.create_chart.assert_not_called()


def test_chart_generation_failure(mock_data_fetcher, mock_chart_generator, mock_analyzer):
    """Test handling of chart generation exception."""
    mock_chart_generator.create_chart.side_effect = Exception("Matplotlib error")

    integration = GeminiIntegration(mock_data_fetcher)
    signal = SignalResult("BTCUSDT", 1.0, "LONG", {}, strengths={})

    result = integration.analyze_candidate(signal)

    assert result is None


def test_cleanup_called(mock_data_fetcher, mock_chart_generator, mock_analyzer, sample_gemini_json):
    """Test that temp file cleanup is attempted."""
    mock_analyzer.analyze_chart.return_value = f"```json\n{sample_gemini_json}\n```"

    integration = GeminiIntegration(mock_data_fetcher)
    signal = SignalResult("BTCUSDT", 1.0, "LONG", {}, strengths={})

    with mock_patch("os.remove") as mock_remove, mock_patch("os.path.exists", return_value=True):
        integration = GeminiIntegration(mock_data_fetcher)
        signal = SignalResult("BTCUSDT", 1.0, "LONG", {}, strengths={})
        mock_analyzer.analyze_chart.return_value = sample_gemini_json
        integration.analyze_candidate(signal)
        mock_remove.assert_called_once()


def test_analyze_candidate_retry_logic(mock_data_fetcher, mock_chart_generator, mock_analyzer, sample_gemini_json):
    """Test that analyze_candidate retries on failure."""
    # Fail twice, then succeed
    mock_analyzer.analyze_chart.side_effect = [
        Exception("Fail 1"),
        Exception("Fail 2"),
        f"```json\n{sample_gemini_json}\n```",
    ]

    integration = GeminiIntegration(mock_data_fetcher)
    signal = SignalResult("BTCUSDT", 1.0, "LONG", {}, strengths={})

    with (
        mock_patch("os.remove"),
        mock_patch("os.path.exists", return_value=True),
        mock_patch("time.sleep") as mock_sleep,
    ):
        result = integration.analyze_candidate(signal)

    assert result is not None
    assert mock_analyzer.analyze_chart.call_count == 3
    # Should initiate sleep twice (1s and 2s)
    assert mock_sleep.call_count == 2


class TestGeminiIntegrationConfiguration:
    """Tests for configuration and initialization."""

    def test_init_with_defaults(self, mock_data_fetcher):
        """Test initialization with default parameters."""
        integration = GeminiIntegration(mock_data_fetcher)

        assert integration.analysis_timeframe == "1h"
        assert integration.history_limit == 200
        assert integration.indicators == GeminiIntegration.DEFAULT_INDICATORS
        assert integration.cache_ttl.total_seconds() == 3600  # 1 hour

    def test_init_with_custom_timeframe(self, mock_data_fetcher):
        """Test initialization with custom timeframe."""
        integration = GeminiIntegration(mock_data_fetcher, analysis_timeframe="15m")

        assert integration.analysis_timeframe == "15m"

    def test_init_with_invalid_timeframe_raises_error(self, mock_data_fetcher):
        """Test invalid timeframe raises ValueError."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            GeminiIntegration(mock_data_fetcher, analysis_timeframe="3m")

    def test_init_with_invalid_history_limit_raises_error(self, mock_data_fetcher):
        """Test invalid history limit raises ValueError."""
        with pytest.raises(ValueError, match="history_limit must be positive"):
            GeminiIntegration(mock_data_fetcher, history_limit=-10)

    def test_init_with_custom_indicators(self, mock_data_fetcher):
        """Test initialization with custom indicators."""
        custom_indicators = {"MA": {"periods": [10, 20]}, "RSI": {"period": 7}}
        integration = GeminiIntegration(mock_data_fetcher, indicators=custom_indicators)

        assert integration.indicators == custom_indicators

    def test_is_available_with_api_key(self, mock_data_fetcher):
        """Test is_available returns True when API key is set."""
        with mock_patch("os.getenv", return_value="test_api_key"):
            integration = GeminiIntegration(mock_data_fetcher)

            assert integration.is_available() is True

    def test_is_available_without_api_key(self, mock_data_fetcher):
        """Test is_available returns False when no API key."""
        with mock_patch("os.getenv", return_value=None):
            integration = GeminiIntegration(mock_data_fetcher)

            assert integration.is_available() is False

    def test_temp_directory_created(self, mock_data_fetcher):
        """Test temp directory is created during initialization."""
        with tempfile.TemporaryDirectory() as base_temp:
            with mock_patch("tempfile.gettempdir", return_value=base_temp):
                integration = GeminiIntegration(mock_data_fetcher)

                # Check temp_dir is set correctly
                expected_path = Path(base_temp) / "gemini_charts"
                assert integration.temp_dir == expected_path

                # Check directory was created
                assert expected_path.exists()

    def test_init_with_custom_cache_ttl(self, mock_data_fetcher):
        """Test initialization with custom cache TTL."""
        integration = GeminiIntegration(mock_data_fetcher, cache_ttl_seconds=1800)
        assert integration.cache_ttl.total_seconds() == 1800


class TestGeminiIntegrationCaching:
    """Tests for caching behavior."""

    def test_cache_hit(self, mock_data_fetcher, mock_chart_generator, mock_analyzer, sample_gemini_json):
        """Test that cached results are reused."""
        mock_analyzer.analyze_chart.return_value = f"```json\n{sample_gemini_json}\n```"

        with mock_patch("os.remove"), mock_patch("os.path.exists", return_value=True):
            integration = GeminiIntegration(mock_data_fetcher)
            signal = SignalResult("BTCUSDT", 1.0, "LONG", {}, strengths={})

            # First call - cache miss
            result1 = integration.analyze_candidate(signal)
            assert result1 is not None

            # Second call - cache hit (same symbol, within TTL)
            result2 = integration.analyze_candidate(signal)
            assert result2 is not None

            # Analyzer should only be called once due to caching
            assert mock_analyzer.analyze_chart.call_count == 1

    def test_cache_expiration(self, mock_data_fetcher, mock_chart_generator, mock_analyzer, sample_gemini_json):
        """Test that cache expires after TTL."""
        from datetime import datetime

        t1 = datetime(2023, 1, 1, 10, 0, 0)
        t2 = datetime(2023, 1, 1, 12, 0, 0)
        t3 = datetime(2023, 1, 1, 12, 0, 1)

        mock_analyzer.analyze_chart.return_value = f"```json\n{sample_gemini_json}\n```"

        with (
            mock_patch("modules.auto_trade.core.gemini_integration.datetime") as mock_datetime,
            mock_patch("os.remove"),
            mock_patch("os.path.exists", return_value=True),
        ):
            mock_datetime.now.side_effect = [t1, t2, t3]

            integration = GeminiIntegration(mock_data_fetcher)
            signal = SignalResult("BTCUSDT", 1.0, "LONG", {}, strengths={})

            # First call - cache miss
            result1 = integration.analyze_candidate(signal)
            assert result1 is not None

            # Second call - cache expired (t2 - t1 = 2h > 1h)
            result2 = integration.analyze_candidate(signal)
            assert result2 is not None

            assert mock_analyzer.analyze_chart.call_count == 2

    def test_clear_cache(self, mock_data_fetcher, mock_chart_generator, mock_analyzer, sample_gemini_json):
        """Test clear_cache method."""
        mock_analyzer.analyze_chart.return_value = f"```json\n{sample_gemini_json}\n```"

        with mock_patch("os.remove"), mock_patch("os.path.exists", return_value=True):
            integration = GeminiIntegration(mock_data_fetcher)
            signal = SignalResult("BTCUSDT", 1.0, "LONG", {}, strengths={})

            # First call - populates cache
            result1 = integration.analyze_candidate(signal)
            assert result1 is not None

            # Clear cache
            integration.clear_cache()
            assert len(integration._cache) == 0

            # Second call - cache miss, new API call
            result2 = integration.analyze_candidate(signal)
            assert result2 is not None

            # Should be 2 API calls total
            assert mock_analyzer.analyze_chart.call_count == 2


def test_analyze_candidate_async_success(mock_data_fetcher, mock_chart_generator, mock_analyzer, sample_gemini_json):
    """Test successful async analysis."""
    import asyncio  # Added import for asyncio.run

    mock_analyzer.analyze_chart.return_value = f"```json\n{sample_gemini_json}\n```"

    integration = GeminiIntegration(mock_data_fetcher)
    signal = SignalResult("BTCUSDT", 1.0, "LONG", {}, strengths={})

    with mock_patch("os.remove"), mock_patch("os.path.exists", return_value=True):
        result = asyncio.run(integration.analyze_candidate_async(signal))

    assert result is not None
    assert result.signal == "LONG"
    mock_chart_generator.create_chart.assert_called_once()


def test_analyze_candidates_batch_async(mock_data_fetcher, mock_chart_generator, mock_analyzer, sample_gemini_json):
    """Test batch async analysis."""
    import asyncio  # Added import for asyncio.run

    mock_analyzer.analyze_chart.return_value = f"```json\n{sample_gemini_json}\n```"

    integration = GeminiIntegration(mock_data_fetcher)
    signals = [SignalResult("BTCUSDT", 1.0, "LONG", {}, strengths={}), SignalResult("ETHUSDT", 0.9, "SHORT", {}, strengths={})]

    with mock_patch("os.remove"), mock_patch("os.path.exists", return_value=True):
        results = asyncio.run(integration.analyze_candidates_batch_async(signals))

    assert len(results) == 2
    assert "BTCUSDT" in results
    assert "ETHUSDT" in results
    assert results["BTCUSDT"] is not None
    assert mock_chart_generator.create_chart.call_count == 2


def test_mask_api_key(mock_data_fetcher):
    """Test API key masking logic."""
    integration = GeminiIntegration(mock_data_fetcher, api_key="secret_key_123")

    masked = integration._mask_api_key("Error with key secret_key_123 here")
    assert "secret_key_123" not in masked
    assert "********" in masked

    # Test no key
    integration_no_key = GeminiIntegration(mock_data_fetcher)
    original = "Some error message"
    assert integration_no_key._mask_api_key(original) == original


def test_safe_logging(mock_data_fetcher):
    """Test that logging methods use masking."""
    integration = GeminiIntegration(mock_data_fetcher, api_key="secret_key_123")

    with mock_patch("modules.auto_trade.core.gemini_integration.log_error") as mock_log:
        integration._safe_log_error("Error: secret_key_123 failed")
        mock_log.assert_called_once()
        args = mock_log.call_args[0][0]
        assert "secret_key_123" not in args
        assert "********" in args
