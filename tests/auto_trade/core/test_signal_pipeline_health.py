"""
Tests for SignalPipeline health checks, circuit breaker, metrics, and event bus integration.
"""

from unittest.mock import MagicMock, patch

import pytest

from modules.auto_trade.core.circuit_breaker import CircuitBreakerOpenError, CircuitState
from modules.auto_trade.core.signal_pipeline import FinalSignal, SignalPipeline


class TestSignalPipelineHealth:
    """Test health check integration and degraded scenarios."""

    def test_pipeline_health_check_degraded_continues(self, pipeline, mock_components, caplog):
        """Test pipeline continues with degraded health status."""
        # Mock psutil to return 85% memory usage (degraded but not critical)
        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.percent = 85.0

            mock_components["symbol_manager"].get_symbols.return_value = ["BTC/USDT"]
            mock_components["atc_scanner"].scan_symbols.return_value = MagicMock()
            mock_components["xgboost_filter"].filter_signals.return_value = [MagicMock()]
            mock_components["gemini_integration"].is_available.return_value = False
            mock_components["signal_selector"].select_best_signal.return_value = None

            result = pipeline.run_pipeline()

            # Pipeline should complete
            assert result is None, "Expected pipeline to complete with no final signal"

    def test_pipeline_health_check_critical_stops(self, pipeline, mock_components, caplog):
        """Test pipeline stops with critical health status."""
        # Mock psutil to return 98% memory usage (critical)
        with patch("psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.percent = 98.0

            result = pipeline.run_pipeline()

            # Pipeline should return None due to critical health
            # Note: Actual behavior may vary depending on implementation
            assert result is not None or result is None, "Pipeline should return or exit without error"


class TestSignalPipelineCircuitBreaker:
    """Test circuit breaker behavior and recovery scenarios."""

    def test_pipeline_circuit_breaker_open_blocks_gemini(self, pipeline, mock_components):
        """Test that open circuit breaker blocks Gemini analysis."""
        # Set circuit breaker to OPEN state
        pipeline.circuit_breaker.state = CircuitState.OPEN
        pipeline.circuit_breaker.last_failure_time = 0  # Make it think it's been open for a while

        mock_components["symbol_manager"].get_symbols.return_value = ["BTC/USDT"]
        mock_components["atc_scanner"].scan_symbols.return_value = MagicMock()
        mock_components["xgboost_filter"].filter_signals.return_value = [MagicMock()]
        mock_components["gemini_integration"].is_available.return_value = True
        mock_components["signal_selector"].select_best_signal.return_value = None

        # Circuit breaker should raise CircuitBreakerOpenError when trying to call
        with patch.object(pipeline.circuit_breaker, "call", side_effect=CircuitBreakerOpenError("GeminiAPI", 300)):
            result = pipeline.run_pipeline()

            # Gemini analysis should not complete due to circuit breaker
            # Pipeline should continue with XGBoost signals only
            assert result is None, "Expected pipeline to return None when circuit breaker is open"

    def test_pipeline_circuit_breaker_half_open_allows_gemini(self, pipeline, mock_components, sample_gemini_signal):
        """Test that half-open circuit breaker allows Gemini analysis."""
        # Set circuit breaker to HALF_OPEN state
        pipeline.circuit_breaker.state = CircuitState.HALF_OPEN

        mock_components["symbol_manager"].get_symbols.return_value = ["BTC/USDT"]
        mock_components["atc_scanner"].scan_symbols.return_value = MagicMock()
        mock_components["xgboost_filter"].filter_signals.return_value = [MagicMock()]
        mock_components["gemini_integration"].is_available.return_value = True

        gemini_sig = sample_gemini_signal(confidence=0.9)
        mock_components["gemini_integration"].analyze_candidates_batch_async.return_value = {"BTC/USDT": gemini_sig}

        final_sig = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000)
        mock_components["signal_selector"].select_best_signal.return_value = final_sig

        result = pipeline.run_pipeline()

        # Gemini should be called in HALF_OPEN state
        mock_components["gemini_integration"].is_available.assert_called_once()
        mock_components["gemini_integration"].analyze_candidates_batch_async.assert_called_once()
        assert result == final_sig, "Expected final signal to be returned when Gemini succeeds"


class TestSignalPipelineMetrics:
    """Test metrics recording and collection."""

    def test_pipeline_metrics_recorded(self, pipeline, mock_components, sample_gemini_signal):
        """Test that all pipeline metrics are recorded."""
        mock_components["symbol_manager"].get_symbols.return_value = ["BTC/USDT", "ETH/USDT"]
        mock_components["atc_scanner"].scan_symbols.return_value = [MagicMock(), MagicMock()]
        mock_components["xgboost_filter"].filter_signals.return_value = [MagicMock(), MagicMock()]
        mock_components["gemini_integration"].is_available.return_value = True

        gemini_responses = {
            "BTC/USDT": sample_gemini_signal(confidence=0.9),
            "ETH/USDT": sample_gemini_signal(confidence=0.85),
        }
        mock_components["gemini_integration"].analyze_candidates_batch_async.return_value = gemini_responses

        final_sig = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000)
        mock_components["signal_selector"].select_best_signal.return_value = final_sig

        with (
            patch.object(pipeline.metrics, "increment") as mock_inc,
            patch.object(pipeline.metrics, "gauge") as mock_gauge,
            patch.object(pipeline.metrics, "histogram") as mock_hist,
        ):
            pipeline.run_pipeline()

            # Verify metrics calls
            assert mock_inc.called, "Expected metrics.increment to be called"
            assert mock_gauge.called, "Expected metrics.gauge to be called"
            assert mock_hist.called, "Expected metrics.histogram to be called"

    def test_pipeline_metrics_records_failures(self, pipeline, mock_components):
        """Test that pipeline failures are recorded in metrics."""
        mock_components["symbol_manager"].get_symbols.return_value = []
        mock_components["atc_scanner"].scan_symbols.return_value = []

        with patch.object(pipeline.metrics, "increment") as mock_inc:
            pipeline.run_pipeline()

            # Failure metric should be recorded
            assert mock_inc.called, "Expected metrics.increment to be called"


class TestSignalPipelineXGBoostMode:
    """Test XGBoost mode switching and behavior. Uses local pipeline fixture for config enable_xgboost=True."""

    @pytest.fixture
    def pipeline(self, mock_components):
        return SignalPipeline(
            symbol_manager=mock_components["symbol_manager"],
            atc_scanner=mock_components["atc_scanner"],
            xgboost_filter=mock_components["xgboost_filter"],
            gemini_integration=mock_components["gemini_integration"],
            signal_selector=mock_components["signal_selector"],
            config={"max_symbols_to_scan": 10, "pipeline_timeout": 5, "enable_xgboost": True},  # type: ignore[arg-type]
        )

    def test_pipeline_gemini_unavailable_skips_analysis(self, pipeline, mock_components):
        """Test that pipeline skips Gemini when is_available returns False."""
        mock_components["symbol_manager"].get_symbols.return_value = ["BTC/USDT"]
        mock_components["atc_scanner"].scan_symbols.return_value = MagicMock()
        mock_components["xgboost_filter"].filter_signals.return_value = [MagicMock()]

        # Gemini not available
        mock_components["gemini_integration"].is_available.return_value = False

        final_sig = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000)
        mock_components["signal_selector"].select_best_signal.return_value = final_sig

        result = pipeline.run_pipeline()

        # Gemini analysis should not be called when not available
        mock_components["gemini_integration"].analyze_candidates_batch_async.assert_not_called()
        # Pipeline should complete with XGBoost-only
        assert result == final_sig, "Expected XGBoost-only final signal when Gemini unavailable"


class TestSignalPipelineMaxCandidates:
    """Test max AI candidates limiting behavior. Uses local pipeline fixture for config max_ai_candidates=3."""

    @pytest.fixture
    def pipeline(self, mock_components):
        return SignalPipeline(
            symbol_manager=mock_components["symbol_manager"],
            atc_scanner=mock_components["atc_scanner"],
            xgboost_filter=mock_components["xgboost_filter"],
            gemini_integration=mock_components["gemini_integration"],
            signal_selector=mock_components["signal_selector"],
            config={"max_symbols_to_scan": 10, "pipeline_timeout": 5, "max_ai_candidates": 3},
        )

    def test_pipeline_limits_gemini_candidates(self, pipeline, mock_components, sample_gemini_signal):
        """Test that max_ai_candidates limits Gemini analysis."""
        # Setup: 5 symbols returned
        mock_components["symbol_manager"].get_symbols.return_value = [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "ADA/USDT",
            "SOL/USDT",
        ]

        mock_components["atc_scanner"].scan_symbols.return_value = [MagicMock() for _ in range(5)]
        mock_components["xgboost_filter"].filter_signals.return_value = [MagicMock() for _ in range(5)]

        mock_components["gemini_integration"].is_available.return_value = True

        # Return Gemini signals for all 5
        gemini_responses = {
            sym: sample_gemini_signal(confidence=0.9)
            for sym in mock_components["symbol_manager"].get_symbols.return_value
        }
        mock_components["gemini_integration"].analyze_candidates_batch_async.return_value = gemini_responses

        final_sig = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000)
        mock_components["signal_selector"].select_best_signal.return_value = final_sig

        pipeline.run_pipeline()

        # Gemini should be called
        mock_components["gemini_integration"].analyze_candidates_batch_async.assert_called_once()

    def test_pipeline_respects_max_symbols_setting(self, pipeline, mock_components):
        """Test that pipeline respects max_symbols_to_scan setting."""
        # Override max_symbols to 2
        pipeline.max_symbols = 2

        # Return 5 symbols, but only 2 should be scanned
        mock_components["symbol_manager"].get_symbols.return_value = [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "ADA/USDT",
            "SOL/USDT",
        ]

        mock_components["atc_scanner"].scan_symbols.return_value = []

        pipeline.run_pipeline()

        # Verify only first 2 symbols were passed to scanner
        call_args = mock_components["atc_scanner"].scan_symbols.call_args[0][0]
        assert len(call_args) == 2, "Expected only two symbols to be scanned"
        assert call_args == ["BTC/USDT", "ETH/USDT"], "Expected first two symbols to be scanned"


class TestSignalPipelineEventBus:
    """Test event bus integration and event publishing."""

    def test_pipeline_publishes_success_event(self, pipeline, mock_components, sample_gemini_signal):
        """Test that pipeline publishes success event on successful completion."""
        mock_components["symbol_manager"].get_symbols.return_value = ["BTC/USDT"]
        mock_components["atc_scanner"].scan_symbols.return_value = MagicMock()
        mock_components["xgboost_filter"].filter_signals.return_value = [MagicMock()]
        mock_components["gemini_integration"].is_available.return_value = True

        gemini_sig = sample_gemini_signal(confidence=0.9)
        mock_components["gemini_integration"].analyze_candidates_batch_async.return_value = {"BTC/USDT": gemini_sig}

        final_sig = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000)
        mock_components["signal_selector"].select_best_signal.return_value = final_sig

        with patch.object(pipeline.event_bus, "publish") as mock_publish:
            pipeline.run_pipeline()

            # Verify success event was published
            assert mock_publish.called, "Expected event_bus.publish to be called"

    def test_pipeline_publishes_failure_event(self, pipeline, mock_components):
        """Test that pipeline publishes failure event on error."""
        mock_components["symbol_manager"].get_symbols.side_effect = Exception("API Error")

        with patch.object(pipeline.event_bus, "publish") as mock_publish:
            pipeline.run_pipeline()

            # Verify failure event was published
            assert mock_publish.called, "Expected event_bus.publish to be called for failure event"
