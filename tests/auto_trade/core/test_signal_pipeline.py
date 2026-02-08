"""
Tests for SignalPipeline.
"""

import time
from unittest.mock import MagicMock

import pytest

from modules.auto_trade.core.signal_pipeline import FinalSignal, SignalPipeline


class TestSignalPipeline:
    """Uses conftest fixtures: pipeline, mock_components."""

    def test_run_pipeline_success(self, pipeline, mock_components, sample_signal_result, sample_gemini_signal):
        """Test a successful pipeline run."""
        # Setup mocks
        mock_components["symbol_manager"].get_symbols.return_value = ["BTC/USDT"]

        atc_sig = sample_signal_result(symbol="BTC/USDT", score=0.9, signal_type="LONG")
        mock_components["atc_scanner"].scan_symbols.return_value = [atc_sig]

        mock_components["xgboost_filter"].filter_signals.return_value = [atc_sig]

        mock_components["gemini_integration"].is_available.return_value = True

        gemini_sig = sample_gemini_signal(trend="UP", signal="LONG", confidence=0.9)
        mock_components["gemini_integration"].analyze_candidates_batch_async.return_value = {"BTC/USDT": gemini_sig}

        final_sig = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000)
        mock_components["signal_selector"].select_best_signal.return_value = final_sig

        # Run
        result = pipeline.run_pipeline()

        # Verify
        assert result == final_sig, "Expected final signal on successful pipeline run"
        mock_components["symbol_manager"].refresh_symbols.assert_called_once()
        mock_components["atc_scanner"].scan_symbols.assert_called_once()
        mock_components["xgboost_filter"].filter_signals.assert_called_once()
        mock_components["gemini_integration"].is_available.assert_called_once()
        mock_components["gemini_integration"].analyze_candidates_batch_async.assert_called_once()
        mock_components["signal_selector"].select_best_signal.assert_called_once()

    def test_run_pipeline_no_symbols(self, pipeline, mock_components):
        """Test pipeline when no symbols are returned."""
        mock_components["symbol_manager"].get_symbols.return_value = []

        result = pipeline.run_pipeline()

        assert result is None, "Expected None when no symbols are returned"
        mock_components["atc_scanner"].scan_symbols.assert_not_called()

    def test_run_pipeline_timeout(self, pipeline, mock_components, sample_signal_result):
        """Test pipeline timeout interruption after ATC scan."""
        # Set a very short timeout
        pipeline.pipeline_timeout = 0.1

        mock_components["symbol_manager"].get_symbols.return_value = ["BTC/USDT"]

        # Simulate delays in ATC scan
        def slow_scan(*args):
            time.sleep(0.2)
            return [sample_signal_result(symbol="BTC/USDT")]

        mock_components["atc_scanner"].scan_symbols.side_effect = slow_scan

        result = pipeline.run_pipeline()

        # Pipeline should timeout after ATC scan and return None
        assert result is None, "Expected None when pipeline times out"

        # Check that Gemini analysis was skipped because timeout happened
        mock_components["gemini_integration"].is_available.assert_not_called()

        # Selector should NOT be called due to timeout
        mock_components["signal_selector"].select_best_signal.assert_not_called()

    def test_run_pipeline_exception(self, pipeline, mock_components):
        """Test generic exception handling."""
        mock_components["symbol_manager"].get_symbols.side_effect = Exception("API Error")

        result = pipeline.run_pipeline()

        assert result is None, "Expected None on exception"

    def test_run_pipeline_no_xgboost_signals(self, pipeline, mock_components, sample_signal_result):
        """Test when no signals pass XGBoost filter."""
        mock_components["symbol_manager"].get_symbols.return_value = ["BTC/USDT"]
        mock_components["atc_scanner"].scan_symbols.return_value = [sample_signal_result(symbol="BTC/USDT")]
        mock_components["xgboost_filter"].filter_signals.return_value = []

        result = pipeline.run_pipeline()

        assert result is None, "Expected None when XGBoost filter returns no signals"
        mock_components["gemini_integration"].analyze_candidate.assert_not_called()

    def test_run_pipeline_gemini_unavailable(self, pipeline, mock_components, sample_signal_result):
        """Test when Gemini API is not available."""
        mock_components["symbol_manager"].get_symbols.return_value = ["BTC/USDT"]
        mock_components["atc_scanner"].scan_symbols.return_value = [sample_signal_result(symbol="BTC/USDT")]
        mock_components["xgboost_filter"].filter_signals.return_value = [sample_signal_result(symbol="BTC/USDT")]
        mock_components["gemini_integration"].is_available.return_value = False
        mock_components["signal_selector"].select_best_signal.return_value = None

        result = pipeline.run_pipeline()

        assert result is None, "Expected None when Gemini is unavailable and no final signal is selected"
        mock_components["gemini_integration"].analyze_candidate.assert_not_called()

    def test_run_pipeline_persistence_success(
        self, pipeline, mock_components, sample_signal_result, sample_gemini_signal
    ):
        """Test signal persistence on successful pipeline."""
        mock_persistence = MagicMock()
        pipeline.signal_persistence = mock_persistence

        mock_components["symbol_manager"].get_symbols.return_value = ["BTC/USDT"]
        mock_components["atc_scanner"].scan_symbols.return_value = [sample_signal_result(symbol="BTC/USDT")]
        mock_components["xgboost_filter"].filter_signals.return_value = [sample_signal_result(symbol="BTC/USDT")]
        mock_components["gemini_integration"].analyze_candidate.return_value = sample_gemini_signal()

        final_signal = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000)
        mock_components["signal_selector"].select_best_signal.return_value = final_signal

        result = pipeline.run_pipeline()

        assert result == final_signal
        mock_persistence.save_signal.assert_called_once_with(final_signal)

    def test_run_pipeline_no_persistence_configured(
        self, pipeline, mock_components, sample_signal_result, sample_gemini_signal
    ):
        """Test pipeline without persistence configured."""
        pipeline.signal_persistence = None

        mock_components["symbol_manager"].get_symbols.return_value = ["BTC/USDT"]
        mock_components["atc_scanner"].scan_symbols.return_value = [sample_signal_result(symbol="BTC/USDT")]
        mock_components["xgboost_filter"].filter_signals.return_value = [sample_signal_result(symbol="BTC/USDT")]
        mock_components["gemini_integration"].analyze_candidate.return_value = sample_gemini_signal()

        final_signal = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000)
        mock_components["signal_selector"].select_best_signal.return_value = final_signal

        result = pipeline.run_pipeline()

        assert result == final_signal
        # Should not crash when persistence is None

    def test_run_pipeline_multiple_candidates(self, pipeline, mock_components, sample_signal_result):
        """Test pipeline with multiple signals through each stage."""
        mock_components["symbol_manager"].get_symbols.return_value = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

        atc_signals = [
            sample_signal_result(symbol="BTC/USDT", score=0.9),
            sample_signal_result(symbol="ETH/USDT", score=0.8),
            sample_signal_result(symbol="BNB/USDT", score=0.7, signal_type="SHORT"),
        ]
        mock_components["atc_scanner"].scan_symbols.return_value = atc_signals

        xgb_signals = [
            sample_signal_result(symbol="BTC/USDT", score=0.9),
            sample_signal_result(symbol="ETH/USDT", score=0.8),
        ]
        mock_components["xgboost_filter"].filter_signals.return_value = xgb_signals

        mock_components["gemini_integration"].is_available.return_value = True

        mock_components["signal_selector"].select_best_signal.return_value = None

        result = pipeline.run_pipeline()

        # Gemini batch analysis should be called
        mock_components["gemini_integration"].analyze_candidates_batch_async.assert_called_once()
        assert result is None, "Expected None when selector returns no final signal"

    def test_run_pipeline_max_symbols_limiting(self, pipeline, mock_components):
        """Test that max_symbols correctly limits the scan."""
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

    def test_run_pipeline_config_validation(self):
        """Test that invalid config values raise ValueError."""
        mock_components = {
            "symbol_manager": MagicMock(),
            "atc_scanner": MagicMock(),
            "xgboost_filter": MagicMock(),
            "gemini_integration": MagicMock(),
            "signal_selector": MagicMock(),
        }

        # Test negative max_symbols
        with pytest.raises(ValueError, match="max_symbols_to_scan must be positive"):
            SignalPipeline(
                symbol_manager=mock_components["symbol_manager"],
                atc_scanner=mock_components["atc_scanner"],
                xgboost_filter=mock_components["xgboost_filter"],
                gemini_integration=mock_components["gemini_integration"],
                signal_selector=mock_components["signal_selector"],
                config={"max_symbols_to_scan": -1},
            )

        # Test zero pipeline_timeout
        with pytest.raises(ValueError, match="pipeline_timeout must be positive"):
            SignalPipeline(
                symbol_manager=mock_components["symbol_manager"],
                atc_scanner=mock_components["atc_scanner"],
                xgboost_filter=mock_components["xgboost_filter"],
                gemini_integration=mock_components["gemini_integration"],
                signal_selector=mock_components["signal_selector"],
                config={"pipeline_timeout": 0},
            )
