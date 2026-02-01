"""
Tests for SignalPipeline.
"""

import time
from unittest.mock import MagicMock

import pytest

from modules.auto_trade.core.atc_scanner import SignalResult
from modules.auto_trade.core.gemini_integration import GeminiSignal
from modules.auto_trade.core.signal_pipeline import FinalSignal, SignalPipeline


class TestSignalPipeline:
    @pytest.fixture
    def mock_components(self):
        return {
            "symbol_manager": MagicMock(),
            "atc_scanner": MagicMock(),
            "xgboost_filter": MagicMock(),
            "gemini_integration": MagicMock(),
            "signal_selector": MagicMock(),
        }

    @pytest.fixture
    def pipeline(self, mock_components):
        return SignalPipeline(
            symbol_manager=mock_components["symbol_manager"],
            atc_scanner=mock_components["atc_scanner"],
            xgboost_filter=mock_components["xgboost_filter"],
            gemini_integration=mock_components["gemini_integration"],
            signal_selector=mock_components["signal_selector"],
            config={"max_symbols_to_scan": 10, "pipeline_timeout": 1},
        )

    def test_run_pipeline_success(self, pipeline, mock_components):
        """Test a successful pipeline run."""
        # Setup mocks
        mock_components["symbol_manager"].get_symbols.return_value = ["BTC/USDT"]

        atc_sig = SignalResult("BTC/USDT", 0.9, "LONG", {})
        mock_components["atc_scanner"].scan_symbols.return_value = [atc_sig]

        mock_components["xgboost_filter"].filter_signals.return_value = [atc_sig]

        gemini_sig = GeminiSignal("UP", "LONG", 0.9)
        mock_components["gemini_integration"].analyze_candidate.return_value = gemini_sig

        final_sig = FinalSignal("BTC/USDT", "LONG", 50000, 49000, 52000)
        mock_components["signal_selector"].select_best_signal.return_value = final_sig

        # Run
        result = pipeline.run_pipeline()

        # Verify
        assert result == final_sig
        mock_components["symbol_manager"].refresh_symbols.assert_called_once()
        mock_components["atc_scanner"].scan_symbols.assert_called_once()
        mock_components["xgboost_filter"].filter_signals.assert_called_once()
        mock_components["gemini_integration"].analyze_candidate.assert_called_once()
        mock_components["signal_selector"].select_best_signal.assert_called_once()

    def test_run_pipeline_no_symbols(self, pipeline, mock_components):
        """Test pipeline when no symbols are returned."""
        mock_components["symbol_manager"].get_symbols.return_value = []

        result = pipeline.run_pipeline()

        assert result is None
        mock_components["atc_scanner"].scan_symbols.assert_not_called()

    def test_run_pipeline_timeout(self, pipeline, mock_components):
        """Test pipeline timeout interruption."""
        # Set a very short timeout
        pipeline.pipeline_timeout = 0.1

        mock_components["symbol_manager"].get_symbols.return_value = ["BTC/USDT"]

        # Simulate delays in ATC scan
        def slow_scan(*args):
            time.sleep(0.2)
            return [SignalResult("BTC/USDT", 0.9, "LONG", {})]

        mock_components["atc_scanner"].scan_symbols.side_effect = slow_scan
        mock_components["xgboost_filter"].filter_signals.return_value = [SignalResult("BTC/USDT", 0.9, "LONG", {})]

        result = pipeline.run_pipeline()

        # Check that Gemini analysis was skipped because timeout happened during ATC scan
        # start_time (T) -> scan (sleep 0.2) -> loop check (T+0.2). T+0.2 - T = 0.2 > 0.1. Break.

        mock_components["gemini_integration"].analyze_candidate.assert_not_called()

        # Selector IS called with whatever we have (empty gemini map)
        mock_components["signal_selector"].select_best_signal.assert_called()

    def test_run_pipeline_exception(self, pipeline, mock_components):
        """Test generic exception handling."""
        mock_components["symbol_manager"].get_symbols.side_effect = Exception("API Error")

        result = pipeline.run_pipeline()

        assert result is None
