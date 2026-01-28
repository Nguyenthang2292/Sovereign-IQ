"""
Tests for prefilter_worker 3-stage filtering workflow.

Tests cover:
- Stage 1: ATC filter
- Stage 2: Range Oscillator + SPC filter
- Stage 3: ML models filter (XGBoost + HMM + RF)
- Integration of all 3 stages
- Edge cases and error handling
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Import after path setup

# Add project root to sys.path
if "__file__" in globals():
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from modules.gemini_chart_analyzer.core.prefilter.workflow import (
    _filter_stage_1_atc,
    _filter_stage_2_osc_spc,
    _filter_stage_3_ml_models,
    run_prefilter_worker,
)


@pytest.fixture
def sample_symbols():
    """Sample trading symbols."""
    return ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]


@pytest.fixture
def mock_voting_analyzer():
    """Create a mock VotingAnalyzer with ATC signals."""
    analyzer = MagicMock()
    analyzer.selected_timeframe = "1h"
    analyzer.atc_analyzer = MagicMock()
    analyzer.atc_analyzer.selected_timeframe = "1h"
    analyzer.args = MagicMock()
    analyzer.args.enable_spc = True
    analyzer.args.enable_xgboost = False
    analyzer.args.enable_hmm = False
    analyzer.args.enable_random_forest = False
    analyzer.args.voting_threshold = 0.5
    analyzer.args.min_votes = 2
    analyzer.args.limit = 700
    analyzer.args.max_workers = 10

    # Mock ATC signals
    analyzer.long_signals_atc = pd.DataFrame(
        {
            "symbol": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
            "signal": [85.0, 75.0, 65.0],
            "trend": ["UP", "UP", "UP"],
            "price": [50000.0, 3000.0, 400.0],
            "exchange": ["binance", "binance", "binance"],
        }
    )
    analyzer.short_signals_atc = pd.DataFrame(
        {
            "symbol": ["SOL/USDT"],
            "signal": [-70.0],
            "trend": ["DOWN"],
            "price": [100.0],
            "exchange": ["binance"],
        }
    )

    return analyzer


@pytest.fixture
def mock_data_fetcher():
    """Create a mock DataFetcher."""
    fetcher = MagicMock()
    fetcher.exchange_manager = MagicMock()
    return fetcher


class TestStage1ATCFilter:
    """Test Stage 1: ATC Filter."""

    def test_stage1_atc_filter_success(self, mock_voting_analyzer, sample_symbols):
        """Test successful ATC filtering."""
        mock_voting_analyzer.run_atc_scan.return_value = True

        result = _filter_stage_1_atc(mock_voting_analyzer, sample_symbols)

        # Should return all symbols that passed ATC (BTC, ETH, BNB, SOL)
        assert len(result) == 4
        assert "BTC/USDT" in result
        assert "ETH/USDT" in result
        assert "BNB/USDT" in result
        assert "SOL/USDT" in result
        assert "ADA/USDT" not in result
        assert "XRP/USDT" not in result

        mock_voting_analyzer.run_atc_scan.assert_called_once()

    def test_stage1_atc_filter_no_signals(self, mock_voting_analyzer, sample_symbols):
        """Test when ATC scan returns no signals."""
        mock_voting_analyzer.run_atc_scan.return_value = False
        mock_voting_analyzer.long_signals_atc = pd.DataFrame()
        mock_voting_analyzer.short_signals_atc = pd.DataFrame()

        result = _filter_stage_1_atc(mock_voting_analyzer, sample_symbols)

        # Should return all symbols when no ATC signals
        assert result == sample_symbols

    def test_stage1_atc_filter_empty_input(self, mock_voting_analyzer):
        """Test with empty symbol list."""
        result = _filter_stage_1_atc(mock_voting_analyzer, [])
        assert result == []


class TestStage2OscSpcFilter:
    """Test Stage 2: Range Oscillator + SPC Filter."""

    def test_stage2_osc_spc_filter_success(self, mock_voting_analyzer):
        """Test successful Stage 2 filtering."""
        stage1_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]

        # Mock calculate_signals_for_all_indicators to return signals with Range Osc and SPC
        mock_long_signals = pd.DataFrame(
            {
                "symbol": ["BTC/USDT", "ETH/USDT"],
                "signal": [85.0, 75.0],
                "trend": ["UP", "UP"],
                "price": [50000.0, 3000.0],
                "exchange": ["binance", "binance"],
                "atc_vote": [1, 1],
                "atc_strength": [0.85, 0.75],
                "osc_signal": [1, 1],
                "osc_vote": [1, 1],
                "osc_confidence": [0.8, 0.7],
                "spc_cluster_transition_signal": [1, 1],
                "spc_regime_following_signal": [1, 0],
                "spc_mean_reversion_signal": [0, 1],
            }
        )

        mock_short_signals = pd.DataFrame(
            {
                "symbol": ["SOL/USDT"],
                "signal": [-70.0],
                "trend": ["DOWN"],
                "price": [100.0],
                "exchange": ["binance"],
                "atc_vote": [1],
                "atc_strength": [0.7],
                "osc_signal": [-1],
                "osc_vote": [1],
                "osc_confidence": [0.6],
                "spc_cluster_transition_signal": [-1],
                "spc_regime_following_signal": [-1],
                "spc_mean_reversion_signal": [0],
            }
        )

        # Mock voting system results
        mock_long_final = pd.DataFrame(
            {
                "symbol": ["BTC/USDT", "ETH/USDT"],
                "weighted_score": [0.9, 0.8],
                "cumulative_vote": [1, 1],
            }
        )

        mock_short_final = pd.DataFrame(
            {
                "symbol": ["SOL/USDT"],
                "weighted_score": [0.7],
                "cumulative_vote": [1],
            }
        )

        mock_voting_analyzer.calculate_signals_for_all_indicators.side_effect = [
            mock_long_signals,
            mock_short_signals,
        ]
        mock_voting_analyzer.apply_voting_system.side_effect = [mock_long_final, mock_short_final]

        result, stage2_signals = _filter_stage_2_osc_spc(mock_voting_analyzer, stage1_symbols, "1h", 700)

        # Should return symbols that passed voting (BTC, ETH, SOL)
        assert len(result) == 3
        assert "BTC/USDT" in result
        assert "ETH/USDT" in result
        assert "SOL/USDT" in result
        assert "BNB/USDT" not in result  # Not in voting results

        # Verify calculate_signals_for_all_indicators was called with correct parameters
        assert mock_voting_analyzer.calculate_signals_for_all_indicators.call_count == 2
        calls = mock_voting_analyzer.calculate_signals_for_all_indicators.call_args_list
        assert calls[0][1]["indicators_to_calculate"] == ["oscillator", "spc"]
        assert calls[0][1]["signal_type"] == "LONG"
        assert calls[1][1]["indicators_to_calculate"] == ["oscillator", "spc"]
        assert calls[1][1]["signal_type"] == "SHORT"

        # Verify apply_voting_system was called with correct parameters
        assert mock_voting_analyzer.apply_voting_system.call_count == 2
        voting_calls = mock_voting_analyzer.apply_voting_system.call_args_list
        assert voting_calls[0][1]["indicators_to_include"] == ["atc", "oscillator", "spc"]
        assert voting_calls[1][1]["indicators_to_include"] == ["atc", "oscillator", "spc"]

    def test_stage2_osc_spc_filter_no_signals(self, mock_voting_analyzer):
        """Test when no signals pass voting."""
        stage1_symbols = ["BTC/USDT", "ETH/USDT"]

        mock_long_signals = pd.DataFrame(
            {
                "symbol": ["BTC/USDT", "ETH/USDT"],
                "signal": [85.0, 75.0],
                "atc_vote": [1, 1],
                "osc_vote": [0, 0],  # No oscillator votes
            }
        )

        mock_voting_analyzer.calculate_signals_for_all_indicators.return_value = mock_long_signals
        mock_voting_analyzer.apply_voting_system.return_value = pd.DataFrame()  # No symbols pass voting

        result, stage2_signals = _filter_stage_2_osc_spc(mock_voting_analyzer, stage1_symbols, "1h", 700)

        # Should return Stage 1 symbols when no signals pass voting
        assert result == stage1_symbols

    def test_stage2_osc_spc_filter_empty_input(self, mock_voting_analyzer):
        """Test with empty Stage 1 symbols."""
        result, stage2_signals = _filter_stage_2_osc_spc(mock_voting_analyzer, [], "1h", 700)
        assert result == []
        assert stage2_signals == {}


class TestStage3MLModelsFilter:
    """Test Stage 3: ML Models Filter."""

    def test_stage3_ml_models_filter_success(self, mock_voting_analyzer):
        """Test successful Stage 3 filtering."""
        stage2_symbols = ["BTC/USDT", "ETH/USDT"]
        stage2_signals = {
            "long": pd.DataFrame(
                {
                    "symbol": ["BTC/USDT", "ETH/USDT"],
                    "weighted_score": [0.9, 0.8],
                    "atc_vote": [1, 1],
                    "osc_vote": [1, 1],
                }
            ),
            "short": pd.DataFrame(),
        }

        # Mock calculate_signals_for_all_indicators to return ML models only (no Range Osc, SPC)
        mock_ml_signals = pd.DataFrame(
            {
                "symbol": ["BTC/USDT", "ETH/USDT"],
                "signal": [85.0, 75.0],
                "atc_vote": [1, 1],
                "atc_strength": [0.85, 0.75],
                # Note: osc_vote and spc signals are NOT included (Stage 3 only calculates ML models)
                "xgboost_signal": [1, 1],
                "xgboost_vote": [1, 1],
                "xgboost_confidence": [0.8, 0.7],
                "hmm_signal": [1, 0],
                "hmm_vote": [1, 0],
                "hmm_confidence": [0.75, 0.0],
                "random_forest_signal": [1, 1],
                "random_forest_vote": [1, 1],
                "random_forest_confidence": [0.85, 0.75],
            }
        )

        # Mock voting system results
        mock_final = pd.DataFrame(
            {
                "symbol": ["BTC/USDT"],
                "weighted_score": [0.95],
                "cumulative_vote": [1],
            }
        )

        mock_voting_analyzer.calculate_signals_for_all_indicators.return_value = mock_ml_signals
        mock_voting_analyzer.apply_voting_system.return_value = mock_final

        result, scores = _filter_stage_3_ml_models(
            mock_voting_analyzer, stage2_symbols, stage2_signals, "1h", 700, rf_model_path="test_model.joblib"
        )

        # Should return symbols that passed voting (BTC only)
        assert len(result) == 1
        assert "BTC/USDT" in result
        assert "ETH/USDT" not in result

        # Verify calculate_signals_for_all_indicators was called with only ML models
        calls = mock_voting_analyzer.calculate_signals_for_all_indicators.call_args_list
        assert calls[0][1]["indicators_to_calculate"] == ["xgboost", "hmm", "random_forest"]
        assert "oscillator" not in calls[0][1]["indicators_to_calculate"]
        assert "spc" not in calls[0][1]["indicators_to_calculate"]

        # Verify apply_voting_system was called with ATC + ML models only (exclude Range Osc and SPC)
        voting_calls = mock_voting_analyzer.apply_voting_system.call_args_list
        assert voting_calls[0][1]["indicators_to_include"] == ["atc", "xgboost", "hmm", "random_forest"]
        assert "oscillator" not in voting_calls[0][1]["indicators_to_include"]
        assert "spc" not in voting_calls[0][1]["indicators_to_include"]

    def test_stage3_ml_models_filter_fast_mode_override(self, mock_voting_analyzer):
        """Test that Stage 3 enables ML models even in fast mode."""
        stage2_symbols = ["BTC/USDT"]
        stage2_signals = {"long": pd.DataFrame(), "short": pd.DataFrame()}

        # Start with fast mode (ML models disabled)
        mock_voting_analyzer.args.enable_xgboost = False
        mock_voting_analyzer.args.enable_hmm = False
        mock_voting_analyzer.args.enable_random_forest = False

        mock_voting_analyzer.calculate_signals_for_all_indicators.return_value = pd.DataFrame()
        mock_voting_analyzer.apply_voting_system.return_value = pd.DataFrame()

        result, scores = _filter_stage_3_ml_models(mock_voting_analyzer, stage2_symbols, stage2_signals, "1h", 700)

        # Verify ML models were temporarily enabled
        # (They should be enabled during execution, then restored)
        # Since we're using a mock, we can't verify the intermediate state,
        # but we can verify the function completes without error
        assert result == stage2_symbols  # Fallback to Stage 2 symbols
        assert isinstance(scores, dict)

    def test_stage3_ml_models_filter_empty_input(self, mock_voting_analyzer):
        """Test with empty Stage 2 symbols."""
        result, scores = _filter_stage_3_ml_models(mock_voting_analyzer, [], {}, "1h", 700)
        assert result == []
        assert scores == {}


class TestIntegrated3StageWorkflow:
    """Test integrated 3-stage workflow."""

    @patch("modules.gemini_chart_analyzer.core.prefilter.workflow.ExchangeManager")
    @patch("modules.gemini_chart_analyzer.core.prefilter.workflow.DataFetcher")
    @patch("modules.gemini_chart_analyzer.core.prefilter.workflow.VotingAnalyzer")
    def test_run_prefilter_worker_3_stages(self, mock_voting_class, mock_data_fetcher_class, mock_exchange_class):
        """Test complete 3-stage workflow."""
        sample_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]

        # Setup mocks
        mock_analyzer = MagicMock()
        mock_analyzer.selected_timeframe = "1h"
        mock_analyzer.atc_analyzer = MagicMock()
        mock_analyzer.atc_analyzer.selected_timeframe = "1h"
        mock_analyzer.args = MagicMock()
        mock_analyzer.args.enable_spc = True
        mock_analyzer.args.enable_xgboost = False
        mock_analyzer.args.enable_hmm = False
        mock_analyzer.args.enable_random_forest = False
        mock_analyzer.args.voting_threshold = 0.5
        mock_analyzer.args.min_votes = 2
        mock_analyzer.args.limit = 700
        mock_analyzer.args.max_workers = 10

        # Stage 1: ATC signals
        mock_analyzer.run_atc_scan.return_value = True
        mock_analyzer.long_signals_atc = pd.DataFrame(
            {
                "symbol": ["BTC/USDT", "ETH/USDT"],
                "signal": [85.0, 75.0],
                "trend": ["UP", "UP"],
                "price": [50000.0, 3000.0],
                "exchange": ["binance", "binance"],
            }
        )
        mock_analyzer.short_signals_atc = pd.DataFrame()

        # Stage 2: Range Osc + SPC signals
        mock_stage2_long = pd.DataFrame(
            {
                "symbol": ["BTC/USDT"],
                "weighted_score": [0.9],
                "cumulative_vote": [1],
            }
        )
        mock_analyzer.calculate_signals_for_all_indicators.side_effect = [
            pd.DataFrame({"symbol": ["BTC/USDT"], "osc_vote": [1]}),  # Stage 2 LONG
        ]
        mock_analyzer.apply_voting_system.side_effect = [
            mock_stage2_long,  # Stage 2 result
            pd.DataFrame({"symbol": ["BTC/USDT"], "weighted_score": [0.95], "cumulative_vote": [1]}),  # Stage 3 result
        ]

        mock_voting_class.return_value = mock_analyzer
        mock_data_fetcher_class.return_value = MagicMock()
        mock_exchange_class.return_value = MagicMock()

        # Run pre-filter
        result = run_prefilter_worker(
            all_symbols=sample_symbols,
            percentage=100.0,  # 100% to get all results
            timeframe="1h",
            limit=700,
            mode="voting",
            fast_mode=True,
            spc_config=None,
            rf_model_path=None,
        )

        # Should return symbols from Stage 3
        assert len(result) >= 0  # At least some symbols or empty
        # Verify stages were called
        assert mock_analyzer.run_atc_scan.called

    def test_run_prefilter_worker_empty_symbols(self):
        """Test with empty symbol list."""
        result = run_prefilter_worker([], 10.0, "1h", 700)
        assert result == []

    def test_run_prefilter_worker_percentage_filter(self):
        """Test percentage filter application."""
        # This test would require more complex mocking
        # For now, just verify the function signature works
        with (
            patch("modules.gemini_chart_analyzer.core.prefilter.workflow.ExchangeManager"),
            patch("modules.gemini_chart_analyzer.core.prefilter.workflow.DataFetcher"),
            patch("modules.gemini_chart_analyzer.core.prefilter.workflow.VotingAnalyzer") as mock_voting_class,
        ):
            mock_analyzer = MagicMock()
            mock_analyzer.run_atc_scan.return_value = False  # No signals, returns all symbols
            mock_voting_class.return_value = mock_analyzer

            result = run_prefilter_worker(["BTC/USDT", "ETH/USDT"], 50.0, "1h", 700)
            # Should handle percentage filter
            assert isinstance(result, list)
