"""
Tests for pre-filter functions (pre_filter.py).

Tests cover:
- pre_filter_symbols_with_voting()
- pre_filter_symbols_with_hybrid()
- Error handling
- Edge cases
"""

import sys
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from modules.gemini_chart_analyzer.core.prefilter.workflow import run_prefilter_worker

# Test constants
TEST_LIMIT = 500
TEST_TIMEFRAME = "1h"
TEST_PERCENTAGE = 100.0


@pytest.fixture
def sample_symbols():
    """Sample trading symbols."""
    return ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]


@pytest.fixture
def mock_voting_analyzer():
    """Mock VotingAnalyzer with signals."""
    analyzer = Mock()
    analyzer.selected_timeframe = TEST_TIMEFRAME
    analyzer.atc_analyzer = Mock()
    analyzer.atc_analyzer.selected_timeframe = TEST_TIMEFRAME
    analyzer.run_atc_scan.return_value = True
    analyzer.args = Mock()  # Add args mock

    # Mock long_signals_atc and short_signals_atc (needed for Stage 1)
    analyzer.long_signals_atc = pd.DataFrame({"symbol": ["BTC/USDT", "ETH/USDT"], "signal": [0.9, 0.8]})
    analyzer.short_signals_atc = pd.DataFrame({"symbol": ["BNB/USDT"], "signal": [-0.7]})

    # Mock calculate_signals_for_all_indicators return value (needed for Stage 2/3)
    analyzer.calculate_signals_for_all_indicators.return_value = pd.DataFrame(
        {"symbol": ["BTC/USDT", "ETH/USDT", "BNB/USDT"], "atc_signal": [0.9, 0.8, -0.7], "atc_vote": [1, 1, 1]}
    )

    # Mock apply_voting_system return value
    # Stage 2: Range Osc + SPC
    analyzer.apply_voting_system.side_effect = [
        # Stage 2 long
        pd.DataFrame({"symbol": ["BTC/USDT", "ETH/USDT"], "weighted_score": [0.9, 0.8]}),
        # Stage 2 short
        pd.DataFrame({"symbol": ["BNB/USDT"], "weighted_score": [0.7]}),
        # Stage 3 long (if reached)
        pd.DataFrame({"symbol": ["BTC/USDT", "ETH/USDT"], "weighted_score": [0.9, 0.8]}),
        # Stage 3 short (if reached)
        pd.DataFrame({"symbol": ["BNB/USDT"], "weighted_score": [0.7]}),
    ]

    return analyzer


class TestPreFilterWorker:
    """Test run_prefilter_worker function."""

    def test_pre_filter_empty_symbols(self):
        """Test with empty symbol list."""
        result = run_prefilter_worker([], TEST_PERCENTAGE, TEST_TIMEFRAME, TEST_LIMIT)
        assert result == []

    def test_pre_filter_voting_success(self, sample_symbols, mock_voting_analyzer):
        """Test successful pre-filtering with voting mode."""
        with (
            patch("modules.common.core.exchange_manager.ExchangeManager"),
            patch("modules.common.core.data_fetcher.DataFetcher"),
            patch("modules.gemini_chart_analyzer.core.prefilter.workflow.VotingAnalyzer") as mock_voting_class,
        ):
            mock_voting_class.return_value = mock_voting_analyzer

            # Mock _filter_stage_1_atc to return symbols directly to simplify test
            with patch("modules.gemini_chart_analyzer.core.prefilter.workflow._filter_stage_1_atc") as mock_stage1:
                mock_stage1.return_value = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

                # Mock _filter_stage_2_osc_spc
                with patch(
                    "modules.gemini_chart_analyzer.core.prefilter.workflow._filter_stage_2_osc_spc"
                ) as mock_stage2:
                    mock_stage2.return_value = (["BTC/USDT", "ETH/USDT", "BNB/USDT"], {})

                    # Mock _filter_stage_3_ml_models
                    with patch(
                        "modules.gemini_chart_analyzer.core.prefilter.workflow._filter_stage_3_ml_models"
                    ) as mock_stage3:
                        mock_stage3.return_value = (["BTC/USDT", "ETH/USDT", "BNB/USDT"], {})

                        result = run_prefilter_worker(
                            sample_symbols, TEST_PERCENTAGE, TEST_TIMEFRAME, TEST_LIMIT, mode="voting"
                        )

                        # Should return symbols with signals (BTC, ETH, BNB)
                        assert len(result) == 3
                        assert "BTC/USDT" in result
                        assert "ETH/USDT" in result
                        assert "BNB/USDT" in result
                        assert "SOL/USDT" not in result
                        assert "ADA/USDT" not in result

    def test_pre_filter_error_handling(self, sample_symbols):
        """Test error handling in pre-filter."""
        with (
            patch("modules.common.core.exchange_manager.ExchangeManager"),
            patch("modules.common.core.data_fetcher.DataFetcher"),
            patch("modules.gemini_chart_analyzer.core.prefilter.workflow.VotingAnalyzer") as mock_voting_class,
        ):
            mock_voting_class.side_effect = Exception("Test error")

            result = run_prefilter_worker(sample_symbols, TEST_PERCENTAGE, TEST_TIMEFRAME, TEST_LIMIT, mode="voting")

            # Should return all symbols on error
            assert result == sample_symbols
