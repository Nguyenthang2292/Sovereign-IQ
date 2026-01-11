
from unittest.mock import Mock, patch
import sys

import pandas as pd
import pytest

from modules.gemini_chart_analyzer.cli.pre_filter import pre_filter_symbols_with_hybrid, pre_filter_symbols_with_voting

from modules.gemini_chart_analyzer.cli.pre_filter import pre_filter_symbols_with_hybrid, pre_filter_symbols_with_voting

"""
Tests for pre-filter functions (pre_filter.py).

Tests cover:
- pre_filter_symbols_with_voting()
- pre_filter_symbols_with_hybrid()
- Error handling
- Edge cases
"""




# Test constants
TEST_LIMIT = 500
TEST_TIMEFRAME = "1h"


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

    # Mock long_signals_final and short_signals_final DataFrames
    analyzer.long_signals_final = pd.DataFrame({"symbol": ["BTC/USDT", "ETH/USDT"], "weighted_score": [0.9, 0.8]})
    analyzer.short_signals_final = pd.DataFrame({"symbol": ["BNB/USDT"], "weighted_score": [0.7]})

    return analyzer


@pytest.fixture
def mock_hybrid_analyzer():
    """Mock HybridAnalyzer with signals."""
    analyzer = Mock()
    analyzer.selected_timeframe = TEST_TIMEFRAME
    analyzer.atc_analyzer = Mock()
    analyzer.atc_analyzer.selected_timeframe = TEST_TIMEFRAME
    analyzer.run_atc_scan.return_value = True

    # Mock long_signals_confirmed and short_signals_confirmed DataFrames
    analyzer.long_signals_confirmed = pd.DataFrame({"symbol": ["BTC/USDT", "ETH/USDT"]})
    analyzer.short_signals_confirmed = pd.DataFrame({"symbol": ["BNB/USDT"]})

    return analyzer


class TestPreFilterWithVoting:
    """Test pre_filter_symbols_with_voting function."""

    def test_pre_filter_voting_empty_symbols(self):
        """Test with empty symbol list."""
        result = pre_filter_symbols_with_voting([], TEST_TIMEFRAME, TEST_LIMIT)
        assert result == []

    def test_pre_filter_voting_no_atc_signals(self, sample_symbols):
        """Test when ATC scan returns no signals."""
        with (
            patch("modules.gemini_chart_analyzer.cli.pre_filter.ExchangeManager") as mock_exchange_mgr,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.DataFetcher") as mock_data_fetcher,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.VotingAnalyzer") as mock_voting_class,
        ):
            mock_analyzer = Mock()
            mock_analyzer.selected_timeframe = TEST_TIMEFRAME
            mock_analyzer.atc_analyzer = Mock()
            mock_analyzer.atc_analyzer.selected_timeframe = TEST_TIMEFRAME
            mock_analyzer.run_atc_scan.return_value = False  # No ATC signals
            mock_voting_class.return_value = mock_analyzer

            result = pre_filter_symbols_with_voting(sample_symbols, TEST_TIMEFRAME, TEST_LIMIT)

            # Should return all symbols when no ATC signals
            assert result == sample_symbols

    def test_pre_filter_voting_success(self, sample_symbols, mock_voting_analyzer):
        """Test successful pre-filtering with voting mode."""
        with (
            patch("modules.gemini_chart_analyzer.cli.pre_filter.ExchangeManager") as mock_exchange_mgr,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.DataFetcher") as mock_data_fetcher,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.VotingAnalyzer") as mock_voting_class,
        ):
            mock_voting_class.return_value = mock_voting_analyzer
            mock_voting_analyzer.calculate_and_vote = Mock()

            result = pre_filter_symbols_with_voting(sample_symbols, TEST_TIMEFRAME, TEST_LIMIT)

            # Should return symbols with signals (BTC, ETH, BNB)
            assert len(result) == 3
            assert "BTC/USDT" in result
            assert "ETH/USDT" in result
            assert "BNB/USDT" in result
            assert "SOL/USDT" not in result
            assert "ADA/USDT" not in result

            # Verify VotingAnalyzer was called correctly
            mock_voting_class.assert_called_once()
            mock_voting_analyzer.run_atc_scan.assert_called_once()
            mock_voting_analyzer.calculate_and_vote.assert_called_once()

    def test_pre_filter_voting_no_signals_found(self, sample_symbols):
        """Test when no signals are found after voting."""
        with (
            patch("modules.gemini_chart_analyzer.cli.pre_filter.ExchangeManager") as mock_exchange_mgr,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.DataFetcher") as mock_data_fetcher,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.VotingAnalyzer") as mock_voting_class,
        ):
            mock_analyzer = Mock()
            mock_analyzer.selected_timeframe = TEST_TIMEFRAME
            mock_analyzer.atc_analyzer = Mock()
            mock_analyzer.atc_analyzer.selected_timeframe = TEST_TIMEFRAME
            mock_analyzer.run_atc_scan.return_value = True
            mock_analyzer.long_signals_final = pd.DataFrame()
            mock_analyzer.short_signals_final = pd.DataFrame()
            mock_analyzer.calculate_and_vote = Mock()
            mock_voting_class.return_value = mock_analyzer

            result = pre_filter_symbols_with_voting(sample_symbols, TEST_TIMEFRAME, TEST_LIMIT)

            # Should return all symbols when no signals found
            assert result == sample_symbols

    def test_pre_filter_voting_error_handling(self, sample_symbols):
        """Test error handling in pre-filter."""
        with (
            patch("modules.gemini_chart_analyzer.cli.pre_filter.ExchangeManager") as mock_exchange_mgr,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.DataFetcher") as mock_data_fetcher,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.VotingAnalyzer") as mock_voting_class,
        ):
            mock_voting_class.side_effect = Exception("Test error")

            result = pre_filter_symbols_with_voting(sample_symbols, TEST_TIMEFRAME, TEST_LIMIT)

            # Should return all symbols on error
            assert result == sample_symbols

    def test_pre_filter_voting_sorted_by_score(self, sample_symbols, mock_voting_analyzer):
        """Test that results are sorted by weighted_score descending."""
        with (
            patch("modules.gemini_chart_analyzer.cli.pre_filter.ExchangeManager") as mock_exchange_mgr,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.DataFetcher") as mock_data_fetcher,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.VotingAnalyzer") as mock_voting_class,
        ):
            mock_voting_class.return_value = mock_voting_analyzer
            mock_voting_analyzer.calculate_and_vote = Mock()

            result = pre_filter_symbols_with_voting(sample_symbols, TEST_TIMEFRAME, TEST_LIMIT)

            # Should be sorted by weighted_score (BTC=0.9, ETH=0.8, BNB=0.7)
            assert result[0] == "BTC/USDT"
            assert result[1] == "ETH/USDT"
            assert result[2] == "BNB/USDT"


class TestPreFilterWithHybrid:
    """Test pre_filter_symbols_with_hybrid function."""

    def test_pre_filter_hybrid_empty_symbols(self):
        """Test with empty symbol list."""
        result = pre_filter_symbols_with_hybrid([], TEST_TIMEFRAME, TEST_LIMIT)
        assert result == []

    def test_pre_filter_hybrid_no_atc_signals(self, sample_symbols):
        """Test when ATC scan returns no signals."""
        with (
            patch("modules.gemini_chart_analyzer.cli.pre_filter.ExchangeManager") as mock_exchange_mgr,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.DataFetcher") as mock_data_fetcher,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.HybridAnalyzer") as mock_hybrid_class,
        ):
            mock_analyzer = Mock()
            mock_analyzer.selected_timeframe = TEST_TIMEFRAME
            mock_analyzer.atc_analyzer = Mock()
            mock_analyzer.atc_analyzer.selected_timeframe = TEST_TIMEFRAME
            mock_analyzer.run_atc_scan.return_value = False  # No ATC signals
            mock_hybrid_class.return_value = mock_analyzer

            result = pre_filter_symbols_with_hybrid(sample_symbols, TEST_TIMEFRAME, TEST_LIMIT)

            # Should return all symbols when no ATC signals
            assert result == sample_symbols

    def test_pre_filter_hybrid_success(self, sample_symbols, mock_hybrid_analyzer):
        """Test successful pre-filtering with hybrid mode."""
        with (
            patch("modules.gemini_chart_analyzer.cli.pre_filter.ExchangeManager") as mock_exchange_mgr,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.DataFetcher") as mock_data_fetcher,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.HybridAnalyzer") as mock_hybrid_class,
        ):
            mock_hybrid_class.return_value = mock_hybrid_analyzer
            mock_hybrid_analyzer.filter_by_oscillator = Mock()
            mock_hybrid_analyzer.calculate_spc_signals_for_all = Mock()
            mock_hybrid_analyzer.filter_by_decision_matrix = Mock()

            result = pre_filter_symbols_with_hybrid(sample_symbols, TEST_TIMEFRAME, TEST_LIMIT)

            # Should return symbols with signals (BTC, ETH, BNB)
            assert len(result) == 3
            assert "BTC/USDT" in result
            assert "ETH/USDT" in result
            assert "BNB/USDT" in result
            assert "SOL/USDT" not in result
            assert "ADA/USDT" not in result

            # Verify HybridAnalyzer workflow was called
            mock_hybrid_class.assert_called_once()
            mock_hybrid_analyzer.run_atc_scan.assert_called_once()
            mock_hybrid_analyzer.filter_by_oscillator.assert_called_once()
            mock_hybrid_analyzer.calculate_spc_signals_for_all.assert_called_once()
            mock_hybrid_analyzer.filter_by_decision_matrix.assert_called_once()

    def test_pre_filter_hybrid_no_signals_found(self, sample_symbols):
        """Test when no signals are found after hybrid filtering."""
        with (
            patch("modules.gemini_chart_analyzer.cli.pre_filter.ExchangeManager") as mock_exchange_mgr,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.DataFetcher") as mock_data_fetcher,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.HybridAnalyzer") as mock_hybrid_class,
        ):
            mock_analyzer = Mock()
            mock_analyzer.selected_timeframe = TEST_TIMEFRAME
            mock_analyzer.atc_analyzer = Mock()
            mock_analyzer.atc_analyzer.selected_timeframe = TEST_TIMEFRAME
            mock_analyzer.run_atc_scan.return_value = True
            mock_analyzer.long_signals_confirmed = pd.DataFrame()
            mock_analyzer.short_signals_confirmed = pd.DataFrame()
            mock_analyzer.filter_by_oscillator = Mock()
            mock_analyzer.calculate_spc_signals_for_all = Mock()
            mock_analyzer.filter_by_decision_matrix = Mock()
            mock_hybrid_class.return_value = mock_analyzer

            result = pre_filter_symbols_with_hybrid(sample_symbols, TEST_TIMEFRAME, TEST_LIMIT)

            # Should return all symbols when no signals found
            assert result == sample_symbols

    def test_pre_filter_hybrid_error_handling(self, sample_symbols):
        """Test error handling in pre-filter."""
        with (
            patch("modules.gemini_chart_analyzer.cli.pre_filter.ExchangeManager") as mock_exchange_mgr,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.DataFetcher") as mock_data_fetcher,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.HybridAnalyzer") as mock_hybrid_class,
        ):
            mock_hybrid_class.side_effect = Exception("Test error")

            result = pre_filter_symbols_with_hybrid(sample_symbols, TEST_TIMEFRAME, TEST_LIMIT)

            # Should return all symbols on error
            assert result == sample_symbols

    def test_pre_filter_hybrid_no_duplicates(self, sample_symbols):
        """Test that duplicate symbols are not included."""
        with (
            patch("modules.gemini_chart_analyzer.cli.pre_filter.ExchangeManager") as mock_exchange_mgr,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.DataFetcher") as mock_data_fetcher,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.HybridAnalyzer") as mock_hybrid_class,
        ):
            mock_analyzer = Mock()
            mock_analyzer.selected_timeframe = TEST_TIMEFRAME
            mock_analyzer.atc_analyzer = Mock()
            mock_analyzer.atc_analyzer.selected_timeframe = TEST_TIMEFRAME
            mock_analyzer.run_atc_scan.return_value = True
            # BTC appears in both LONG and SHORT
            mock_analyzer.long_signals_confirmed = pd.DataFrame({"symbol": ["BTC/USDT", "ETH/USDT"]})
            mock_analyzer.short_signals_confirmed = pd.DataFrame(
                {
                    "symbol": ["BTC/USDT", "BNB/USDT"]  # BTC is duplicate
                }
            )
            mock_analyzer.filter_by_oscillator = Mock()
            mock_analyzer.calculate_spc_signals_for_all = Mock()
            mock_analyzer.filter_by_decision_matrix = Mock()
            mock_hybrid_class.return_value = mock_analyzer

            result = pre_filter_symbols_with_hybrid(sample_symbols, TEST_TIMEFRAME, TEST_LIMIT)

            # Should have 3 unique symbols (BTC, ETH, BNB)
            assert len(result) == 3
            assert result.count("BTC/USDT") == 1  # No duplicates
            assert "ETH/USDT" in result
            assert "BNB/USDT" in result

    def test_pre_filter_hybrid_spc_enabled(self, sample_symbols, mock_hybrid_analyzer):
        """Test hybrid mode with SPC enabled (default behavior)."""
        with (
            patch("modules.gemini_chart_analyzer.cli.pre_filter.ExchangeManager") as mock_exchange_mgr,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.DataFetcher") as mock_data_fetcher,
            patch("modules.gemini_chart_analyzer.cli.pre_filter.HybridAnalyzer") as mock_hybrid_class,
        ):
            mock_hybrid_class.return_value = mock_hybrid_analyzer
            mock_hybrid_analyzer.filter_by_oscillator = Mock()
            mock_hybrid_analyzer.calculate_spc_signals_for_all = Mock()
            mock_hybrid_analyzer.filter_by_decision_matrix = Mock()

            result = pre_filter_symbols_with_hybrid(sample_symbols, TEST_TIMEFRAME, TEST_LIMIT)

            # SPC should be called when enabled (default)
            mock_hybrid_analyzer.calculate_spc_signals_for_all.assert_called_once()
            # Other steps should also be called
            mock_hybrid_analyzer.filter_by_oscillator.assert_called_once()
            mock_hybrid_analyzer.filter_by_decision_matrix.assert_called_once()


class TestPreFilterStdinHandling:
    """Test stdin handling on Windows."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_pre_filter_voting_stdin_restoration_windows(self, sample_symbols, mock_voting_analyzer):
        """Test stdin restoration on Windows."""
        original_stdin = sys.stdin

        try:
            with (
                patch("modules.gemini_chart_analyzer.cli.pre_filter.ExchangeManager") as mock_exchange_mgr,
                patch("modules.gemini_chart_analyzer.cli.pre_filter.DataFetcher") as mock_data_fetcher,
                patch("modules.gemini_chart_analyzer.cli.pre_filter.VotingAnalyzer") as mock_voting_class,
            ):
                mock_voting_class.return_value = mock_voting_analyzer
                mock_voting_analyzer.calculate_and_vote = Mock()

                # Simulate stdin being closed
                sys.stdin = Mock()
                sys.stdin.closed = True

                result = pre_filter_symbols_with_voting(sample_symbols, "1h", 500)

                # Should still work and restore stdin
                assert result is not None
        finally:
            # Restore original stdin
            sys.stdin = original_stdin
