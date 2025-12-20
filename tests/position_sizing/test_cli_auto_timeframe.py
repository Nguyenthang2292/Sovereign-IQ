"""
Tests for position_sizing CLI auto timeframe feature.

Tests cover:
- try_timeframes_auto function
- Auto timeframe testing logic
- Finding valid timeframe at different positions
- Error handling during timeframe testing
- Edge cases (empty results, no valid results)
"""

import sys
from pathlib import Path

# Add project root to Python path for imports
ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
from types import SimpleNamespace

from modules.position_sizing.cli.main import try_timeframes_auto
from modules.position_sizing.core.position_sizer import PositionSizer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_data_fetcher():
    """Create a mock DataFetcher."""
    fetcher = SimpleNamespace()
    fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock()
    fetcher.fetch_binance_account_balance = MagicMock(return_value=None)
    return fetcher


@pytest.fixture
def sample_symbols():
    """Create sample symbols list."""
    return [
        {'symbol': 'BTC/USDT', 'signal': 1},
        {'symbol': 'ETH/USDT', 'signal': 1},
    ]


@pytest.fixture
def valid_results_df():
    """Create a valid results DataFrame with position sizing."""
    return pd.DataFrame({
        'symbol': ['BTC/USDT', 'ETH/USDT'],
        'position_size_usdt': [1000.0, 500.0],
        'position_size_pct': [10.0, 5.0],
        'kelly_fraction': [0.1, 0.05],
    })


@pytest.fixture
def empty_results_df():
    """Create an empty results DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def zero_results_df():
    """Create a results DataFrame with zero position sizes."""
    return pd.DataFrame({
        'symbol': ['BTC/USDT', 'ETH/USDT'],
        'position_size_usdt': [0.0, 0.0],
        'position_size_pct': [0.0, 0.0],
        'kelly_fraction': [0.0, 0.0],
    })


@pytest.fixture
def results_df_no_position_size_column():
    """Create a results DataFrame without position_size_usdt column."""
    return pd.DataFrame({
        'symbol': ['BTC/USDT', 'ETH/USDT'],
        'kelly_fraction': [0.1, 0.05],
    })


# ============================================================================
# Tests for try_timeframes_auto
# ============================================================================

class TestTryTimeframesAuto:
    """Test suite for try_timeframes_auto function"""
    
    def test_finds_valid_timeframe_at_first_attempt(self, mock_data_fetcher, sample_symbols, valid_results_df):
        """Test that function finds valid timeframe at first attempt (15m)"""
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_position_sizer_class:
            mock_position_sizer = MagicMock()
            mock_position_sizer.calculate_portfolio_allocation.return_value = valid_results_df
            mock_position_sizer_class.return_value = mock_position_sizer
            
            timeframe, results = try_timeframes_auto(
                symbols=sample_symbols,
                account_balance=10000.0,
                lookback_days=30,
                data_fetcher=mock_data_fetcher,
                max_position_size=0.2,
                signal_mode='single_signal',
                signal_calculation_mode='precomputed',
            )
            
            assert timeframe == '15m'
            assert results is not None
            assert isinstance(results, pd.DataFrame)
            assert not results.empty
            
            # Verify PositionSizer was created only once (stopped at first timeframe)
            assert mock_position_sizer_class.call_count == 1
            mock_position_sizer_class.assert_called_once_with(
                data_fetcher=mock_data_fetcher,
                timeframe='15m',
                lookback_days=30,
                max_position_size=0.2,
                signal_mode='single_signal',
                signal_calculation_mode='precomputed',
            )
    
    def test_finds_valid_timeframe_at_second_attempt(self, mock_data_fetcher, sample_symbols, valid_results_df, zero_results_df):
        """Test that function finds valid timeframe at second attempt (30m)"""
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_position_sizer_class:
            mock_position_sizer = MagicMock()
            # First timeframe returns zero results, second returns valid results
            mock_position_sizer.calculate_portfolio_allocation.side_effect = [
                zero_results_df,  # 15m: no valid results
                valid_results_df,  # 30m: valid results
            ]
            mock_position_sizer_class.return_value = mock_position_sizer
            
            timeframe, results = try_timeframes_auto(
                symbols=sample_symbols,
                account_balance=10000.0,
                lookback_days=30,
                data_fetcher=mock_data_fetcher,
                max_position_size=0.2,
                signal_mode='single_signal',
                signal_calculation_mode='precomputed',
            )
            
            assert timeframe == '30m'
            assert results is not None
            assert isinstance(results, pd.DataFrame)
            assert not results.empty
            
            # Verify PositionSizer was created twice (tried 15m, then 30m)
            assert mock_position_sizer_class.call_count == 2
            assert mock_position_sizer.calculate_portfolio_allocation.call_count == 2
    
    def test_finds_valid_timeframe_at_third_attempt(self, mock_data_fetcher, sample_symbols, valid_results_df, zero_results_df):
        """Test that function finds valid timeframe at third attempt (1h)"""
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_position_sizer_class:
            mock_position_sizer = MagicMock()
            # First two timeframes return zero results, third returns valid results
            mock_position_sizer.calculate_portfolio_allocation.side_effect = [
                zero_results_df,  # 15m: no valid results
                zero_results_df,  # 30m: no valid results
                valid_results_df,  # 1h: valid results
            ]
            mock_position_sizer_class.return_value = mock_position_sizer
            
            timeframe, results = try_timeframes_auto(
                symbols=sample_symbols,
                account_balance=10000.0,
                lookback_days=30,
                data_fetcher=mock_data_fetcher,
                max_position_size=0.2,
                signal_mode='single_signal',
                signal_calculation_mode='precomputed',
            )
            
            assert timeframe == '1h'
            assert results is not None
            assert isinstance(results, pd.DataFrame)
            assert not results.empty
            
            # Verify PositionSizer was created three times (tried all timeframes)
            assert mock_position_sizer_class.call_count == 3
            assert mock_position_sizer.calculate_portfolio_allocation.call_count == 3
    
    def test_no_valid_timeframe_found(self, mock_data_fetcher, sample_symbols, zero_results_df):
        """Test that function returns (None, None) when no valid timeframe is found"""
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_position_sizer_class:
            mock_position_sizer = MagicMock()
            # All timeframes return zero results
            mock_position_sizer.calculate_portfolio_allocation.return_value = zero_results_df
            mock_position_sizer_class.return_value = mock_position_sizer
            
            timeframe, results = try_timeframes_auto(
                symbols=sample_symbols,
                account_balance=10000.0,
                lookback_days=30,
                data_fetcher=mock_data_fetcher,
                max_position_size=0.2,
                signal_mode='single_signal',
                signal_calculation_mode='precomputed',
            )
            
            assert timeframe is None
            assert results is None
            
            # Verify PositionSizer was created for all timeframes
            assert mock_position_sizer_class.call_count == 3
            assert mock_position_sizer.calculate_portfolio_allocation.call_count == 3
    
    def test_empty_dataframe_returns_none(self, mock_data_fetcher, sample_symbols, empty_results_df):
        """Test that function handles empty DataFrame correctly"""
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_position_sizer_class:
            mock_position_sizer = MagicMock()
            mock_position_sizer.calculate_portfolio_allocation.return_value = empty_results_df
            mock_position_sizer_class.return_value = mock_position_sizer
            
            timeframe, results = try_timeframes_auto(
                symbols=sample_symbols,
                account_balance=10000.0,
                lookback_days=30,
                data_fetcher=mock_data_fetcher,
                max_position_size=0.2,
                signal_mode='single_signal',
                signal_calculation_mode='precomputed',
            )
            
            assert timeframe is None
            assert results is None
            
            # Verify all timeframes were tried
            assert mock_position_sizer_class.call_count == 3
    
    def test_missing_position_size_column(self, mock_data_fetcher, sample_symbols, results_df_no_position_size_column):
        """Test that function handles missing position_size_usdt column"""
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_position_sizer_class:
            mock_position_sizer = MagicMock()
            mock_position_sizer.calculate_portfolio_allocation.return_value = results_df_no_position_size_column
            mock_position_sizer_class.return_value = mock_position_sizer
            
            timeframe, results = try_timeframes_auto(
                symbols=sample_symbols,
                account_balance=10000.0,
                lookback_days=30,
                data_fetcher=mock_data_fetcher,
                max_position_size=0.2,
                signal_mode='single_signal',
                signal_calculation_mode='precomputed',
            )
            
            assert timeframe is None
            assert results is None
    
    def test_exception_at_first_timeframe_continues(self, mock_data_fetcher, sample_symbols, valid_results_df):
        """Test that function continues to next timeframe when exception occurs"""
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_position_sizer_class:
            mock_position_sizer = MagicMock()
            # First timeframe raises exception, second returns valid results
            mock_position_sizer.calculate_portfolio_allocation.side_effect = [
                ValueError("Test error"),  # 15m: exception
                valid_results_df,  # 30m: valid results
            ]
            mock_position_sizer_class.return_value = mock_position_sizer
            
            timeframe, results = try_timeframes_auto(
                symbols=sample_symbols,
                account_balance=10000.0,
                lookback_days=30,
                data_fetcher=mock_data_fetcher,
                max_position_size=0.2,
                signal_mode='single_signal',
                signal_calculation_mode='precomputed',
            )
            
            assert timeframe == '30m'
            assert results is not None
            assert isinstance(results, pd.DataFrame)
            
            # Verify PositionSizer was created twice (tried 15m with error, then 30m)
            assert mock_position_sizer_class.call_count == 2
    
    def test_exception_at_all_timeframes_returns_none(self, mock_data_fetcher, sample_symbols):
        """Test that function returns (None, None) when all timeframes raise exceptions"""
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_position_sizer_class:
            mock_position_sizer = MagicMock()
            # All timeframes raise exceptions
            mock_position_sizer.calculate_portfolio_allocation.side_effect = ValueError("Test error")
            mock_position_sizer_class.return_value = mock_position_sizer
            
            timeframe, results = try_timeframes_auto(
                symbols=sample_symbols,
                account_balance=10000.0,
                lookback_days=30,
                data_fetcher=mock_data_fetcher,
                max_position_size=0.2,
                signal_mode='single_signal',
                signal_calculation_mode='precomputed',
            )
            
            assert timeframe is None
            assert results is None
            
            # Verify PositionSizer was created for all timeframes
            assert mock_position_sizer_class.call_count == 3
    
    def test_valid_result_with_only_position_size_usdt(self, mock_data_fetcher, sample_symbols):
        """Test that function accepts results with only position_size_usdt > 0"""
        results_df = pd.DataFrame({
            'symbol': ['BTC/USDT'],
            'position_size_usdt': [1000.0],  # > 0
            # No position_size_pct column
        })
        
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_position_sizer_class:
            mock_position_sizer = MagicMock()
            mock_position_sizer.calculate_portfolio_allocation.return_value = results_df
            mock_position_sizer_class.return_value = mock_position_sizer
            
            timeframe, results = try_timeframes_auto(
                symbols=sample_symbols,
                account_balance=10000.0,
                lookback_days=30,
                data_fetcher=mock_data_fetcher,
                max_position_size=0.2,
                signal_mode='single_signal',
                signal_calculation_mode='precomputed',
            )
            
            assert timeframe == '15m'
            assert results is not None
    
    def test_valid_result_with_only_position_size_pct(self, mock_data_fetcher, sample_symbols):
        """Test that function accepts results with only position_size_pct > 0"""
        results_df = pd.DataFrame({
            'symbol': ['BTC/USDT'],
            'position_size_usdt': [0.0],  # = 0
            'position_size_pct': [10.0],  # > 0
        })
        
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_position_sizer_class:
            mock_position_sizer = MagicMock()
            mock_position_sizer.calculate_portfolio_allocation.return_value = results_df
            mock_position_sizer_class.return_value = mock_position_sizer
            
            timeframe, results = try_timeframes_auto(
                symbols=sample_symbols,
                account_balance=10000.0,
                lookback_days=30,
                data_fetcher=mock_data_fetcher,
                max_position_size=0.2,
                signal_mode='single_signal',
                signal_calculation_mode='precomputed',
            )
            
            assert timeframe == '15m'
            assert results is not None
    
    def test_timeframe_order(self, mock_data_fetcher, sample_symbols, valid_results_df):
        """Test that timeframes are tried in correct order: 15m -> 30m -> 1h"""
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_position_sizer_class:
            mock_position_sizer = MagicMock()
            # Return valid results only at 30m
            mock_position_sizer.calculate_portfolio_allocation.side_effect = [
                pd.DataFrame({'symbol': ['BTC/USDT'], 'position_size_usdt': [0.0]}),  # 15m: zero
                valid_results_df,  # 30m: valid
            ]
            mock_position_sizer_class.return_value = mock_position_sizer
            
            timeframe, results = try_timeframes_auto(
                symbols=sample_symbols,
                account_balance=10000.0,
                lookback_days=30,
                data_fetcher=mock_data_fetcher,
                max_position_size=0.2,
                signal_mode='single_signal',
                signal_calculation_mode='precomputed',
            )
            
            assert timeframe == '30m'
            
            # Verify PositionSizer was called with correct timeframes in order
            calls = mock_position_sizer_class.call_args_list
            assert len(calls) == 2
            assert calls[0].kwargs['timeframe'] == '15m'
            assert calls[1].kwargs['timeframe'] == '30m'
    
    def test_calculate_portfolio_allocation_called_with_correct_params(self, mock_data_fetcher, sample_symbols, valid_results_df):
        """Test that calculate_portfolio_allocation is called with correct parameters"""
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_position_sizer_class:
            mock_position_sizer = MagicMock()
            mock_position_sizer.calculate_portfolio_allocation.return_value = valid_results_df
            mock_position_sizer_class.return_value = mock_position_sizer
            
            timeframe, results = try_timeframes_auto(
                symbols=sample_symbols,
                account_balance=10000.0,
                lookback_days=30,
                data_fetcher=mock_data_fetcher,
                max_position_size=0.2,
                signal_mode='single_signal',
                signal_calculation_mode='precomputed',
            )
            
            # Verify calculate_portfolio_allocation was called with correct params
            mock_position_sizer.calculate_portfolio_allocation.assert_called_once_with(
                symbols=sample_symbols,
                account_balance=10000.0,
                timeframe='15m',
                lookback=30,
            )


# ============================================================================
# Integration Tests (with mocked logging)
# ============================================================================

class TestTryTimeframesAutoLogging:
    """Test suite for logging behavior in try_timeframes_auto"""
    
    def test_logs_progress_for_each_timeframe(self, mock_data_fetcher, sample_symbols, zero_results_df, valid_results_df):
        """Test that function logs progress for each timeframe attempt"""
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_position_sizer_class, \
             patch('modules.position_sizing.cli.main.log_progress') as mock_log_progress, \
             patch('modules.position_sizing.cli.main.log_warn') as mock_log_warn, \
             patch('modules.position_sizing.cli.main.log_success') as mock_log_success:
            
            mock_position_sizer = MagicMock()
            mock_position_sizer.calculate_portfolio_allocation.side_effect = [
                zero_results_df,  # 15m: no results
                valid_results_df,  # 30m: valid results
            ]
            mock_position_sizer_class.return_value = mock_position_sizer
            
            timeframe, results = try_timeframes_auto(
                symbols=sample_symbols,
                account_balance=10000.0,
                lookback_days=30,
                data_fetcher=mock_data_fetcher,
                max_position_size=0.2,
                signal_mode='single_signal',
                signal_calculation_mode='precomputed',
            )
            
            # Verify logging calls
            assert mock_log_progress.call_count >= 2  # At least 2 progress logs
            assert mock_log_warn.call_count >= 1  # At least 1 warn for 15m
            assert mock_log_success.call_count == 1  # 1 success for 30m
    
    def test_logs_error_when_exception_occurs(self, mock_data_fetcher, sample_symbols, valid_results_df):
        """Test that function logs error when exception occurs"""
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_position_sizer_class, \
             patch('modules.position_sizing.cli.main.log_warn') as mock_log_warn:
            
            mock_position_sizer = MagicMock()
            mock_position_sizer.calculate_portfolio_allocation.side_effect = [
                ValueError("Test error"),  # 15m: exception
                valid_results_df,  # 30m: valid results
            ]
            mock_position_sizer_class.return_value = mock_position_sizer
            
            timeframe, results = try_timeframes_auto(
                symbols=sample_symbols,
                account_balance=10000.0,
                lookback_days=30,
                data_fetcher=mock_data_fetcher,
                max_position_size=0.2,
                signal_mode='single_signal',
                signal_calculation_mode='precomputed',
            )
            
            # Verify error was logged
            assert mock_log_warn.call_count >= 1
            # Check that error message contains timeframe info
            warn_calls = [str(call) for call in mock_log_warn.call_args_list]
            assert any('15m' in str(call) or 'Error' in str(call) for call in warn_calls)

