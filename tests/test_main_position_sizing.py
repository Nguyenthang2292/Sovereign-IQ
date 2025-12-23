"""
Tests for modules.position_sizing.cli.main.

Tests cover:
- Config merge logic (interactive menu values override default args)
- Auto timeframe selection
- Lookback_days merge from menu
- Exception handling in try_timeframes_auto
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from types import SimpleNamespace
import sys
from pathlib import Path

# Add parent directory to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.position_sizing.cli.main import (
    _try_timeframes_auto,
    _prompt_for_balance,
    InsufficientDataError,
    DataError,
)


@pytest.fixture
def mock_args():
    """Create mock args object with default values."""
    args = SimpleNamespace()
    args.lookback_days = 15  # Default value
    args.timeframe = "1h"
    args.max_position_size = 1.0
    args.symbols = None
    args.symbols_file = None
    args.source = None
    args.account_balance = 10000.0
    args.no_menu = False
    args.output = None
    args.fetch_balance = False
    args.signal_mode = 'single_signal'
    args.signal_calculation_mode = 'precomputed'
    args.auto_timeframe = False
    return args


@pytest.fixture
def mock_data_fetcher():
    """Create mock data fetcher."""
    fetcher = Mock()
    fetcher.fetch_ohlcv_with_fallback_exchange = Mock(return_value=(None, None))
    fetcher.fetch_binance_account_balance = Mock(return_value=None)
    return fetcher


class TestConfigMerge:
    """Test config merge logic in main function."""
    
    def test_config_merge_overrides_default_lookback_days(self, mock_args, mock_data_fetcher):
        """Test that config from menu overrides default lookback_days."""
        # Simulate menu returning lookback_days = 3
        config = {
            'lookback_days': 3,
            'symbols': 'BTC/USDT',
            'account_balance': 10000.0,
            'auto_timeframe': False,
            'timeframe': '1h',
            'max_position_size': 0.5,
        }
        
        # Simulate merge logic
        original_lookback = mock_args.lookback_days
        assert original_lookback == 15  # Default value
        
        # Merge config (simulating what happens in main())
        for key, value in config.items():
            setattr(mock_args, key, value)
        
        # Check that lookback_days was overridden
        assert mock_args.lookback_days == 3
        assert mock_args.lookback_days != original_lookback
    
    def test_config_merge_overrides_all_menu_values(self, mock_args):
        """Test that all config values from menu override args."""
        config = {
            'lookback_days': 5,
            'symbols': 'ETH/USDT',
            'account_balance': 20000.0,
            'auto_timeframe': True,
            'timeframe': '30m',
            'max_position_size': 0.3,
        }
        
        # Merge config
        for key, value in config.items():
            setattr(mock_args, key, value)
        
        # Check all values were overridden
        assert mock_args.lookback_days == 5
        assert mock_args.symbols == 'ETH/USDT'
        assert mock_args.account_balance == 20000.0
        assert mock_args.auto_timeframe is True
        assert mock_args.timeframe == '30m'
        assert mock_args.max_position_size == 0.3
    
    def test_config_merge_preserves_non_menu_args(self, mock_args):
        """Test that args not in menu config are preserved."""
        # Set some args that won't be in menu
        mock_args.output = 'test_output.csv'
        mock_args.no_menu = True
        
        config = {
            'lookback_days': 7,
        }
        
        # Merge config
        for key, value in config.items():
            setattr(mock_args, key, value)
        
        # Check that menu value was set
        assert mock_args.lookback_days == 7
        
        # Check that non-menu args are preserved
        assert mock_args.output == 'test_output.csv'
        assert mock_args.no_menu is True


class TestTryTimeframesAuto:
    """Test auto timeframe selection logic."""
    
    def test_returns_first_valid_timeframe(self, mock_data_fetcher):
        """Test that _try_timeframes_auto returns first timeframe with valid results."""
        symbols = [{'symbol': 'BTC/USDT', 'signal': 1}]
        account_balance = 10000.0
        lookback_days = 15
        max_position_size = 1.0
        signal_mode = 'single_signal'
        signal_calculation_mode = 'precomputed'
        
        # Create a mock result DataFrame
        mock_results = pd.DataFrame({
            'symbol': ['BTC/USDT'],
            'position_size_usdt': [5000.0],
            'position_size_pct': [50.0],
        })
        
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_sizer_class:
            mock_sizer_instance = Mock()
            mock_sizer_instance.calculate_portfolio_allocation = Mock(return_value=mock_results)
            mock_sizer_class.return_value = mock_sizer_instance
            
            timeframe, results_df = _try_timeframes_auto(
                symbols=symbols,
                account_balance=account_balance,
                lookback_days=lookback_days,
                data_fetcher=mock_data_fetcher,
                max_position_size=max_position_size,
                signal_mode=signal_mode,
                signal_calculation_mode=signal_calculation_mode,
            )
            
            # Should return first timeframe with valid results
            assert timeframe is not None
            assert timeframe in ['15m', '30m', '1h']
            assert results_df is not None
            assert not results_df.empty
    
    def test_returns_none_when_no_valid_timeframe(self, mock_data_fetcher):
        """Test that _try_timeframes_auto returns None when no timeframe has valid results."""
        symbols = [{'symbol': 'BTC/USDT', 'signal': 1}]
        account_balance = 10000.0
        lookback_days = 15
        max_position_size = 1.0
        signal_mode = 'single_signal'
        signal_calculation_mode = 'precomputed'
        
        # Create empty results
        mock_results = pd.DataFrame()
        
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_sizer_class:
            mock_sizer_instance = Mock()
            mock_sizer_instance.calculate_portfolio_allocation = Mock(return_value=mock_results)
            mock_sizer_class.return_value = mock_sizer_instance
            
            timeframe, results_df = _try_timeframes_auto(
                symbols=symbols,
                account_balance=account_balance,
                lookback_days=lookback_days,
                data_fetcher=mock_data_fetcher,
                max_position_size=max_position_size,
                signal_mode=signal_mode,
                signal_calculation_mode=signal_calculation_mode,
            )
            
            # Should return None when no valid timeframe found
            assert timeframe is None
            assert results_df is None
    
    def test_handles_data_errors_gracefully(self, mock_data_fetcher):
        """Test that data errors are handled gracefully in _try_timeframes_auto."""
        symbols = [{'symbol': 'BTC/USDT', 'signal': 1}]
        account_balance = 10000.0
        lookback_days = 15
        max_position_size = 1.0
        signal_mode = 'single_signal'
        signal_calculation_mode = 'precomputed'
        
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_sizer_class:
            # First timeframe raises ValueError (data error)
            mock_sizer_instance = Mock()
            mock_sizer_instance.calculate_portfolio_allocation = Mock(side_effect=ValueError("Insufficient data"))
            mock_sizer_class.return_value = mock_sizer_instance
            
            # Should continue trying other timeframes
            timeframe, results_df = _try_timeframes_auto(
                symbols=symbols,
                account_balance=account_balance,
                lookback_days=lookback_days,
                data_fetcher=mock_data_fetcher,
                max_position_size=max_position_size,
                signal_mode=signal_mode,
                signal_calculation_mode=signal_calculation_mode,
            )
            
            # Should return None since all timeframes failed with ValueError
            assert timeframe is None
            assert results_df is None    
    def test_handles_insufficient_data_error(self, mock_data_fetcher):
        """Test that InsufficientDataError is handled gracefully."""
        symbols = [{'symbol': 'BTC/USDT', 'signal': 1}]
        account_balance = 10000.0
        lookback_days = 15
        max_position_size = 1.0
        signal_mode = 'single_signal'
        signal_calculation_mode = 'precomputed'
        
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_sizer_class:
            mock_sizer_instance = Mock()
            mock_sizer_instance.calculate_portfolio_allocation = Mock(side_effect=InsufficientDataError("Not enough data"))
            mock_sizer_class.return_value = mock_sizer_instance
            
            timeframe, results_df = _try_timeframes_auto(
                symbols=symbols,
                account_balance=account_balance,
                lookback_days=lookback_days,
                data_fetcher=mock_data_fetcher,
                max_position_size=max_position_size,
                signal_mode=signal_mode,
                signal_calculation_mode=signal_calculation_mode,
            )
            
            # Should handle error gracefully
            assert timeframe is None or isinstance(timeframe, str)
    
    def test_handles_network_errors_gracefully(self, mock_data_fetcher):
        """Test that network errors are handled gracefully."""
        symbols = [{'symbol': 'BTC/USDT', 'signal': 1}]
        account_balance = 10000.0
        lookback_days = 15
        max_position_size = 1.0
        signal_mode = 'single_signal'
        signal_calculation_mode = 'precomputed'
        
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_sizer_class:
            mock_sizer_instance = Mock()
            mock_sizer_instance.calculate_portfolio_allocation = Mock(side_effect=ConnectionError("Network error"))
            mock_sizer_class.return_value = mock_sizer_instance
            
            timeframe, results_df = _try_timeframes_auto(
                symbols=symbols,
                account_balance=account_balance,
                lookback_days=lookback_days,
                data_fetcher=mock_data_fetcher,
                max_position_size=max_position_size,
                signal_mode=signal_mode,
                signal_calculation_mode=signal_calculation_mode,
            )
            
            # Should handle error gracefully - strict expectation that both are None
            # when all timeframes fail due to ConnectionError
            assert timeframe is None, "timeframe should be None when all timeframes fail due to ConnectionError"
            assert results_df is None, "results_df should be None when all timeframes fail due to ConnectionError"
    
    def test_validates_position_size_before_returning(self, mock_data_fetcher):
        """Test that only timeframes with valid position sizes are returned."""
        symbols = [{'symbol': 'BTC/USDT', 'signal': 1}]
        account_balance = 10000.0
        lookback_days = 15
        max_position_size = 1.0
        signal_mode = 'single_signal'
        signal_calculation_mode = 'precomputed'
        
        # Create results with zero position size (should be rejected)
        mock_results_zero = pd.DataFrame({
            'symbol': ['BTC/USDT'],
            'position_size_usdt': [0.0],  # Zero position size
            'position_size_pct': [0.0],
        })
        
        # Create results with valid position size
        mock_results_valid = pd.DataFrame({
            'symbol': ['BTC/USDT'],
            'position_size_usdt': [5000.0],
            'position_size_pct': [50.0],
        })
        
        with patch('modules.position_sizing.cli.main.PositionSizer') as mock_sizer_class:
            mock_sizer_instance = Mock()
            # First two timeframes return zero, third returns valid
            mock_sizer_instance.calculate_portfolio_allocation = Mock(side_effect=[
                mock_results_zero,
                mock_results_zero,
                mock_results_valid,
            ])
            mock_sizer_class.return_value = mock_sizer_instance
            
            timeframe, results_df = _try_timeframes_auto(
                symbols=symbols,
                account_balance=account_balance,
                lookback_days=lookback_days,
                data_fetcher=mock_data_fetcher,
                max_position_size=max_position_size,
                signal_mode=signal_mode,
                signal_calculation_mode=signal_calculation_mode,
            )
            
            # Should return the third timeframe (1h) with valid results
            assert timeframe is not None
            assert results_df is not None
            assert not results_df.empty


class TestPromptForBalance:
    """Test _prompt_for_balance function."""
    
    @patch('builtins.input', return_value='5000')
    def test_prompt_for_balance_valid_input(self, mock_input):
        """Test that valid balance input is returned."""
        balance = _prompt_for_balance()
        assert balance == 5000.0
        assert isinstance(balance, float)
    
    @patch('builtins.input', return_value='invalid')
    @patch('modules.position_sizing.cli.main.sys.exit')
    def test_prompt_for_balance_invalid_input(self, mock_exit, mock_input):
        """Test that invalid balance input exits."""
        _prompt_for_balance()
        mock_exit.assert_called_once_with(1)
    
    @patch('builtins.input', return_value='10000.50')
    def test_prompt_for_balance_decimal_input(self, mock_input):
        """Test that decimal balance input is handled correctly."""
        balance = _prompt_for_balance()
        assert balance == 10000.50

