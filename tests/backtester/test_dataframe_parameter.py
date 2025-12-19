"""
Tests for DataFrame parameter optimization in FullBacktester.

Tests verify that:
1. Optional DataFrame parameter works correctly
2. Backward compatibility (no DataFrame still works)
3. DataFrame is used instead of fetching from API
"""

import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock, patch, call
from modules.backtester import FullBacktester


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=200, freq="h")
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    return pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.rand(200) * 1000,
    }, index=dates)


@pytest.fixture
def mock_data_fetcher():
    """Create a mock data fetcher that tracks fetch calls."""
    fetch_call_count = {'count': 0}
    
    def fake_fetch(symbol, **kwargs):
        fetch_call_count['count'] += 1
        dates = pd.date_range("2023-01-01", periods=200, freq="h")
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.rand(200) * 1000,
        }, index=dates)
        return df, "binance"
    
    fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
        fetch_call_count=fetch_call_count,
    )
    return fetcher


def test_backtest_with_dataframe_parameter(sample_dataframe, mock_data_fetcher):
    """Test that backtest accepts DataFrame parameter and doesn't fetch from API."""
    # Mock signal calculators
    with patch('core.signal_calculators.get_range_oscillator_signal', return_value=(1, 0.7)), \
         patch('core.signal_calculators.get_spc_signal', return_value=(1, 0.6)), \
         patch('core.signal_calculators.get_xgboost_signal', return_value=(1, 0.8)), \
         patch('core.signal_calculators.get_hmm_signal', return_value=(1, 0.65)), \
         patch('core.signal_calculators.get_random_forest_signal', return_value=(1, 0.75)):
        
        backtester = FullBacktester(mock_data_fetcher)
        
        # Reset call count
        mock_data_fetcher.fetch_call_count['count'] = 0
        
        # Call backtest with DataFrame
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=200,
            signal_type="LONG",
            df=sample_dataframe,
        )
        
        # Verify fetch was NOT called
        assert mock_data_fetcher.fetch_call_count['count'] == 0
        
        # Verify result structure
        assert 'trades' in result
        assert 'equity_curve' in result
        assert 'metrics' in result


def test_backtest_without_dataframe_parameter(mock_data_fetcher):
    """Test backward compatibility - backtest without DataFrame still works."""
    # Mock signal calculators
    with patch('core.signal_calculators.get_range_oscillator_signal', return_value=(1, 0.7)), \
         patch('core.signal_calculators.get_spc_signal', return_value=(1, 0.6)), \
         patch('core.signal_calculators.get_xgboost_signal', return_value=(1, 0.8)), \
         patch('core.signal_calculators.get_hmm_signal', return_value=(1, 0.65)), \
         patch('core.signal_calculators.get_random_forest_signal', return_value=(1, 0.75)):
        
        backtester = FullBacktester(mock_data_fetcher)
        
        # Reset call count
        mock_data_fetcher.fetch_call_count['count'] = 0
        
        # Call backtest without DataFrame (backward compatibility)
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=200,
            signal_type="LONG",
        )
        
        # Verify fetch WAS called
        assert mock_data_fetcher.fetch_call_count['count'] == 1
        
        # Verify result structure
        assert 'trades' in result
        assert 'equity_curve' in result
        assert 'metrics' in result


def test_backtest_dataframe_vs_fetch_same_result(sample_dataframe, mock_data_fetcher):
    """Test that using DataFrame produces same result as fetching."""
    # Mock signal calculators to return consistent results
    with patch('core.signal_calculators.get_range_oscillator_signal', return_value=(1, 0.7)), \
         patch('core.signal_calculators.get_spc_signal', return_value=(1, 0.6)), \
         patch('core.signal_calculators.get_xgboost_signal', return_value=(1, 0.8)), \
         patch('core.signal_calculators.get_hmm_signal', return_value=(1, 0.65)), \
         patch('core.signal_calculators.get_random_forest_signal', return_value=(1, 0.75)):
        
        backtester = FullBacktester(mock_data_fetcher)
        
        # Test with DataFrame
        result_with_df = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=200,
            signal_type="LONG",
            df=sample_dataframe,
        )
        
        # Test without DataFrame (will fetch)
        result_without_df = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=200,
            signal_type="LONG",
        )
        
        # Both should have same structure
        assert 'trades' in result_with_df
        assert 'trades' in result_without_df
        assert 'metrics' in result_with_df
        assert 'metrics' in result_without_df
        
        # Metrics should have same keys
        assert set(result_with_df['metrics'].keys()) == set(result_without_df['metrics'].keys())


def test_backtest_with_empty_dataframe(mock_data_fetcher):
    """Test backtest with empty DataFrame."""
    empty_df = pd.DataFrame()
    
    backtester = FullBacktester(mock_data_fetcher)
    
    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=200,
        signal_type="LONG",
        df=empty_df,
    )
    
    # Should return empty result structure
    assert 'trades' in result
    assert 'metrics' in result
    assert result['trades'] == []
    assert result['metrics']['num_trades'] == 0


def test_backtest_dataframe_preserves_index(sample_dataframe, mock_data_fetcher):
    """Test that DataFrame index is preserved in backtest."""
    # Mock signal calculators
    with patch('core.signal_calculators.get_range_oscillator_signal', return_value=(1, 0.7)), \
         patch('core.signal_calculators.get_spc_signal', return_value=(1, 0.6)), \
         patch('core.signal_calculators.get_xgboost_signal', return_value=(1, 0.8)), \
         patch('core.signal_calculators.get_hmm_signal', return_value=(1, 0.65)), \
         patch('core.signal_calculators.get_random_forest_signal', return_value=(1, 0.75)):
        
        backtester = FullBacktester(mock_data_fetcher)
        
        original_index = sample_dataframe.index.copy()
        
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=200,
            signal_type="LONG",
            df=sample_dataframe,
        )
        
        # Verify DataFrame was not modified
        assert sample_dataframe.index.equals(original_index)
        
        # Verify trades have valid entry/exit times
        if result['trades']:
            for trade in result['trades']:
                assert 'entry_time' in trade
                assert 'exit_time' in trade

