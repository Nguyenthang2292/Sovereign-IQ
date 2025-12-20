"""
Edge case tests for Full Backtester.
"""

import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch
from modules.backtester import FullBacktester

# Fixtures from conftest.py will be automatically available


def test_backtest_with_no_signals(mock_data_fetcher):
    """Test backtest when no signals are generated."""
    # Override the autouse fixture to return no signals
    with patch('core.signal_calculators.get_range_oscillator_signal', return_value=(0, 0.0)), \
         patch('core.signal_calculators.get_spc_signal', return_value=(0, 0.0)), \
         patch('core.signal_calculators.get_xgboost_signal', return_value=(0, 0.0)), \
         patch('core.signal_calculators.get_hmm_signal', return_value=(0, 0.0)), \
         patch('core.signal_calculators.get_random_forest_signal', return_value=(0, 0.0)):
        
        backtester = FullBacktester(mock_data_fetcher)
        
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=200,
            signal_type="LONG",
        )
        
        # Should return valid structure even with no trades
        assert 'trades' in result
        assert 'metrics' in result
        assert isinstance(result['trades'], list)


def test_backtest_with_very_small_dataset(mock_data_fetcher):
    """Test backtest with very small dataset."""
    def small_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=10, freq="h")
        prices = [100 + i * 0.1 for i in range(10)]
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
        }, index=dates)
        return df, "binance"
    
    small_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=small_fetch,
    )
    
    # Mock signal calculators to avoid API calls
    with patch('core.signal_calculators.get_range_oscillator_signal', return_value=(1, 0.7)), \
         patch('core.signal_calculators.get_spc_signal', return_value=(1, 0.6)), \
         patch('core.signal_calculators.get_xgboost_signal', return_value=(1, 0.8)), \
         patch('core.signal_calculators.get_hmm_signal', return_value=(1, 0.65)), \
         patch('core.signal_calculators.get_random_forest_signal', return_value=(1, 0.75)):
        
        backtester = FullBacktester(small_fetcher)
        
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=10,
            signal_type="LONG",
        )
        
        # Should handle small dataset gracefully
        assert 'trades' in result
        assert 'metrics' in result


def test_backtest_with_extreme_price_movements(mock_data_fetcher):
    """Test backtest with extreme price movements."""
    def extreme_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=200, freq="h")
        # Create extreme volatility
        prices = 100 + np.cumsum(np.random.randn(200) * 5.0)  # High volatility
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.05,  # 5% range
            'low': prices * 0.95,
            'close': prices,
        }, index=dates)
        return df, "binance"
    
    extreme_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=extreme_fetch,
    )
    
    # Mock signal calculators to avoid API calls
    with patch('core.signal_calculators.get_range_oscillator_signal', return_value=(1, 0.7)), \
         patch('core.signal_calculators.get_spc_signal', return_value=(1, 0.6)), \
         patch('core.signal_calculators.get_xgboost_signal', return_value=(1, 0.8)), \
         patch('core.signal_calculators.get_hmm_signal', return_value=(1, 0.65)), \
         patch('core.signal_calculators.get_random_forest_signal', return_value=(1, 0.75)):
        
        backtester = FullBacktester(extreme_fetcher)
        
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=200,
            signal_type="LONG",
        )
        
        # Should handle extreme movements
        assert 'trades' in result
        assert 'metrics' in result
        
        # Metrics should still be valid
        metrics = result['metrics']
        assert 0.0 <= metrics['win_rate'] <= 1.0
        assert metrics['num_trades'] >= 0


def test_backtest_with_constant_price(mock_data_fetcher):
    """Test backtest with constant price (no movement)."""
    def constant_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=200, freq="h")
        df = pd.DataFrame({
            'open': [100.0] * 200,
            'high': [100.0] * 200,
            'low': [100.0] * 200,
            'close': [100.0] * 200,
        }, index=dates)
        return df, "binance"
    
    constant_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=constant_fetch,
    )
    
    # Mock signal calculators to avoid API calls
    with patch('core.signal_calculators.get_range_oscillator_signal', return_value=(0, 0.0)), \
         patch('core.signal_calculators.get_spc_signal', return_value=(0, 0.0)), \
         patch('core.signal_calculators.get_xgboost_signal', return_value=(0, 0.0)), \
         patch('core.signal_calculators.get_hmm_signal', return_value=(0, 0.0)), \
         patch('core.signal_calculators.get_random_forest_signal', return_value=(0, 0.0)):
        
        backtester = FullBacktester(constant_fetcher)
        
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=200,
            signal_type="LONG",
        )
        
        # Should handle constant price
        assert 'trades' in result
        assert 'metrics' in result


def test_equity_curve_calculation(mock_data_fetcher):
    """Test equity curve calculation."""
    backtester = FullBacktester(mock_data_fetcher)
    
    # Create mock trades
    trades = [
        {
            'entry_index': 0,
            'exit_index': 10,
            'entry_price': 100.0,
            'exit_price': 105.0,
            'pnl': 0.05,
            'pnl_pct': 5.0,
            'hold_periods': 10,
        },
        {
            'entry_index': 20,
            'exit_index': 30,
            'entry_price': 105.0,
            'exit_price': 102.0,
            'pnl': -0.03,
            'pnl_pct': -3.0,
            'hold_periods': 10,
        },
    ]
    
    from modules.backtester.core.equity_curve import calculate_equity_curve
    
    equity_curve = calculate_equity_curve(
        trades=trades,
        initial_capital=10000.0,
        num_periods=100,
    )
    
    assert isinstance(equity_curve, pd.Series)
    assert len(equity_curve) == 100
    assert equity_curve.iloc[0] == 10000.0


def test_metrics_calculation_with_no_trades(mock_data_fetcher):
    """Test metrics calculation when there are no trades."""
    from modules.backtester.core.metrics import calculate_metrics
    
    metrics = calculate_metrics(
        trades=[],
        equity_curve=pd.Series([10000.0] * 100),
    )
    
    assert metrics['num_trades'] == 0
    assert metrics['win_rate'] == 0.0
    assert metrics['total_return'] == 0.0
    assert metrics['profit_factor'] == 0.0


def test_metrics_calculation_with_only_winning_trades(mock_data_fetcher):
    """Test metrics calculation with only winning trades."""
    from modules.backtester.core.metrics import calculate_metrics
    
    trades = [
        {'pnl': 0.05, 'entry_index': 0, 'exit_index': 10},
        {'pnl': 0.03, 'entry_index': 20, 'exit_index': 30},
        {'pnl': 0.02, 'entry_index': 40, 'exit_index': 50},
    ]
    
    equity_curve = pd.Series([10000.0 + i * 10 for i in range(100)])
    
    metrics = calculate_metrics(trades, equity_curve)
    
    assert metrics['win_rate'] == 1.0
    assert metrics['num_trades'] == 3
    assert metrics['avg_loss'] == 0.0  # No losses


def test_metrics_calculation_with_only_losing_trades(mock_data_fetcher):
    """Test metrics calculation with only losing trades."""
    from modules.backtester.core.metrics import calculate_metrics
    
    trades = [
        {'pnl': -0.05, 'entry_index': 0, 'exit_index': 10},
        {'pnl': -0.03, 'entry_index': 20, 'exit_index': 30},
    ]
    
    equity_curve = pd.Series([10000.0 - i * 10 for i in range(100)])
    
    metrics = calculate_metrics(trades, equity_curve)
    
    assert metrics['win_rate'] == 0.0
    assert metrics['num_trades'] == 2
    assert metrics['avg_win'] == 0.0  # No wins

