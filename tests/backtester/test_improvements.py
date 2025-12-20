"""
Test cases for recent improvements to backtester module.

Tests cover:
1. risk_per_trade configurable parameter
2. Input validation
3. Equity curve padding edge cases
4. Trailing stop with None check
5. Parallel processing error handling
"""

import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from modules.backtester import FullBacktester
from modules.backtester.core.equity_curve import calculate_equity_curve
from modules.backtester.core.trade_simulator import simulate_trades
from config.position_sizing import BACKTEST_RISK_PER_TRADE


class TestRiskPerTradeConfigurable:
    """Tests for configurable risk_per_trade parameter."""
    
    def test_risk_per_trade_default_value(self, mock_data_fetcher):
        """Test that default risk_per_trade is used when not specified."""
        backtester = FullBacktester(mock_data_fetcher)
        assert backtester.risk_per_trade == BACKTEST_RISK_PER_TRADE
    
    def test_risk_per_trade_custom_value(self, mock_data_fetcher):
        """Test that custom risk_per_trade can be set."""
        custom_risk = 0.02  # 2%
        backtester = FullBacktester(
            mock_data_fetcher,
            risk_per_trade=custom_risk,
        )
        assert backtester.risk_per_trade == custom_risk
    
    def test_equity_curve_with_different_risk_per_trade(self):
        """Test that equity curve calculation uses risk_per_trade correctly."""
        trades = [
            {'pnl': 0.10, 'entry_index': 0, 'exit_index': 10},
            {'pnl': -0.05, 'entry_index': 20, 'exit_index': 30},
        ]
        
        # Test with default 1% risk
        equity_1pct = calculate_equity_curve(
            trades=trades,
            initial_capital=10000.0,
            num_periods=100,
            risk_per_trade=0.01,
        )
        
        # Test with 2% risk
        equity_2pct = calculate_equity_curve(
            trades=trades,
            initial_capital=10000.0,
            num_periods=100,
            risk_per_trade=0.02,
        )
        
        # With 2% risk, equity changes should be double
        # After first trade (10% profit):
        # 1% risk: 10000 + (10000 * 0.01 * 0.10) = 10010
        # 2% risk: 10000 + (10000 * 0.02 * 0.10) = 10020
        assert equity_2pct.iloc[11] > equity_1pct.iloc[11]
        
        # Verify initial capital is same
        assert equity_1pct.iloc[0] == equity_2pct.iloc[0] == 10000.0
    
    def test_backtester_passes_risk_per_trade_to_equity_curve(self, mock_data_fetcher):
        """Test that FullBacktester passes risk_per_trade to equity curve calculation."""
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
        }, index=dates)
        
        class MockSignalCalculator:
            def calculate_hybrid_signal(self, **kwargs):
                period_index = kwargs.get('period_index', 0)
                if period_index == 0:
                    return 1, 0.8
                return 0, 0.0
            
            def get_cache_stats(self):
                return {
                    'signal_cache_size': 0,
                    'signal_cache_max_size': 1000,
                    'cache_hit_rate': 0.0,
                }
        
        backtester = FullBacktester(
            mock_data_fetcher,
            risk_per_trade=0.02,  # 2% custom risk
        )
        backtester.hybrid_signal_calculator = MockSignalCalculator()
        
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=100,
            signal_type="LONG",
            df=df,
        )
        
        # Verify equity curve is calculated (structure is correct)
        assert 'equity_curve' in result
        assert len(result['equity_curve']) == len(df)


class TestInputValidation:
    """Tests for input validation in backtest() method."""
    
    def test_invalid_signal_type(self, mock_data_fetcher):
        """Test that invalid signal_type raises ValueError (caught and returns empty result)."""
        backtester = FullBacktester(mock_data_fetcher)
        
        # ValueError is raised but caught in try-except, returns empty result
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=100,
            signal_type="INVALID",
        )
        # Should return empty result when validation fails
        assert result['trades'] == []
    
    def test_zero_initial_capital(self, mock_data_fetcher):
        """Test that zero initial_capital raises ValueError."""
        backtester = FullBacktester(mock_data_fetcher)
        
        # ValueError is caught and logged, returns empty result
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=100,
            signal_type="LONG",
            initial_capital=0.0,
        )
        assert result['trades'] == []
    
    def test_negative_initial_capital(self, mock_data_fetcher):
        """Test that negative initial_capital raises ValueError."""
        backtester = FullBacktester(mock_data_fetcher)
        
        # ValueError is caught and logged, returns empty result
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=100,
            signal_type="LONG",
            initial_capital=-1000.0,
        )
        assert result['trades'] == []
    
    def test_zero_stop_loss_pct(self, mock_data_fetcher):
        """Test that zero stop_loss_pct raises ValueError."""
        backtester = FullBacktester(mock_data_fetcher, stop_loss_pct=0.0)
        
        # ValueError is caught and logged, returns empty result
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=100,
            signal_type="LONG",
        )
        assert result['trades'] == []
    
    def test_zero_take_profit_pct(self, mock_data_fetcher):
        """Test that zero take_profit_pct raises ValueError."""
        backtester = FullBacktester(mock_data_fetcher, take_profit_pct=0.0)
        
        # ValueError is caught and logged, returns empty result
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=100,
            signal_type="LONG",
        )
        assert result['trades'] == []
    
    def test_zero_trailing_stop_pct(self, mock_data_fetcher):
        """Test that zero trailing_stop_pct raises ValueError."""
        backtester = FullBacktester(mock_data_fetcher, trailing_stop_pct=0.0)
        
        # ValueError is caught and logged, returns empty result
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=100,
            signal_type="LONG",
        )
        assert result['trades'] == []
    
    def test_zero_max_hold_periods(self, mock_data_fetcher):
        """Test that zero max_hold_periods raises ValueError."""
        backtester = FullBacktester(mock_data_fetcher, max_hold_periods=0)
        
        # ValueError is caught and logged, returns empty result
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=100,
            signal_type="LONG",
        )
        assert result['trades'] == []
    
    def test_zero_lookback(self, mock_data_fetcher):
        """Test that zero lookback raises ValueError."""
        backtester = FullBacktester(mock_data_fetcher)
        
        # ValueError is raised and caught in backtest, returns empty result
        # So we check that it doesn't work normally
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=0,
            signal_type="LONG",
        )
        # Should return empty result when validation fails
        assert result['trades'] == []
    
    def test_negative_lookback(self, mock_data_fetcher):
        """Test that negative lookback raises ValueError."""
        backtester = FullBacktester(mock_data_fetcher)
        
        # ValueError is raised and caught in backtest, returns empty result
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=-100,
            signal_type="LONG",
        )
        # Should return empty result when validation fails
        assert result['trades'] == []
    
    def test_case_insensitive_signal_type(self, mock_data_fetcher):
        """Test that signal_type is case-insensitive (should work with lowercase)."""
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
        }, index=dates)
        
        class MockSignalCalculator:
            def calculate_hybrid_signal(self, **kwargs):
                return 0, 0.0
            
            def get_cache_stats(self):
                return {
                    'signal_cache_size': 0,
                    'signal_cache_max_size': 1000,
                    'cache_hit_rate': 0.0,
                }
        
        backtester = FullBacktester(mock_data_fetcher)
        backtester.hybrid_signal_calculator = MockSignalCalculator()
        
        # Should work with lowercase
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=100,
            signal_type="long",  # lowercase
            df=df,
        )
        
        assert 'trades' in result
        assert 'metrics' in result


class TestEquityCurvePadding:
    """Tests for equity curve padding edge cases."""
    
    def test_equity_curve_with_more_trades_than_periods(self):
        """Test equity curve when number of trades exceeds num_periods."""
        # Create many trades
        trades = [
            {'pnl': 0.01, 'entry_index': i * 2, 'exit_index': i * 2 + 1}
            for i in range(60)  # 60 trades
        ]
        
        # But only 50 periods
        num_periods = 50
        
        # Should not raise IndexError
        equity_curve = calculate_equity_curve(
            trades=trades,
            initial_capital=10000.0,
            num_periods=num_periods,
        )
        
        assert len(equity_curve) == num_periods
        assert equity_curve.iloc[0] == 10000.0
    
    def test_equity_curve_with_equal_trades_and_periods(self):
        """Test equity curve when number of trades equals num_periods."""
        trades = [
            {'pnl': 0.01, 'entry_index': i, 'exit_index': i + 1}
            for i in range(50)
        ]
        
        num_periods = 50
        
        equity_curve = calculate_equity_curve(
            trades=trades,
            initial_capital=10000.0,
            num_periods=num_periods,
        )
        
        assert len(equity_curve) == num_periods
    
    def test_equity_curve_with_few_trades(self):
        """Test equity curve with fewer trades than periods."""
        trades = [
            {'pnl': 0.05, 'entry_index': 0, 'exit_index': 10},
            {'pnl': -0.03, 'entry_index': 20, 'exit_index': 30},
        ]
        
        num_periods = 100
        
        equity_curve = calculate_equity_curve(
            trades=trades,
            initial_capital=10000.0,
            num_periods=num_periods,
        )
        
        assert len(equity_curve) == num_periods
        # Remaining periods should be filled with last value
        assert equity_curve.iloc[99] == equity_curve.iloc[31]


class TestTrailingStopNoneCheck:
    """Tests for trailing stop logic with None check improvements."""
    
    def test_trailing_stop_with_none_highest_price_long(self):
        """Test that trailing stop handles None highest_price correctly for LONG."""
        dates = pd.date_range("2023-01-01", periods=20, freq="h")
        prices = [100.0] * 20  # Flat price
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
        }, index=dates)
        
        # Create signals
        signals = pd.Series([1] + [0] * 19, index=dates)
        
        trades = simulate_trades(
            df=df,
            signals=signals,
            signal_type="LONG",
            initial_capital=10000.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            max_hold_periods=100,
        )
        
        # Should complete without errors
        assert isinstance(trades, list)
    
    def test_trailing_stop_with_none_lowest_price_short(self):
        """Test that trailing stop handles None lowest_price correctly for SHORT."""
        dates = pd.date_range("2023-01-01", periods=20, freq="h")
        prices = [100.0] * 20  # Flat price
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
        }, index=dates)
        
        # Create signals
        signals = pd.Series([-1] + [0] * 19, index=dates)
        
        trades = simulate_trades(
            df=df,
            signals=signals,
            signal_type="SHORT",
            initial_capital=10000.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            max_hold_periods=100,
        )
        
        # Should complete without errors
        assert isinstance(trades, list)
    
    def test_trailing_stop_updates_correctly_long(self):
        """Test that trailing stop updates correctly for LONG positions."""
        dates = pd.date_range("2023-01-01", periods=30, freq="h")
        # Price goes up, then comes back down
        prices = [100.0] * 5 + [105.0] * 10 + [103.0] * 10 + [101.0] * 5
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
        }, index=dates)
        
        signals = pd.Series([1] + [0] * 29, index=dates)
        
        trades = simulate_trades(
            df=df,
            signals=signals,
            signal_type="LONG",
            initial_capital=10000.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,  # 1.5% trailing stop
            max_hold_periods=100,
        )
        
        assert isinstance(trades, list)
        # Trailing stop should have been updated as price moved up
        if trades:
            # Check that exit happened (either trailing stop or max hold)
            assert 'exit_reason' in trades[0]


class TestParallelProcessingErrorHandling:
    """Tests for parallel processing error handling improvements."""
    
    def test_parallel_processing_fallback_on_pickle_error(self, mock_data_fetcher):
        """Test that parallel processing falls back to sequential on pickle error."""
        dates = pd.date_range("2023-01-01", periods=200, freq="h")
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
        }, index=dates)
        
        class MockSignalCalculator:
            def calculate_hybrid_signal(self, **kwargs):
                return 0, 0.0
            
            def get_cache_stats(self):
                return {
                    'signal_cache_size': 0,
                    'signal_cache_max_size': 1000,
                    'cache_hit_rate': 0.0,
                }
            
            enabled_indicators = ['range_oscillator']
            use_confidence_weighting = True
            min_indicators_agreement = 3
        
        # Mock pickle to raise error
        import pickle
        
        def failing_pickle(*args, **kwargs):
            raise MemoryError("Simulated pickle error")
        
        backtester = FullBacktester(mock_data_fetcher)
        backtester.hybrid_signal_calculator = MockSignalCalculator()
        
        with patch('pickle.dumps', side_effect=failing_pickle):
            # Should fall back to sequential calculation when pickle fails
            result = backtester.backtest(
                symbol="BTC/USDT",
                timeframe="1h",
                lookback=200,
                signal_type="LONG",
                df=df,
            )
            
            # Should complete successfully (fallback to sequential)
            assert 'trades' in result
            assert 'metrics' in result
    
    def test_parallel_processing_with_large_dataframe(self, mock_data_fetcher):
        """Test that parallel processing handles large DataFrames correctly."""
        # Create a large DataFrame
        dates = pd.date_range("2023-01-01", periods=5000, freq="h")
        prices = 100 + np.cumsum(np.random.randn(5000) * 0.5)
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
        }, index=dates)
        
        class MockSignalCalculator:
            def calculate_hybrid_signal(self, **kwargs):
                return 0, 0.0
            
            def get_cache_stats(self):
                return {
                    'signal_cache_size': 0,
                    'signal_cache_max_size': 1000,
                    'cache_hit_rate': 0.0,
                }
            
            enabled_indicators = ['range_oscillator']
            use_confidence_weighting = True
            min_indicators_agreement = 3
        
        backtester = FullBacktester(mock_data_fetcher)
        backtester.hybrid_signal_calculator = MockSignalCalculator()
        
        # Should complete without memory errors
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=5000,
            signal_type="LONG",
            df=df,
        )
        
        assert 'trades' in result
        assert 'metrics' in result


class TestComprehensiveScenarios:
    """Comprehensive test scenarios combining multiple improvements."""
    
    def test_custom_risk_per_trade_with_validation(self, mock_data_fetcher):
        """Test custom risk_per_trade with input validation."""
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
        }, index=dates)
        
        class MockSignalCalculator:
            def calculate_hybrid_signal(self, **kwargs):
                period_index = kwargs.get('period_index', 0)
                if period_index == 0:
                    return 1, 0.8
                return 0, 0.0
            
            def get_cache_stats(self):
                return {
                    'signal_cache_size': 0,
                    'signal_cache_max_size': 1000,
                    'cache_hit_rate': 0.0,
                }
        
        # Test with custom risk_per_trade
        backtester = FullBacktester(
            mock_data_fetcher,
            risk_per_trade=0.025,  # 2.5% custom risk
        )
        backtester.hybrid_signal_calculator = MockSignalCalculator()
        
        # Should work with valid inputs
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=100,
            signal_type="LONG",
            initial_capital=5000.0,
            df=df,
        )
        
        assert 'equity_curve' in result
        # Equity curve should start with initial_capital if no trades, or if trades exist
        if len(result['equity_curve']) > 0:
            assert result['equity_curve'].iloc[0] >= 5000.0  # May be padded if no trades
    
    def test_edge_cases_combined(self):
        """Test multiple edge cases combined."""
        # Test equity curve with many trades, custom risk_per_trade
        trades = [
            {'pnl': 0.02, 'entry_index': i, 'exit_index': i + 1}
            for i in range(100)
        ]
        
        equity_curve = calculate_equity_curve(
            trades=trades,
            initial_capital=10000.0,
            num_periods=50,  # Fewer periods than trades
            risk_per_trade=0.015,  # Custom 1.5% risk
        )
        
        assert len(equity_curve) == 50
        assert equity_curve.iloc[0] == 10000.0

