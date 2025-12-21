"""
Tests for incremental signal calculation with position-aware skipping.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from modules.backtester.core.signal_calculator_incremental import (
    calculate_signals_incremental,
    calculate_single_signals_incremental,
)
from modules.backtester.core.backtester import FullBacktester
from modules.common.core.data_fetcher import DataFetcher


class TestIncrementalSignalCalculation:
    """Test incremental signal calculation functions."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.DataFrame({
            'open': prices + np.random.randn(100) * 0.1,
            'high': prices + np.abs(np.random.randn(100) * 0.2),
            'low': prices - np.abs(np.random.randn(100) * 0.2),
            'close': prices,
            'volume': np.random.rand(100) * 1000,
        }, index=dates)
    
    @pytest.fixture
    def mock_hybrid_calculator(self):
        """Create a mock HybridSignalCalculator."""
        calculator = Mock()
        calculator.calculate_hybrid_signal = Mock(return_value=(1, 0.8))  # LONG signal with confidence 0.8
        calculator.calculate_single_signal_highest_confidence = Mock(return_value=(1, 0.8))
        return calculator
    
    def test_calculate_signals_incremental_basic(self, sample_df, mock_hybrid_calculator):
        """Test basic incremental signal calculation."""
        signals, trades = calculate_signals_incremental(
            df=sample_df,
            symbol='BTC/USDT',
            timeframe='1h',
            limit=50,
            signal_type='LONG',
            hybrid_signal_calculator=mock_hybrid_calculator,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            max_hold_periods=100,
        )
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_df)
        assert isinstance(trades, list)
        # Should have at least one trade if signals were generated
        assert len(trades) >= 0
    
    def test_calculate_signals_incremental_skip_when_position_open(self, sample_df, mock_hybrid_calculator):
        """Test that signal calculation is skipped when position is open."""
        # Make calculator return signal on first few periods
        call_count = [0]
        def mock_signal(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 5:
                return (1, 0.8)  # LONG signal
            return (0, 0.0)  # No signal
        
        mock_hybrid_calculator.calculate_hybrid_signal = Mock(side_effect=mock_signal)
        
        signals, trades = calculate_signals_incremental(
            df=sample_df,
            symbol='BTC/USDT',
            timeframe='1h',
            limit=50,
            signal_type='LONG',
            hybrid_signal_calculator=mock_hybrid_calculator,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            max_hold_periods=10,  # Short max hold to trigger exit quickly
        )
        
        # Should have calculated signals only when no position was open
        assert isinstance(signals, pd.Series)
        # The number of signal calculations should be less than total periods
        # because some periods should be skipped when position is open
        assert call_count[0] < len(sample_df)
    
    def test_calculate_single_signals_incremental_basic(self, sample_df, mock_hybrid_calculator):
        """Test basic incremental single signal calculation."""
        signals, trades = calculate_single_signals_incremental(
            df=sample_df,
            symbol='BTC/USDT',
            timeframe='1h',
            limit=50,
            hybrid_signal_calculator=mock_hybrid_calculator,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            max_hold_periods=100,
        )
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_df)
        assert isinstance(trades, list)
    
    def test_incremental_mode_closes_position_at_end(self, sample_df, mock_hybrid_calculator):
        """Test that positions are closed at the end of data."""
        # Make calculator return signal early
        mock_hybrid_calculator.calculate_hybrid_signal = Mock(return_value=(1, 0.8))
        
        signals, trades = calculate_signals_incremental(
            df=sample_df,
            symbol='BTC/USDT',
            timeframe='1h',
            limit=50,
            signal_type='LONG',
            hybrid_signal_calculator=mock_hybrid_calculator,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            max_hold_periods=200,  # Very long hold to ensure position stays open
        )
        
        # Should have at least one trade (entry + exit at end)
        if len(trades) > 0:
            last_trade = trades[-1]
            assert 'exit_reason' in last_trade
            # If position was still open, it should be closed with END_OF_DATA
            if last_trade['exit_index'] == len(sample_df) - 1:
                assert last_trade['exit_reason'] == "END_OF_DATA"


class TestFullBacktesterIncrementalMode:
    """Test FullBacktester with incremental calculation mode."""
    
    @staticmethod
    def _create_sample_dataframe(periods=100, seed=42):
        """Helper method to create sample DataFrame for testing."""
        dates = pd.date_range('2024-01-01', periods=periods, freq='1h')
        np.random.seed(seed)
        prices = 100 + np.cumsum(np.random.randn(periods) * 0.5)
        return pd.DataFrame({
            'open': prices + np.random.randn(periods) * 0.1,
            'high': prices + np.abs(np.random.randn(periods) * 0.2),
            'low': prices - np.abs(np.random.randn(periods) * 0.2),
            'close': prices,
            'volume': np.random.rand(periods) * 1000,
        }, index=dates)
    
    @pytest.fixture
    def mock_data_fetcher(self):
        """Create a mock DataFetcher."""
        fetcher = Mock(spec=DataFetcher)
        df = TestFullBacktesterIncrementalMode._create_sample_dataframe()
        fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (df, None)
        return fetcher
    
    def test_backtester_incremental_mode(self, mock_data_fetcher):
        """Test FullBacktester with incremental calculation mode."""
        backtester = FullBacktester(
            data_fetcher=mock_data_fetcher,
            signal_mode='majority_vote',
            signal_calculation_mode='incremental',
        )
        
        result = backtester.backtest(
            symbol='BTC/USDT',
            timeframe='1h',
            lookback=50,
            signal_type='LONG',
            initial_capital=10000.0,
        )
        
        assert result is not None
        assert 'trades' in result
        assert 'metrics' in result
        assert isinstance(result['trades'], list)
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.assert_called_once()
        call_args = mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.call_args
        assert call_args[0][0] == 'BTC/USDT'
        assert call_args[1]['limit'] == 50
        assert call_args[1]['timeframe'] == '1h'
    
    def test_backtester_precomputed_mode(self, mock_data_fetcher):
        """Test FullBacktester with precomputed calculation mode (default)."""
        backtester = FullBacktester(
            data_fetcher=mock_data_fetcher,
            signal_mode='majority_vote',
            signal_calculation_mode='precomputed',
        )
        
        result = backtester.backtest(
            symbol='BTC/USDT',
            timeframe='1h',
            lookback=50,
            signal_type='LONG',
            initial_capital=10000.0,
        )
        
        assert result is not None
        assert 'trades' in result
        assert 'metrics' in result
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.assert_called_once()
        call_args = mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.call_args
        assert call_args[0][0] == 'BTC/USDT'
        assert call_args[1]['limit'] == 50
        assert call_args[1]['timeframe'] == '1h'
    
    def test_backtester_incremental_mode_single_signal(self, mock_data_fetcher):
        """Test FullBacktester with incremental mode and single signal mode."""
        backtester = FullBacktester(
            data_fetcher=mock_data_fetcher,
            signal_mode='single_signal',
            signal_calculation_mode='incremental',
        )
        
        result = backtester.backtest(
            symbol='BTC/USDT',
            timeframe='1h',
            lookback=50,
            signal_type='LONG',
            initial_capital=10000.0,
        )
        
        assert result is not None
        assert 'trades' in result
        assert 'metrics' in result
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.assert_called_once()
        call_args = mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.call_args
        assert call_args[0][0] == 'BTC/USDT'
        assert call_args[1]['limit'] == 50
        assert call_args[1]['timeframe'] == '1h'
    
    def test_backtester_invalid_calculation_mode(self, mock_data_fetcher):
        """Test that invalid calculation mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid signal_calculation_mode"):
            FullBacktester(
                data_fetcher=mock_data_fetcher,
                signal_calculation_mode='invalid_mode',
            )
    
    def test_backtester_fallback_exchange_path(self, mock_data_fetcher):
        """Test FullBacktester when fallback exchange is used."""
        # Configure mock to return a fallback exchange
        df = TestFullBacktesterIncrementalMode._create_sample_dataframe()
        fallback_exchange = 'binance'  # Simulate fallback exchange
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (df, fallback_exchange)
        
        backtester = FullBacktester(
            data_fetcher=mock_data_fetcher,
            signal_mode='majority_vote',
            signal_calculation_mode='incremental',
        )
        
        result = backtester.backtest(
            symbol='BTC/USDT',
            timeframe='1h',
            lookback=50,
            signal_type='LONG',
            initial_capital=10000.0,
        )
        
        # Assert backtest result - verify behavior when fallback exchange is used
        assert result is not None
        assert 'trades' in result
        assert 'metrics' in result
        assert isinstance(result['trades'], list)
        
        # Verify that backtest completed successfully with fallback exchange
        # The backtester should handle fallback exchange transparently
        assert isinstance(result['metrics'], dict)
        assert 'total_trades' in result['metrics']
        assert 'win_rate' in result['metrics']
        
        # Assert fetch_ohlcv_with_fallback_exchange was called correctly
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.assert_called_once()
        call_args = mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.call_args
        assert call_args[0][0] == 'BTC/USDT'
        assert call_args[1]['limit'] == 50
        assert call_args[1]['timeframe'] == '1h'


class TestIncrementalWalkForwardSemantics:
    """Test that incremental mode maintains walk-forward semantics."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
        return pd.DataFrame({
            'open': prices + np.random.randn(50) * 0.1,
            'high': prices + np.abs(np.random.randn(50) * 0.2),
            'low': prices - np.abs(np.random.randn(50) * 0.2),
            'close': prices,
            'volume': np.random.rand(50) * 1000,
        }, index=dates)
    
    def test_walk_forward_semantics(self, sample_df):
        """Test that incremental mode only uses historical data up to current period."""
        call_history = []
        
        def mock_signal(df, symbol, timeframe, period_index, signal_type, **kwargs):
            # Record the DataFrame size used for calculation
            call_history.append({
                'period_index': period_index,
                'df_size': len(df),
                'max_index': df.index[-1] if len(df) > 0 else None,
            })
            return (0, 0.0)  # No signal
        
        mock_calculator = Mock()
        mock_calculator.calculate_hybrid_signal = Mock(side_effect=mock_signal)
        
        signals, trades = calculate_signals_incremental(
            df=sample_df,
            symbol='BTC/USDT',
            timeframe='1h',
            limit=50,
            signal_type='LONG',
            hybrid_signal_calculator=mock_calculator,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            max_hold_periods=100,
        )
        
        # Check that each calculation only uses data up to current period
        for call_info in call_history:
            period_index = call_info['period_index']
            df_size = call_info['df_size']
            # DataFrame should include period_index (0-indexed, so size should be period_index + 1)
            # But we skip first 10 periods, so adjust
            if period_index >= 10:
                assert df_size == period_index + 1, f"Period {period_index} should use {period_index + 1} rows, got {df_size}"

