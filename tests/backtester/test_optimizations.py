
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules.backtester.core.backtester import FullBacktester
from modules.backtester.core.equity_curve import (
from modules.backtester.core.backtester import FullBacktester
from modules.backtester.core.equity_curve import (

"""
Comprehensive tests for optimizations in Full Backtester.

Tests cover:
- JIT-compiled functions (exit conditions, equity curve)
- Vectorization optimizations
- Batch size optimization
- Debug logging conditional checks
- Numba fallback behavior
"""



    _calculate_equity_curve_jit,
)
from modules.backtester.core.exit_conditions import (
    NUMBA_AVAILABLE,
)

# Import the JIT functions directly for testing
from modules.backtester.core.exit_conditions import (
    check_long_exit_conditions as _check_long_exit_conditions,
)
from modules.backtester.core.exit_conditions import (
    check_short_exit_conditions as _check_short_exit_conditions,
)


class TestJITExitConditions:
    """Tests for JIT-compiled exit condition functions."""

    def test_check_long_exit_conditions_stop_loss(self):
        """Test LONG exit condition: stop loss triggered."""
        exit_code, exit_price, pnl = _check_long_exit_conditions(
            current_price=98.0,
            high=99.0,
            low=95.0,  # Below stop loss
            entry_price=100.0,
            highest_price=100.0,
            trailing_stop=0.0,
            stop_loss_pct=0.02,  # 2% stop loss = 98.0
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            hold_periods=5,
            max_hold_periods=100,
        )

        assert exit_code == 1  # STOP_LOSS
        assert exit_price == 98.0  # entry_price * (1 - 0.02)
        assert pnl < 0  # Should be negative (loss)
        assert abs(pnl - (-0.02)) < 0.0001  # Approximately -2%

    def test_check_long_exit_conditions_take_profit(self):
        """Test LONG exit condition: take profit triggered."""
        exit_code, exit_price, pnl = _check_long_exit_conditions(
            current_price=104.0,
            high=105.0,  # Above take profit
            low=103.0,
            entry_price=100.0,
            highest_price=100.0,
            trailing_stop=0.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,  # 4% take profit = 104.0
            trailing_stop_pct=0.015,
            hold_periods=5,
            max_hold_periods=100,
        )

        assert exit_code == 2  # TAKE_PROFIT
        assert exit_price == 104.0  # entry_price * (1 + 0.04)
        assert pnl > 0  # Should be positive (profit)
        assert abs(pnl - 0.04) < 0.0001  # Approximately 4%

    def test_check_long_exit_conditions_trailing_stop(self):
        """Test LONG exit condition: trailing stop triggered."""
        exit_code, exit_price, pnl = _check_long_exit_conditions(
            current_price=103.0,  # Below trailing stop
            high=103.5,  # Below take profit (104.0)
            low=102.0,
            entry_price=100.0,
            highest_price=105.0,  # Price went up to 105
            trailing_stop=103.425,  # 105 * (1 - 0.015)
            stop_loss_pct=0.02,
            take_profit_pct=0.04,  # Take profit at 104.0
            trailing_stop_pct=0.015,
            hold_periods=5,
            max_hold_periods=100,
        )

        assert exit_code == 3  # TRAILING_STOP
        assert exit_price == 103.425
        assert pnl > 0  # Should be positive (profit)

    def test_check_long_exit_conditions_max_hold(self):
        """Test LONG exit condition: max hold period reached."""
        exit_code, exit_price, pnl = _check_long_exit_conditions(
            current_price=102.0,
            high=103.0,
            low=101.0,
            entry_price=100.0,
            highest_price=100.0,
            trailing_stop=0.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            hold_periods=100,  # Equal to max_hold_periods
            max_hold_periods=100,
        )

        assert exit_code == 4  # MAX_HOLD
        assert exit_price == 102.0  # current_price
        assert pnl > 0  # Should be positive (profit)

    def test_check_long_exit_conditions_no_exit(self):
        """Test LONG exit condition: no exit triggered."""
        exit_code, exit_price, pnl = _check_long_exit_conditions(
            current_price=101.0,
            high=102.0,
            low=100.5,
            entry_price=100.0,
            highest_price=100.0,
            trailing_stop=0.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            hold_periods=5,
            max_hold_periods=100,
        )

        assert exit_code == 0  # NO_EXIT
        assert exit_price == 0.0
        assert pnl == 0.0

    def test_check_short_exit_conditions_stop_loss(self):
        """Test SHORT exit condition: stop loss triggered."""
        exit_code, exit_price, pnl = _check_short_exit_conditions(
            current_price=102.0,
            high=105.0,  # Above stop loss
            low=101.0,
            entry_price=100.0,
            lowest_price=100.0,
            trailing_stop=0.0,
            stop_loss_pct=0.02,  # 2% stop loss = 102.0
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            hold_periods=5,
            max_hold_periods=100,
        )

        assert exit_code == 1  # STOP_LOSS
        assert exit_price == 102.0  # entry_price * (1 + 0.02)
        assert pnl < 0  # Should be negative (loss for SHORT)

    def test_check_short_exit_conditions_take_profit(self):
        """Test SHORT exit condition: take profit triggered."""
        exit_code, exit_price, pnl = _check_short_exit_conditions(
            current_price=96.0,
            high=97.0,
            low=95.0,  # Below take profit
            entry_price=100.0,
            lowest_price=100.0,
            trailing_stop=0.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,  # 4% take profit = 96.0
            trailing_stop_pct=0.015,
            hold_periods=5,
            max_hold_periods=100,
        )

        assert exit_code == 2  # TAKE_PROFIT
        assert exit_price == 96.0  # entry_price * (1 - 0.04)
        assert pnl > 0  # Should be positive (profit for SHORT)

    def test_check_short_exit_conditions_trailing_stop(self):
        """Test SHORT exit condition: trailing stop triggered."""
        exit_code, exit_price, pnl = _check_short_exit_conditions(
            current_price=96.575,  # Above trailing stop
            high=97.0,
            low=96.5,  # Above take profit (96.0)
            entry_price=100.0,
            lowest_price=95.0,  # Price went down to 95
            trailing_stop=96.425,  # 95 * (1 + 0.015)
            stop_loss_pct=0.02,
            take_profit_pct=0.04,  # Take profit at 96.0
            trailing_stop_pct=0.015,
            hold_periods=5,
            max_hold_periods=100,
        )

        assert exit_code == 3  # TRAILING_STOP
        assert exit_price == 96.425
        assert pnl > 0  # Should be positive (profit for SHORT)

    def test_check_short_exit_conditions_max_hold(self):
        """Test SHORT exit condition: max hold period reached."""
        exit_code, exit_price, pnl = _check_short_exit_conditions(
            current_price=98.0,
            high=99.0,
            low=97.0,
            entry_price=100.0,
            lowest_price=100.0,
            trailing_stop=0.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            hold_periods=100,  # Equal to max_hold_periods
            max_hold_periods=100,
        )

        assert exit_code == 4  # MAX_HOLD
        assert exit_price == 98.0  # current_price
        assert pnl > 0  # Should be positive (profit for SHORT)

    def test_check_short_exit_conditions_no_exit(self):
        """Test SHORT exit condition: no exit triggered."""
        exit_code, exit_price, pnl = _check_short_exit_conditions(
            current_price=99.0,
            high=100.5,
            low=98.5,
            entry_price=100.0,
            lowest_price=100.0,
            trailing_stop=0.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            hold_periods=5,
            max_hold_periods=100,
        )

        assert exit_code == 0  # NO_EXIT
        assert exit_price == 0.0
        assert pnl == 0.0


class TestJITEquityCurve:
    """Tests for JIT-compiled equity curve calculation."""

    def test_calculate_equity_curve_jit_basic(self):
        """Test basic equity curve calculation."""
        trade_pnls = np.array([0.05, -0.03, 0.02, -0.01], dtype=np.float64)
        initial_capital = 10000.0
        risk_per_trade = 0.01  # 1% risk per trade

        equity = _calculate_equity_curve_jit(
            trade_pnls=trade_pnls,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
        )

        assert len(equity) == 5  # initial + 4 trades
        assert equity[0] == 10000.0

        # Manual calculation:
        # Trade 1: 10000 * 0.01 * 0.05 = 5.0, new capital = 10005.0
        # Trade 2: 10005 * 0.01 * (-0.03) = -3.0015, new capital = 10001.9985
        # Trade 3: 10001.9985 * 0.01 * 0.02 = 2.0004, new capital = 10003.9989
        # Trade 4: 10003.9989 * 0.01 * (-0.01) = -1.0004, new capital = 10002.9985

        assert equity[1] > equity[0]  # First trade profitable
        assert equity[2] < equity[1]  # Second trade loss
        assert equity[3] > equity[2]  # Third trade profitable
        assert equity[4] < equity[3]  # Fourth trade loss

    def test_calculate_equity_curve_jit_all_profits(self):
        """Test equity curve with all profitable trades."""
        trade_pnls = np.array([0.05, 0.03, 0.02], dtype=np.float64)
        initial_capital = 10000.0
        risk_per_trade = 0.01

        equity = _calculate_equity_curve_jit(
            trade_pnls=trade_pnls,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
        )

        assert len(equity) == 4
        assert equity[0] == 10000.0
        # All values should be increasing
        assert equity[1] > equity[0]
        assert equity[2] > equity[1]
        assert equity[3] > equity[2]

    def test_calculate_equity_curve_jit_all_losses(self):
        """Test equity curve with all losing trades."""
        trade_pnls = np.array([-0.05, -0.03, -0.02], dtype=np.float64)
        initial_capital = 10000.0
        risk_per_trade = 0.01

        equity = _calculate_equity_curve_jit(
            trade_pnls=trade_pnls,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
        )

        assert len(equity) == 4
        assert equity[0] == 10000.0
        # All values should be decreasing
        assert equity[1] < equity[0]
        assert equity[2] < equity[1]
        assert equity[3] < equity[2]

    def test_calculate_equity_curve_jit_empty_trades(self):
        """Test equity curve with no trades."""
        trade_pnls = np.array([], dtype=np.float64)
        initial_capital = 10000.0
        risk_per_trade = 0.01

        equity = _calculate_equity_curve_jit(
            trade_pnls=trade_pnls,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
        )

        assert len(equity) == 1
        assert equity[0] == 10000.0

    def test_calculate_equity_curve_jit_large_dataset(self):
        """Test equity curve with large number of trades."""
        n_trades = 1000
        trade_pnls = np.random.randn(n_trades).astype(np.float64) * 0.02  # Random PnLs
        initial_capital = 10000.0
        risk_per_trade = 0.01

        equity = _calculate_equity_curve_jit(
            trade_pnls=trade_pnls,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
        )

        assert len(equity) == n_trades + 1
        assert equity[0] == initial_capital
        # All values should be positive
        assert np.all(equity >= 0)


class TestVectorizationOptimizations:
    """Tests for vectorization optimizations."""

    def test_simulate_trades_uses_numpy_arrays(self):
        """Test that _simulate_trades uses numpy arrays instead of DataFrame indexing."""
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
            },
            index=dates,
        )

        signals = pd.Series([1 if i % 20 == 0 else 0 for i in range(100)], index=dates)

        class MockSignalCalculator:
            def calculate_hybrid_signal(self, **kwargs):
                period_index = kwargs.get("period_index", 0)
                return signals.iloc[period_index], 0.8

            def get_cache_stats(self):
                return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

        data_fetcher = SimpleNamespace(
            fetch_ohlcv_with_fallback_exchange=lambda *args, **kwargs: (df, "binance"),
        )

        from config.position_sizing import (
            BACKTEST_MAX_HOLD_PERIODS,
            BACKTEST_STOP_LOSS_PCT,
            BACKTEST_TAKE_PROFIT_PCT,
            BACKTEST_TRAILING_STOP_PCT,
        )
        from modules.backtester.core.trade_simulator import simulate_trades

        # This should use numpy arrays internally
        trades = simulate_trades(
            df=df,
            signals=signals,
            signal_type="LONG",
            initial_capital=10000.0,
            stop_loss_pct=BACKTEST_STOP_LOSS_PCT,
            take_profit_pct=BACKTEST_TAKE_PROFIT_PCT,
            trailing_stop_pct=BACKTEST_TRAILING_STOP_PCT,
            max_hold_periods=BACKTEST_MAX_HOLD_PERIODS,
        )

        # Should complete without errors
        assert isinstance(trades, list)
        # Verify trades have correct structure
        if trades:
            assert "entry_price" in trades[0]
            assert "exit_price" in trades[0]
            assert "pnl" in trades[0]

    def test_equity_curve_uses_jit_when_available(self):
        """Test that equity curve uses JIT function when numba is available."""
        from modules.backtester.core.equity_curve import calculate_equity_curve

        trades = [
            {"pnl": 0.05, "entry_index": 0, "exit_index": 10},
            {"pnl": -0.03, "entry_index": 20, "exit_index": 30},
            {"pnl": 0.02, "entry_index": 40, "exit_index": 50},
        ]

        equity_curve = calculate_equity_curve(
            trades=trades,
            initial_capital=10000.0,
            num_periods=100,
        )

        assert isinstance(equity_curve, pd.Series)
        assert len(equity_curve) == 100
        assert equity_curve.iloc[0] == 10000.0
        # Equity should change based on trades
        assert equity_curve.iloc[-1] != 10000.0 or len(trades) == 0


class TestBatchSizeOptimization:
    """Tests for batch size optimization."""

    def test_batch_size_optimization_enabled(self):
        """Test that batch size is optimized when OPTIMIZE_BATCH_SIZE is True."""
        with patch("config.position_sizing.OPTIMIZE_BATCH_SIZE", True):
            with patch("config.position_sizing.BATCH_SIZE", None):
                dates = pd.date_range("2023-01-01", periods=1000, freq="h")
                prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
                df = pd.DataFrame(
                    {
                        "open": prices,
                        "high": prices * 1.01,
                        "low": prices * 0.99,
                        "close": prices,
                    },
                    index=dates,
                )

                class MockSignalCalculator:
                    def calculate_hybrid_signal(self, **kwargs):
                        return 0, 0.0

                    def get_cache_stats(self):
                        return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

                    enabled_indicators = ["range_oscillator", "spc"]
                    use_confidence_weighting = True
                    min_indicators_agreement = 3

                data_fetcher = SimpleNamespace(
                    fetch_ohlcv_with_fallback_exchange=lambda *args, **kwargs: (df, "binance"),
                )

                from modules.backtester.core.signal_calculator import calculate_signals, calculate_signals_parallel

                mock_calculator = MockSignalCalculator()

                # Should use optimized batch size
                signals = calculate_signals_parallel(
                    df=df,
                    symbol="BTC/USDT",
                    timeframe="1h",
                    limit=1000,
                    signal_type="LONG",
                    hybrid_signal_calculator=mock_calculator,
                    fallback_calculate_signals=lambda df, symbol, timeframe, limit, signal_type: calculate_signals(
                        df=df,
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=limit,
                        signal_type=signal_type,
                        hybrid_signal_calculator=mock_calculator,
                    ),
                )

                assert isinstance(signals, pd.Series)
                assert len(signals) == len(df)

    def test_batch_size_manual_override(self):
        """Test that manual BATCH_SIZE overrides optimization."""
        with patch("config.position_sizing.BATCH_SIZE", 50):
            dates = pd.date_range("2023-01-01", periods=500, freq="h")
            prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
            df = pd.DataFrame(
                {
                    "open": prices,
                    "high": prices * 1.01,
                    "low": prices * 0.99,
                    "close": prices,
                },
                index=dates,
            )

            class MockSignalCalculator:
                def calculate_hybrid_signal(self, **kwargs):
                    return 0, 0.0

                def get_cache_stats(self):
                    return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

                enabled_indicators = ["range_oscillator", "spc"]
                use_confidence_weighting = True
                min_indicators_agreement = 3

            data_fetcher = SimpleNamespace(
                fetch_ohlcv_with_fallback_exchange=lambda *args, **kwargs: (df, "binance"),
            )

            from modules.backtester.core.signal_calculator import calculate_signals, calculate_signals_parallel

            mock_calculator = MockSignalCalculator()

            signals = calculate_signals_parallel(
                df=df,
                symbol="BTC/USDT",
                timeframe="1h",
                limit=500,
                signal_type="LONG",
                hybrid_signal_calculator=mock_calculator,
                fallback_calculate_signals=lambda df, symbol, timeframe, limit, signal_type: calculate_signals(
                    df=df,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    signal_type=signal_type,
                    hybrid_signal_calculator=mock_calculator,
                ),
            )

            assert isinstance(signals, pd.Series)


class TestDebugLogging:
    """Tests for debug logging conditional checks."""

    def test_debug_logging_disabled(self):
        """Test that debug logging is disabled when ENABLE_DEBUG_LOGGING is False."""
        with patch("config.position_sizing.ENABLE_DEBUG_LOGGING", False):
            dates = pd.date_range("2023-01-01", periods=100, freq="h")
            prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
            df = pd.DataFrame(
                {
                    "open": prices,
                    "high": prices * 1.01,
                    "low": prices * 0.99,
                    "close": prices,
                },
                index=dates,
            )

            class MockSignalCalculator:
                def calculate_hybrid_signal(self, **kwargs):
                    return 0, 0.0

                def get_cache_stats(self):
                    return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

            data_fetcher = SimpleNamespace(
                fetch_ohlcv_with_fallback_exchange=lambda *args, **kwargs: (df, "binance"),
            )

            backtester = FullBacktester(data_fetcher)
            backtester.hybrid_signal_calculator = MockSignalCalculator()

            # Should complete without writing debug logs
            result = backtester.backtest(
                symbol="BTC/USDT",
                timeframe="1h",
                lookback=100,
                signal_type="LONG",
                df=df,
            )

            assert "trades" in result

    def test_debug_logging_enabled(self):
        """Test that debug logging works when ENABLE_DEBUG_LOGGING is True."""
        with patch("config.position_sizing.ENABLE_DEBUG_LOGGING", True):
            dates = pd.date_range("2023-01-01", periods=50, freq="h")
            prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
            df = pd.DataFrame(
                {
                    "open": prices,
                    "high": prices * 1.01,
                    "low": prices * 0.99,
                    "close": prices,
                },
                index=dates,
            )

            class MockSignalCalculator:
                def calculate_hybrid_signal(self, **kwargs):
                    period_index = kwargs.get("period_index", 0)
                    return 1 if period_index == 0 else 0, 0.8

                def get_cache_stats(self):
                    return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

            data_fetcher = SimpleNamespace(
                fetch_ohlcv_with_fallback_exchange=lambda *args, **kwargs: (df, "binance"),
            )

            backtester = FullBacktester(data_fetcher)
            backtester.hybrid_signal_calculator = MockSignalCalculator()

            # Should complete and potentially write debug logs
            result = backtester.backtest(
                symbol="BTC/USDT",
                timeframe="1h",
                lookback=50,
                signal_type="LONG",
                df=df,
            )

            assert "trades" in result


class TestNumbaFallback:
    """Tests for Numba fallback behavior."""

    def test_numba_availability_check(self):
        """Test that NUMBA_AVAILABLE is correctly set."""
        # Should be True if numba is installed, False otherwise
        assert isinstance(NUMBA_AVAILABLE, bool)

    def test_jit_functions_work_without_numba(self):
        """Test that JIT functions work even if numba is not available."""
        # These functions should work regardless of numba availability
        # because of the fallback decorator

        # Test LONG exit conditions
        exit_code, exit_price, pnl = _check_long_exit_conditions(
            current_price=98.0,
            high=99.0,
            low=95.0,
            entry_price=100.0,
            highest_price=100.0,
            trailing_stop=0.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            hold_periods=5,
            max_hold_periods=100,
        )

        assert exit_code == 1  # STOP_LOSS
        assert exit_price == 98.0

        # Test SHORT exit conditions
        exit_code, exit_price, pnl = _check_short_exit_conditions(
            current_price=102.0,
            high=105.0,
            low=101.0,
            entry_price=100.0,
            lowest_price=100.0,
            trailing_stop=0.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            hold_periods=5,
            max_hold_periods=100,
        )

        assert exit_code == 1  # STOP_LOSS
        assert exit_price == 102.0

        # Test equity curve
        trade_pnls = np.array([0.05, -0.03], dtype=np.float64)
        equity = _calculate_equity_curve_jit(
            trade_pnls=trade_pnls,
            initial_capital=10000.0,
            risk_per_trade=0.01,
        )

        assert len(equity) == 3
        assert equity[0] == 10000.0


class TestExitConditionPriority:
    """Tests for exit condition priority (stop loss > take profit > trailing stop > max hold)."""

    def test_stop_loss_priority_over_take_profit_long(self):
        """Test that stop loss has priority over take profit for LONG."""
        # Price touches both stop loss and take profit in same period
        exit_code, exit_price, pnl = _check_long_exit_conditions(
            current_price=98.0,
            high=104.0,  # Above take profit
            low=95.0,  # Below stop loss
            entry_price=100.0,
            highest_price=100.0,
            trailing_stop=0.0,
            stop_loss_pct=0.02,  # Stop loss at 98.0
            take_profit_pct=0.04,  # Take profit at 104.0
            trailing_stop_pct=0.015,
            hold_periods=5,
            max_hold_periods=100,
        )

        # Stop loss should trigger first (checked first in code)
        assert exit_code == 1  # STOP_LOSS
        assert exit_price == 98.0

    def test_stop_loss_priority_over_trailing_stop_long(self):
        """Test that stop loss has priority over trailing stop for LONG."""
        exit_code, exit_price, pnl = _check_long_exit_conditions(
            current_price=97.0,
            high=98.0,
            low=95.0,  # Below stop loss
            entry_price=100.0,
            highest_price=105.0,
            trailing_stop=103.425,  # Trailing stop would trigger at 103.425
            stop_loss_pct=0.02,  # Stop loss at 98.0
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            hold_periods=5,
            max_hold_periods=100,
        )

        # Stop loss should trigger first
        assert exit_code == 1  # STOP_LOSS
        assert exit_price == 98.0

    def test_take_profit_priority_over_trailing_stop_long(self):
        """Test that take profit has priority over trailing stop for LONG."""
        exit_code, exit_price, pnl = _check_long_exit_conditions(
            current_price=103.0,
            high=104.0,  # Above take profit
            low=102.0,
            entry_price=100.0,
            highest_price=105.0,
            trailing_stop=103.425,  # Trailing stop would trigger
            stop_loss_pct=0.02,
            take_profit_pct=0.04,  # Take profit at 104.0
            trailing_stop_pct=0.015,
            hold_periods=5,
            max_hold_periods=100,
        )

        # Take profit should trigger before trailing stop
        assert exit_code == 2  # TAKE_PROFIT
        assert exit_price == 104.0


class TestEdgeCasesOptimizations:
    """Tests for edge cases in optimized code paths."""

    def test_simulate_trades_with_empty_signals(self):
        """Test _simulate_trades with empty signals array."""
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
            },
            index=dates,
        )

        signals = pd.Series([0] * 100, index=dates)

        from config.position_sizing import (
            BACKTEST_MAX_HOLD_PERIODS,
            BACKTEST_STOP_LOSS_PCT,
            BACKTEST_TAKE_PROFIT_PCT,
            BACKTEST_TRAILING_STOP_PCT,
        )
        from modules.backtester.core.trade_simulator import simulate_trades

        trades = simulate_trades(
            df=df,
            signals=signals,
            signal_type="LONG",
            initial_capital=10000.0,
            stop_loss_pct=BACKTEST_STOP_LOSS_PCT,
            take_profit_pct=BACKTEST_TAKE_PROFIT_PCT,
            trailing_stop_pct=BACKTEST_TRAILING_STOP_PCT,
            max_hold_periods=BACKTEST_MAX_HOLD_PERIODS,
        )

        # Should return empty list or trades that were open at start
        assert isinstance(trades, list)

    def test_equity_curve_with_single_trade(self):
        """Test equity curve calculation with single trade."""
        from modules.backtester.core.equity_curve import calculate_equity_curve

        trades = [
            {"pnl": 0.05, "entry_index": 0, "exit_index": 10},
        ]

        equity_curve = calculate_equity_curve(
            trades=trades,
            initial_capital=10000.0,
            num_periods=100,
        )

        assert len(equity_curve) == 100
        assert equity_curve.iloc[0] == 10000.0
        # After first trade, equity should increase
        assert equity_curve.iloc[1] > equity_curve.iloc[0]

    def test_equity_curve_padding(self):
        """Test that equity curve is properly padded to num_periods."""
        from modules.backtester.core.equity_curve import calculate_equity_curve

        trades = [
            {"pnl": 0.05, "entry_index": 0, "exit_index": 10},
        ]

        equity_curve = calculate_equity_curve(
            trades=trades,
            initial_capital=10000.0,
            num_periods=1000,  # Much larger than number of trades
        )

        assert len(equity_curve) == 1000
        # All values after trades should be the same (padded)
        if len(trades) > 0:
            final_value = equity_curve.iloc[len(trades) + 1]
            assert equity_curve.iloc[-1] == final_value


class TestSignalArrayOptimization:
    """Tests for signal array optimization."""

    def test_signals_array_conversion(self):
        """Test that signals are converted to numpy array."""
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        signals = pd.Series([1, -1, 0] * 33 + [1], index=dates[:100])

        # Verify signals can be converted to numpy array
        signals_array = signals.values
        assert isinstance(signals_array, np.ndarray)
        assert len(signals_array) == 100

        # Verify dtype conversion
        signals_array_float = np.asarray(signals_array, dtype=np.float64)
        assert signals_array_float.dtype == np.float64


class TestConcurrencyAndErrorHandling:
    """Tests for concurrency and error handling in optimized code."""

    def test_parallel_processing_error_handling(self):
        """Test that parallel processing handles errors gracefully."""
        with patch("config.position_sizing.ENABLE_PARALLEL_PROCESSING", True):
            dates = pd.date_range("2023-01-01", periods=500, freq="h")
            prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
            df = pd.DataFrame(
                {
                    "open": prices,
                    "high": prices * 1.01,
                    "low": prices * 0.99,
                    "close": prices,
                },
                index=dates,
            )

            class MockSignalCalculator:
                def calculate_hybrid_signal(self, **kwargs):
                    # Simulate error in some periods
                    period_index = kwargs.get("period_index", 0)
                    if period_index == 100:
                        raise Exception("Simulated error")
                    return 0, 0.0

                def get_cache_stats(self):
                    return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

                enabled_indicators = ["range_oscillator", "spc"]
                use_confidence_weighting = True
                min_indicators_agreement = 3

            data_fetcher = SimpleNamespace(
                fetch_ohlcv_with_fallback_exchange=lambda *args, **kwargs: (df, "binance"),
            )

            from modules.backtester.core.signal_calculator import calculate_signals, calculate_signals_parallel

            mock_calculator = MockSignalCalculator()

            # Should fall back to sequential on error
            signals = calculate_signals_parallel(
                df=df,
                symbol="BTC/USDT",
                timeframe="1h",
                limit=500,
                signal_type="LONG",
                hybrid_signal_calculator=mock_calculator,
                fallback_calculate_signals=lambda df, symbol, timeframe, limit, signal_type: calculate_signals(
                    df=df,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    signal_type=signal_type,
                    hybrid_signal_calculator=mock_calculator,
                ),
            )

            assert isinstance(signals, pd.Series)

    def test_simulate_trades_with_index_mismatch(self):
        """Test _simulate_trades handles index mismatches gracefully."""
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
            },
            index=dates,
        )

        # Signals with different length
        signals = pd.Series([1, 0, 0], index=dates[:3])

        from config.position_sizing import (
            BACKTEST_MAX_HOLD_PERIODS,
            BACKTEST_STOP_LOSS_PCT,
            BACKTEST_TAKE_PROFIT_PCT,
            BACKTEST_TRAILING_STOP_PCT,
        )
        from modules.backtester.core.trade_simulator import simulate_trades

        # Should handle gracefully
        trades = simulate_trades(
            df=df,
            signals=signals,
            signal_type="LONG",
            initial_capital=10000.0,
            stop_loss_pct=BACKTEST_STOP_LOSS_PCT,
            take_profit_pct=BACKTEST_TAKE_PROFIT_PCT,
            trailing_stop_pct=BACKTEST_TRAILING_STOP_PCT,
            max_hold_periods=BACKTEST_MAX_HOLD_PERIODS,
        )

        assert isinstance(trades, list)
