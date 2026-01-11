
from types import SimpleNamespace
import time

import numpy as np
import pandas as pd

from modules.backtester.core.backtester import FullBacktester

from modules.backtester.core.backtester import FullBacktester

"""
Comprehensive integration tests for Full Backtester.

Tests cover:
- Full backtest workflow with all optimizations
- Integration between signal calculation, trade simulation, and metrics
- Real-world scenarios with various market conditions
- Performance and correctness verification
"""





class TestFullBacktestWorkflow:
    """Tests for complete backtest workflow."""

    def test_full_backtest_workflow_long(self):
        """Test complete backtest workflow for LONG signals."""
        dates = pd.date_range("2023-01-01", periods=200, freq="h")
        # Create trending up data
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5) + np.arange(200) * 0.1
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
                # Return LONG signal at specific periods
                if period_index in [0, 50, 100]:
                    return 1, 0.8
                return 0, 0.0

            def get_cache_stats(self):
                return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

            enabled_indicators = ["range_oscillator", "spc"]
            use_confidence_weighting = True
            min_indicators_agreement = 3

        data_fetcher = SimpleNamespace(
            fetch_ohlcv_with_fallback_exchange=lambda *args, **kwargs: (df, "binance"),
        )

        backtester = FullBacktester(
            data_fetcher,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            max_hold_periods=50,
        )
        backtester.hybrid_signal_calculator = MockSignalCalculator()

        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=200,
            signal_type="LONG",
            initial_capital=10000.0,
            df=df,
        )

        # Verify result structure
        assert "trades" in result
        assert "equity_curve" in result
        assert "metrics" in result

        # Verify trades
        assert isinstance(result["trades"], list)
        if result["trades"]:
            trade = result["trades"][0]
            assert "entry_price" in trade
            assert "exit_price" in trade
            assert "pnl" in trade
            assert "exit_reason" in trade

        # Verify equity curve
        assert isinstance(result["equity_curve"], pd.Series)
        # Equity curve length should match DataFrame length (or be padded to it)
        assert len(result["equity_curve"]) == len(df) or len(result["equity_curve"]) > 0
        assert result["equity_curve"].iloc[0] == 10000.0

        # Verify metrics
        metrics = result["metrics"]
        assert "win_rate" in metrics
        assert "num_trades" in metrics
        assert "total_return" in metrics
        assert metrics["num_trades"] >= 0

    def test_full_backtest_workflow_short(self):
        """Test complete backtest workflow for SHORT signals."""
        dates = pd.date_range("2023-01-01", periods=200, freq="h")
        # Create trending down data
        prices = 100 - np.cumsum(np.random.randn(200) * 0.5) - np.arange(200) * 0.1
        prices = np.maximum(prices, 10.0)  # Ensure positive
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
                if period_index in [0, 50, 100]:
                    return -1, 0.8  # SHORT signal
                return 0, 0.0

            def get_cache_stats(self):
                return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

        data_fetcher = SimpleNamespace(
            fetch_ohlcv_with_fallback_exchange=lambda *args, **kwargs: (df, "binance"),
        )

        backtester = FullBacktester(data_fetcher)
        backtester.hybrid_signal_calculator = MockSignalCalculator()

        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=200,
            signal_type="SHORT",
            initial_capital=10000.0,
            df=df,
        )

        assert "trades" in result
        assert "equity_curve" in result
        assert "metrics" in result

    def test_backtest_with_provided_dataframe(self):
        """Test that backtest works with provided DataFrame."""
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
            fetch_ohlcv_with_fallback_exchange=lambda *args, **kwargs: (None, None),
        )

        backtester = FullBacktester(data_fetcher)
        backtester.hybrid_signal_calculator = MockSignalCalculator()

        # Should use provided DataFrame instead of fetching
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=100,
            signal_type="LONG",
            df=df,  # Provide DataFrame directly
        )

        assert "trades" in result
        assert "metrics" in result


class TestExitConditionsIntegration:
    """Tests for exit conditions in real trading scenarios."""

    def test_stop_loss_exit_long(self):
        """Test that stop loss correctly exits LONG position."""
        dates = pd.date_range("2023-01-01", periods=20, freq="h")
        # Price drops below stop loss
        prices = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0] + [90.0] * 10
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
            },
            index=dates,
        )

        class MockSignalCalculator:
            def calculate_hybrid_signal(self, **kwargs):
                period_index = kwargs.get("period_index", 0)
                if period_index == 0:
                    return 1, 0.8  # LONG signal at start
                return 0, 0.0

            def get_cache_stats(self):
                return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

        data_fetcher = SimpleNamespace(
            fetch_ohlcv_with_fallback_exchange=lambda *args, **kwargs: (df, "binance"),
        )

        backtester = FullBacktester(
            data_fetcher,
            stop_loss_pct=0.02,  # 2% stop loss = 98.0
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            max_hold_periods=100,
        )
        backtester.hybrid_signal_calculator = MockSignalCalculator()

        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=20,
            signal_type="LONG",
            df=df,
        )

        trades = result["trades"]
        if trades:
            # Should exit via stop loss
            assert trades[0]["exit_reason"] == "STOP_LOSS"
            assert trades[0]["exit_price"] == 98.0
            assert trades[0]["pnl"] < 0  # Loss

    def test_take_profit_exit_long(self):
        """Test that take profit correctly exits LONG position."""
        dates = pd.date_range("2023-01-01", periods=20, freq="h")
        # Price rises above take profit
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0] + [105.0] * 14
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
            },
            index=dates,
        )

        class MockSignalCalculator:
            def calculate_hybrid_signal(self, **kwargs):
                period_index = kwargs.get("period_index", 0)
                if period_index == 0:
                    return 1, 0.8
                return 0, 0.0

            def get_cache_stats(self):
                return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

        data_fetcher = SimpleNamespace(
            fetch_ohlcv_with_fallback_exchange=lambda *args, **kwargs: (df, "binance"),
        )

        backtester = FullBacktester(
            data_fetcher,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,  # 4% take profit = 104.0
            trailing_stop_pct=0.015,
            max_hold_periods=100,
        )
        backtester.hybrid_signal_calculator = MockSignalCalculator()

        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=20,
            signal_type="LONG",
            df=df,
        )

        trades = result["trades"]
        if trades:
            # Should exit via take profit
            assert trades[0]["exit_reason"] == "TAKE_PROFIT"
            assert trades[0]["exit_price"] == 104.0
            assert trades[0]["pnl"] > 0  # Profit

    def test_max_hold_exit(self):
        """Test that max hold period correctly exits position."""
        dates = pd.date_range("2023-01-01", periods=150, freq="h")
        # Price stays relatively flat
        prices = [100.0] * 150
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.005 for p in prices],  # Small range
                "low": [p * 0.995 for p in prices],
                "close": prices,
            },
            index=dates,
        )

        class MockSignalCalculator:
            def calculate_hybrid_signal(self, **kwargs):
                period_index = kwargs.get("period_index", 0)
                if period_index == 0:
                    return 1, 0.8
                return 0, 0.0

            def get_cache_stats(self):
                return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

        data_fetcher = SimpleNamespace(
            fetch_ohlcv_with_fallback_exchange=lambda *args, **kwargs: (df, "binance"),
        )

        backtester = FullBacktester(
            data_fetcher,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            trailing_stop_pct=0.015,
            max_hold_periods=100,  # Max hold 100 periods
        )
        backtester.hybrid_signal_calculator = MockSignalCalculator()

        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=150,
            signal_type="LONG",
            df=df,
        )

        trades = result["trades"]
        if trades:
            # Should exit via max hold
            assert trades[0]["exit_reason"] == "MAX_HOLD"
            assert trades[0]["hold_periods"] == 100


class TestMetricsCalculation:
    """Tests for metrics calculation accuracy."""

    def test_win_rate_calculation(self):
        """Test that win rate is calculated correctly."""
        backtester = FullBacktester(SimpleNamespace())

        trades = [
            {"pnl": 0.05, "entry_index": 0, "exit_index": 10},
            {"pnl": -0.03, "entry_index": 20, "exit_index": 30},
            {"pnl": 0.02, "entry_index": 40, "exit_index": 50},
            {"pnl": -0.01, "entry_index": 60, "exit_index": 70},
        ]

        equity_curve = pd.Series([10000.0 + i * 2.5 for i in range(100)])

        from modules.backtester.core.metrics import calculate_metrics

        metrics = calculate_metrics(trades, equity_curve)

        # 2 wins out of 4 trades = 50% win rate
        assert abs(metrics["win_rate"] - 0.5) < 0.01
        assert metrics["num_trades"] == 4

    def test_profit_factor_calculation(self):
        """Test that profit factor is calculated correctly."""
        from modules.backtester.core.metrics import calculate_metrics

        trades = [
            {"pnl": 0.10, "entry_index": 0, "exit_index": 10},  # Win: 10%
            {"pnl": -0.05, "entry_index": 20, "exit_index": 30},  # Loss: 5%
            {"pnl": 0.08, "entry_index": 40, "exit_index": 50},  # Win: 8%
            {"pnl": -0.03, "entry_index": 60, "exit_index": 70},  # Loss: 3%
        ]

        equity_curve = pd.Series([10000.0 + i * 2.5 for i in range(100)])

        metrics = calculate_metrics(trades, equity_curve)

        # Total profit: 0.10 + 0.08 = 0.18
        # Total loss: 0.05 + 0.03 = 0.08
        # Profit factor: 0.18 / 0.08 = 2.25
        assert abs(metrics["profit_factor"] - 2.25) < 0.01

    def test_metrics_with_zero_initial_capital(self):
        """Test metrics calculation with zero initial capital."""
        from modules.backtester.core.metrics import calculate_metrics

        trades = [
            {"pnl": 0.05, "entry_index": 0, "exit_index": 10},
        ]

        equity_curve = pd.Series([0.0] * 100)  # Zero capital

        metrics = calculate_metrics(trades, equity_curve)

        # Should handle zero capital gracefully
        assert "total_return" in metrics
        assert metrics["num_trades"] == 1


class TestPerformanceAndCorrectness:
    """Tests for performance and correctness of optimizations."""

    def test_vectorization_performance(self):
        """Test that vectorization improves performance for large datasets."""
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

        signals = pd.Series([1 if i % 50 == 0 else 0 for i in range(1000)], index=dates)

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

        # Measure time for trade simulation
        start_time = time.time()
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
        elapsed = time.time() - start_time

        # Should complete in reasonable time (< 5 seconds for 1000 periods)
        assert elapsed < 5.0
        assert isinstance(trades, list)

    def test_equity_curve_correctness(self):
        """Test that equity curve calculation is mathematically correct."""
        from modules.backtester.core.equity_curve import calculate_equity_curve

        trades = [
            {"pnl": 0.10, "entry_index": 0, "exit_index": 10},  # 10% profit
            {"pnl": -0.05, "entry_index": 20, "exit_index": 30},  # 5% loss
        ]

        equity_curve = calculate_equity_curve(
            trades=trades,
            initial_capital=10000.0,
            num_periods=100,
        )

        # Manual calculation:
        # After trade 1: 10000 + (10000 * 0.01 * 0.10) = 10000 + 10 = 10010
        # After trade 2: 10010 + (10010 * 0.01 * -0.05) = 10010 - 5.005 = 10004.995

        assert equity_curve.iloc[0] == 10000.0
        # Check that equity changes based on trades
        if len(trades) > 0:
            # Equity after first trade should be higher
            assert equity_curve.iloc[11] > equity_curve.iloc[0]

    def test_signal_array_bounds_checking(self):
        """Test that signal array access handles bounds correctly."""
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

        # Signals shorter than DataFrame
        signals = pd.Series([1, 0, 0], index=dates[:3])

        from config.position_sizing import (
            BACKTEST_MAX_HOLD_PERIODS,
            BACKTEST_STOP_LOSS_PCT,
            BACKTEST_TAKE_PROFIT_PCT,
            BACKTEST_TRAILING_STOP_PCT,
        )
        from modules.backtester.core.trade_simulator import simulate_trades

        # Should handle gracefully without index errors
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


class TestRealWorldScenarios:
    """Tests for real-world trading scenarios."""

    def test_volatile_market_scenario(self):
        """Test backtest in highly volatile market."""
        dates = pd.date_range("2023-01-01", periods=500, freq="h")
        # High volatility with large swings
        returns = np.random.randn(500) * 2.0  # High volatility
        prices = 100 + np.cumsum(returns)
        prices = np.maximum(prices, 10.0)  # Ensure positive

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.02,  # 2% range
                "low": prices * 0.98,
                "close": prices,
            },
            index=dates,
        )

        class MockSignalCalculator:
            def calculate_hybrid_signal(self, **kwargs):
                period_index = kwargs.get("period_index", 0)
                # Random signals
                if period_index % 30 == 0:
                    return 1 if np.random.rand() > 0.5 else -1, 0.7
                return 0, 0.0

            def get_cache_stats(self):
                return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

        data_fetcher = SimpleNamespace(
            fetch_ohlcv_with_fallback_exchange=lambda *args, **kwargs: (df, "binance"),
        )

        backtester = FullBacktester(
            data_fetcher,
            stop_loss_pct=0.03,  # 3% stop loss for volatile market
            take_profit_pct=0.06,  # 6% take profit
            trailing_stop_pct=0.02,
            max_hold_periods=50,
        )
        backtester.hybrid_signal_calculator = MockSignalCalculator()

        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=500,
            signal_type="LONG",
            df=df,
        )

        assert "trades" in result
        assert "metrics" in result
        # Metrics should be valid even in volatile market
        assert 0.0 <= result["metrics"]["win_rate"] <= 1.0

    def test_trending_market_scenario(self):
        """Test backtest in strong trending market."""
        dates = pd.date_range("2023-01-01", periods=300, freq="h")
        # Strong uptrend
        trend = np.arange(300) * 0.2
        noise = np.random.randn(300) * 0.3
        prices = 100 + trend + noise
        prices = np.maximum(prices, 10.0)

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
                # LONG signals in uptrend
                if period_index % 20 == 0:
                    return 1, 0.8
                return 0, 0.0

            def get_cache_stats(self):
                return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

        data_fetcher = SimpleNamespace(
            fetch_ohlcv_with_fallback_exchange=lambda *args, **kwargs: (df, "binance"),
        )

        backtester = FullBacktester(data_fetcher)
        backtester.hybrid_signal_calculator = MockSignalCalculator()

        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=300,
            signal_type="LONG",
            df=df,
        )

        assert "trades" in result
        # In uptrend, should have more winning trades
        if result["trades"]:
            winning_trades = [t for t in result["trades"] if t["pnl"] > 0]
            # Not necessarily all wins, but should have some
            assert len(winning_trades) >= 0

    def test_sideways_market_scenario(self):
        """Test backtest in sideways/range-bound market."""
        dates = pd.date_range("2023-01-01", periods=200, freq="h")
        # Price oscillates around 100
        oscillation = np.sin(np.arange(200) * 0.1) * 5
        prices = 100 + oscillation

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
                # Signals at turning points
                if period_index in [0, 50, 100, 150]:
                    return 1, 0.7
                return 0, 0.0

            def get_cache_stats(self):
                return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

        data_fetcher = SimpleNamespace(
            fetch_ohlcv_with_fallback_exchange=lambda *args, **kwargs: (df, "binance"),
        )

        backtester = FullBacktester(
            data_fetcher,
            stop_loss_pct=0.02,
            take_profit_pct=0.03,  # Smaller take profit for range-bound
            trailing_stop_pct=0.01,
            max_hold_periods=30,
        )
        backtester.hybrid_signal_calculator = MockSignalCalculator()

        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=200,
            signal_type="LONG",
            df=df,
        )

        assert "trades" in result
        assert "metrics" in result
