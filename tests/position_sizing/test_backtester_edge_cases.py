
from types import SimpleNamespace

import numpy as np
import pandas as pd

from modules.backtester import FullBacktester

from modules.backtester import FullBacktester

"""
Tests for edge cases in Full Backtester.
"""





def test_backtest_with_empty_dataframe():
    """Test that backtest handles empty DataFrame gracefully."""

    def fake_fetch(symbol, **kwargs):
        return pd.DataFrame(), "binance"

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    backtester = FullBacktester(data_fetcher)

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=100,
        signal_type="LONG",
    )

    assert result["trades"] == []
    assert len(result["equity_curve"]) > 0
    assert result["metrics"]["num_trades"] == 0


def test_backtest_with_missing_columns():
    """Test that backtest handles missing required columns."""

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=10, freq="h")
        # Missing 'high' and 'low' columns
        df = pd.DataFrame(
            {
                "open": [100] * 10,
                "close": [100] * 10,
            },
            index=dates,
        )
        return df, "binance"

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    backtester = FullBacktester(data_fetcher)

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=10,
        signal_type="LONG",
    )

    assert result["trades"] == []
    assert result["metrics"]["num_trades"] == 0


def test_backtest_with_zero_initial_capital():
    """Test that backtest handles zero initial capital."""

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=10, freq="h")
        prices = [100] * 10
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
            },
            index=dates,
        )
        return df, "binance"

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    backtester = FullBacktester(data_fetcher)

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=10,
        signal_type="LONG",
        initial_capital=0.0,
    )

    # Should still return valid structure
    assert "trades" in result
    assert "equity_curve" in result
    assert "metrics" in result


def test_backtest_with_negative_prices():
    """Test that backtest handles invalid price data."""

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=10, freq="h")
        # Negative prices should be handled
        prices = [-100, -99, -98, -97, -96, -95, -94, -93, -92, -91]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
            },
            index=dates,
        )
        return df, "binance"

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    backtester = FullBacktester(data_fetcher)

    # Should not crash, but may produce invalid results
    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=10,
        signal_type="LONG",
    )

    assert "trades" in result
    assert "equity_curve" in result
    assert "metrics" in result


def test_backtest_with_nan_prices():
    """Test that backtest handles NaN prices."""

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=10, freq="h")
        prices = [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 if not np.isnan(p) else np.nan for p in prices],
                "low": [p * 0.99 if not np.isnan(p) else np.nan for p in prices],
                "close": prices,
            },
            index=dates,
        )
        return df, "binance"

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    backtester = FullBacktester(data_fetcher)

    # Should handle NaN gracefully
    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=10,
        signal_type="LONG",
    )

    assert "trades" in result
    assert "equity_curve" in result
    assert "metrics" in result


def test_backtest_with_extreme_stop_loss():
    """Test that backtest handles extreme stop loss values."""

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=10, freq="h")
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
            },
            index=dates,
        )
        return df, "binance"

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    # Very large stop loss (50%)
    backtester = FullBacktester(
        data_fetcher,
        stop_loss_pct=0.5,
        take_profit_pct=0.1,
        trailing_stop_pct=0.015,
        max_hold_periods=100,
    )

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=10,
        signal_type="LONG",
    )

    assert "trades" in result
    assert "metrics" in result


def test_backtest_with_zero_stop_loss():
    """Test that backtest handles zero stop loss."""

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=10, freq="h")
        prices = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
            },
            index=dates,
        )
        return df, "binance"

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    backtester = FullBacktester(
        data_fetcher,
        stop_loss_pct=0.0,  # Zero stop loss
        take_profit_pct=0.1,
        trailing_stop_pct=0.015,
        max_hold_periods=100,
    )

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=10,
        signal_type="LONG",
    )

    assert "trades" in result
    assert "metrics" in result


def test_backtest_with_very_short_lookback():
    """Test that backtest handles very short lookback periods."""

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=5, freq="h")
        prices = [100, 101, 102, 103, 104]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
            },
            index=dates,
        )
        return df, "binance"

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    backtester = FullBacktester(data_fetcher)

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=5,
        signal_type="LONG",
    )

    assert "trades" in result
    assert "equity_curve" in result
    assert "metrics" in result


def test_backtest_metrics_with_no_winning_trades():
    """Test metrics calculation when there are no winning trades."""

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=10, freq="h")
        # All prices go down (all trades will be losses)
        prices = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
            },
            index=dates,
        )
        return df, "binance"

    class MockSignalCalculator:
        def calculate_hybrid_signal(self, **kwargs):
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return 1, 0.8
            return 0, 0.0

        def get_cache_stats(self):
            return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    backtester = FullBacktester(
        data_fetcher,
        stop_loss_pct=0.05,
        take_profit_pct=0.1,
        trailing_stop_pct=0.015,
        max_hold_periods=2,  # Short hold period to trigger exit
    )

    backtester.hybrid_signal_calculator = MockSignalCalculator()

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=10,
        signal_type="LONG",
    )

    metrics = result["metrics"]

    # Should handle no winning trades gracefully
    assert metrics["win_rate"] >= 0.0
    assert metrics["avg_win"] >= 0.0
    assert metrics["profit_factor"] >= 0.0


def test_backtest_metrics_with_no_losing_trades():
    """Test metrics calculation when there are no losing trades."""

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=10, freq="h")
        # All prices go up (all trades will be wins)
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
            },
            index=dates,
        )
        return df, "binance"

    class MockSignalCalculator:
        def calculate_hybrid_signal(self, **kwargs):
            period_index = kwargs.get("period_index", 0)
            if period_index == 0:
                return 1, 0.8
            return 0, 0.0

        def get_cache_stats(self):
            return {"signal_cache_size": 0, "signal_cache_max_size": 1000}

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    backtester = FullBacktester(
        data_fetcher,
        stop_loss_pct=0.05,
        take_profit_pct=0.1,
        trailing_stop_pct=0.015,
        max_hold_periods=2,
    )

    backtester.hybrid_signal_calculator = MockSignalCalculator()

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=10,
        signal_type="LONG",
    )

    metrics = result["metrics"]

    # Should handle no losing trades gracefully
    assert metrics["win_rate"] >= 0.0
    assert metrics["avg_loss"] >= 0.0
    # Profit factor should be calculated correctly (total_profit / 0 = inf or large number)
    assert metrics["profit_factor"] >= 0.0
