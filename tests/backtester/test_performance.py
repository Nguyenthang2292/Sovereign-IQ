
from types import SimpleNamespace
from unittest.mock import patch
import time

import pandas as pd

from modules.backtester import FullBacktester

from modules.backtester import FullBacktester

"""
Tests for performance monitoring and profiling in Full Backtester.
"""




# Fixtures from conftest.py will be automatically available


def test_performance_logging_enabled(mock_data_fetcher):
    """Test that performance metrics are logged when enabled."""
    with patch("config.position_sizing.LOG_PERFORMANCE_METRICS", True):
        backtester = FullBacktester(mock_data_fetcher)

        start_time = time.time()
        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=300,
            signal_type="LONG",
        )
        elapsed = time.time() - start_time

        # Should complete successfully
        assert "trades" in result
        assert "metrics" in result
        assert elapsed > 0  # Should take some time


def test_performance_profiling_disabled(mock_data_fetcher):
    """Test that profiling can be disabled."""
    with patch("config.position_sizing.ENABLE_PERFORMANCE_PROFILING", False):
        backtester = FullBacktester(mock_data_fetcher)

        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=300,
            signal_type="LONG",
        )

        # Should still work without profiling
        assert "trades" in result
        assert "metrics" in result


def test_backtest_handles_empty_data(mock_data_fetcher):
    """Test that backtest handles empty data gracefully."""

    def empty_fetch(symbol, **kwargs):
        return pd.DataFrame(), "binance"

    empty_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=empty_fetch,
    )

    backtester = FullBacktester(empty_fetcher)

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=100,
        signal_type="LONG",
    )

    # Should return empty result structure
    assert "trades" in result
    assert "metrics" in result
    assert len(result["trades"]) == 0
    assert result["metrics"]["num_trades"] == 0


def test_backtest_handles_missing_columns(mock_data_fetcher):
    """Test that backtest handles missing columns gracefully."""

    def incomplete_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        # Missing 'close' column
        df = pd.DataFrame(
            {
                "open": [100] * 100,
                "high": [101] * 100,
                "low": [99] * 100,
            },
            index=dates,
        )
        return df, "binance"

    incomplete_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=incomplete_fetch,
    )

    backtester = FullBacktester(incomplete_fetcher)

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=100,
        signal_type="LONG",
    )

    # Should return empty result structure
    assert "trades" in result
    assert "metrics" in result
    assert result["metrics"]["num_trades"] == 0


def test_backtest_metrics_calculation(mock_data_fetcher):
    """Test that backtest calculates metrics correctly."""
    backtester = FullBacktester(mock_data_fetcher)

    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=300,
        signal_type="LONG",
    )

    metrics = result["metrics"]

    # Check that all required metrics are present and valid
    assert "win_rate" in metrics
    assert "avg_win" in metrics
    assert "avg_loss" in metrics
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "num_trades" in metrics
    assert "profit_factor" in metrics

    # Check value ranges
    assert 0.0 <= metrics["win_rate"] <= 1.0
    assert metrics["num_trades"] >= 0
    assert isinstance(metrics["sharpe_ratio"], (int, float))
    assert isinstance(metrics["max_drawdown"], (int, float))
