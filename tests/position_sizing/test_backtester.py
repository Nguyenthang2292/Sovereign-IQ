from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules.backtester import FullBacktester

"""
Tests for Full Backtester.
"""


def test_backtest_returns_valid_structure():
    """Test that backtest returns valid result structure."""

    def fake_fetch(symbol, **kwargs):
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
        return df, "binance"

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    # Mock the entire HybridSignalCalculator to avoid complex logic
    with patch("modules.position_sizing.core.hybrid_signal_calculator.HybridSignalCalculator") as MockCalc:
        instance = MockCalc.return_value
        instance.get_cache_stats.return_value = {
            "signal_cache_size": 0,
            "signal_cache_max_size": 1000,
            "cache_hit_rate": 0.0,
        }
        instance.precompute_all_indicators_vectorized.return_value = {
            "range_oscillator": pd.DataFrame(
                {"signal": [1] + [0] * 49, "confidence": [0.8] + [0.0] * 49},
                index=pd.date_range("2023-01-01", periods=50, freq="h"),
            )
        }
        instance.calculate_signal_from_precomputed.return_value = (1, 0.8)
        instance.clear_cache.return_value = None

        backtester = FullBacktester(data_fetcher)

        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=50,
            signal_type="LONG",
        )

        assert "trades" in result
        assert "equity_curve" in result
        assert "metrics" in result

        assert isinstance(result["trades"], list)
        assert isinstance(result["equity_curve"], pd.Series)
        assert isinstance(result["metrics"], dict)


def test_backtest_metrics_structure():
    """Test that backtest metrics have required keys."""

    def fake_fetch(symbol, **kwargs):
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
        return df, "binance"

    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )

    with patch("modules.position_sizing.core.hybrid_signal_calculator.HybridSignalCalculator") as MockCalc:
        instance = MockCalc.return_value
        instance.get_cache_stats.return_value = {
            "signal_cache_size": 0,
            "signal_cache_max_size": 1000,
            "cache_hit_rate": 0.0,
        }
        instance.precompute_all_indicators_vectorized.return_value = {
            "range_oscillator": pd.DataFrame(
                {"signal": [1] + [0] * 49, "confidence": [0.8] + [0.0] * 49},
                index=pd.date_range("2023-01-01", periods=50, freq="h"),
            )
        }
        instance.calculate_signal_from_precomputed.return_value = (1, 0.8)
        instance.clear_cache.return_value = None

        backtester = FullBacktester(data_fetcher)

        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=50,
            signal_type="LONG",
        )

        metrics = result["metrics"]

        required_keys = [
            "win_rate",
            "avg_win",
            "avg_loss",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "num_trades",
            "profit_factor",
        ]

        for key in required_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))


def test_empty_backtest_result():
    """Test empty backtest result structure."""
    from modules.backtester.core.metrics import empty_backtest_result

    result = empty_backtest_result()

    assert "trades" in result
    assert "equity_curve" in result
    assert "metrics" in result

    assert result["trades"] == []
    assert result["metrics"]["num_trades"] == 0
