
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules.backtester import FullBacktester

from modules.backtester import FullBacktester

"""
Tests for Full Backtester.
"""





def test_backtest_returns_valid_structure():
    """Test that backtest returns valid result structure."""

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=200, freq="h")
        # Create trending data
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
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

    # Mock signal calculators to avoid API calls
    with (
        patch("core.signal_calculators.get_range_oscillator_signal", return_value=(1, 0.7)),
        patch("core.signal_calculators.get_spc_signal", return_value=(1, 0.6)),
        patch("core.signal_calculators.get_xgboost_signal", return_value=(1, 0.8)),
        patch("core.signal_calculators.get_hmm_signal", return_value=(1, 0.65)),
        patch("core.signal_calculators.get_random_forest_signal", return_value=(1, 0.75)),
    ):
        backtester = FullBacktester(data_fetcher)

        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=200,
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
        dates = pd.date_range("2023-01-01", periods=200, freq="h")
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
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

    # Mock signal calculators to avoid API calls
    with (
        patch("core.signal_calculators.get_range_oscillator_signal", return_value=(1, 0.7)),
        patch("core.signal_calculators.get_spc_signal", return_value=(1, 0.6)),
        patch("core.signal_calculators.get_xgboost_signal", return_value=(1, 0.8)),
        patch("core.signal_calculators.get_hmm_signal", return_value=(1, 0.65)),
        patch("core.signal_calculators.get_random_forest_signal", return_value=(1, 0.75)),
    ):
        backtester = FullBacktester(data_fetcher)

        result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=200,
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
    backtester = FullBacktester(SimpleNamespace())

    result = backtester._empty_backtest_result()

    assert "trades" in result
    assert "equity_curve" in result
    assert "metrics" in result

    assert result["trades"] == []
    assert len(result["equity_curve"]) > 0
    assert result["metrics"]["num_trades"] == 0
