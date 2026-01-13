from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from modules.backtester import FullBacktester
from modules.position_sizing.core.position_sizer import PositionSizer


@pytest.fixture(scope="session")
def heavy_position_data():
    n = 5000
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.rand(n) * 1000,
        },
        index=dates,
    )
    return df


@pytest.fixture()
def small_position_data():
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.rand(n) * 1000,
        },
        index=dates,
    )
    return df


@pytest.fixture
def mock_data_fetcher(heavy_position_data):
    fetch_calls = []

    def fake_fetch_ohlcv_with_fallback_exchange(symbol, limit=2160, **kwargs):
        df = heavy_position_data.head(min(limit, len(heavy_position_data)))
        fetch_calls.append({"symbol": symbol, "kwargs": kwargs})
        return df, "binance"

    fetcher = SimpleNamespace()
    fetcher.fetch_ohlcv_with_fallback_exchange = fake_fetch_ohlcv_with_fallback_exchange
    fetcher.fetch_calls = fetch_calls
    return fetcher


@pytest.fixture(autouse=True)
def mock_signals(monkeypatch):
    monkeypatch.setattr("core.signal_calculators.get_range_oscillator_signal", lambda *a, **k: (1, 0.7))
    monkeypatch.setattr("core.signal_calculators.get_spc_signal", lambda *a, **k: (1, 0.6))
    monkeypatch.setattr("core.signal_calculators.get_xgboost_signal", lambda *a, **k: (1, 0.8))
    monkeypatch.setattr("core.signal_calculators.get_hmm_signal", lambda *a, **k: (1, 0.65))
    monkeypatch.setattr("core.signal_calculators.get_random_forest_signal", lambda *a, **k: (1, 0.75))


def test_end_to_end_dataframe_sharing(mock_data_fetcher, heavy_position_data):
    """Test that data is fetched once and shared across all components."""
    from modules.position_sizing.core.position_sizer import PositionSizer

    with (
        patch("core.signal_calculators.get_range_oscillator_signal", return_value=(1, 0.7)),
        patch("core.signal_calculators.get_spc_signal", return_value=(1, 0.6)),
        patch("core.signal_calculators.get_xgboost_signal", return_value=(1, 0.8)),
        patch("core.signal_calculators.get_hmm_signal", return_value=(1, 0.65)),
        patch("core.signal_calculators.get_random_forest_signal", return_value=(1, 0.75)),
    ):
        position_sizer = PositionSizer(mock_data_fetcher)

        mock_data_fetcher.fetch_calls.clear()
        result = position_sizer.calculate_position_size(symbol="BTC/USDT", account_balance=10000.0, signal_type="LONG")

        assert "symbol" in result
        assert "position_size_usdt" in result
        assert "metrics" in result
        assert len(mock_data_fetcher.fetch_calls) >= 1
        for call in mock_data_fetcher.fetch_calls:
            assert call["symbol"] == "BTC/USDT"


def test_backtester_independent_dataframe(small_position_data, mock_data_fetcher):
    backtester = FullBacktester(mock_data_fetcher)
    mock_data_fetcher.fetch_calls.clear()

    backtester.backtest(symbol="BTC/USDT", timeframe="1h", lookback=2160, signal_type="LONG", df=small_position_data)

    assert len(mock_data_fetcher.fetch_calls) == 0


def test_dataframe_consistency_across_components(small_position_data, mock_data_fetcher):
    with (
        patch("core.signal_calculators.get_range_oscillator_signal", return_value=(1, 0.7)),
        patch("core.signal_calculators.get_spc_signal", return_value=(1, 0.6)),
        patch("core.signal_calculators.get_xgboost_signal", return_value=(1, 0.8)),
        patch("core.signal_calculators.get_hmm_signal", return_value=(1, 0.65)),
        patch("core.signal_calculators.get_random_forest_signal", return_value=(1, 0.75)),
    ):
        backtester = FullBacktester(mock_data_fetcher)
        backtest_result = backtester.backtest(
            symbol="BTC/USDT", timeframe="1h", lookback=2160, signal_type="LONG", df=small_position_data
        )
        assert "trades" in backtest_result and "metrics" in backtest_result
        assert len(small_position_data) == 500
        assert "close" in small_position_data.columns


def test_performance_improvement_with_dataframe(mock_data_fetcher, small_position_data):
    with (
        patch("core.signal_calculators.get_range_oscillator_signal", return_value=(1, 0.7)),
        patch("core.signal_calculators.get_spc_signal", return_value=(1, 0.6)),
        patch("core.signal_calculators.get_xgboost_signal", return_value=(1, 0.8)),
        patch("core.signal_calculators.get_hmm_signal", return_value=(1, 0.65)),
        patch("core.signal_calculators.get_random_forest_signal", return_value=(1, 0.75)),
    ):
        backtester = FullBacktester(mock_data_fetcher)
        mock_data_fetcher.fetch_calls.clear()

        # Without DataFrame
        backtest_result1 = backtester.backtest(symbol="BTC/USDT", timeframe="1h", lookback=2160, signal_type="LONG")
        calls_without_df = len(mock_data_fetcher.fetch_calls)

        # With DataFrame
        mock_data_fetcher.fetch_calls.clear()
        backtest_result2 = backtester.backtest(
            symbol="BTC/USDT", timeframe="1h", lookback=2160, signal_type="LONG", df=small_position_data
        )
        calls_with_df = len(mock_data_fetcher.fetch_calls)

        assert calls_with_df <= calls_without_df
        assert "trades" in backtest_result1 and "trades" in backtest_result2
