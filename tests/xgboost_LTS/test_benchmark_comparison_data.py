"""Unit tests for XGBoost LTS benchmark comparison data fetching."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from modules.xgboost_LTS.benchmarks.benchmark_comparison.data import (
    MIN_BARS_FOR_BENCHMARK,
    fetch_symbols_data,
)


def _make_ohlcv_df(n_rows: int) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with n_rows and required columns."""
    import numpy as np

    ts = pd.date_range("2020-01-01", periods=n_rows, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.ones(n_rows),
            "high": np.ones(n_rows) * 1.1,
            "low": np.ones(n_rows) * 0.9,
            "close": np.ones(n_rows),
            "volume": np.ones(n_rows) * 100,
        },
        index=ts,
    )


@patch("modules.xgboost_LTS.benchmarks.benchmark_comparison.data.DataFetcher")
@patch("modules.xgboost_LTS.benchmarks.benchmark_comparison.data.ExchangeManager")
def test_fetch_symbols_data_accepts_bars_below_requested_when_above_min(
    mock_em: MagicMock,
    mock_df_class: MagicMock,
) -> None:
    """Symbols with bars >= MIN_BARS_FOR_BENCHMARK are accepted even if below requested bars (exchange limit)."""
    symbols = [f"SYM{i}/USDT" for i in range(10)]
    mock_fetcher = MagicMock()
    mock_fetcher.list_binance_futures_symbols.return_value = symbols
    # Simulate exchange returning 1500 bars when 2000 requested (e.g. Binance cap)
    mock_fetcher.fetch_ohlcv_with_fallback_exchange.side_effect = [
        (_make_ohlcv_df(1500), "binance") for _ in symbols
    ]
    mock_df_class.return_value = mock_fetcher

    result = fetch_symbols_data(num_symbols=10, bars=2000, timeframe="1h")

    assert len(result) == 10
    for sym, df in result.items():
        assert len(df) == 1500
    assert mock_fetcher.fetch_ohlcv_with_fallback_exchange.call_count == 10


@patch("modules.xgboost_LTS.benchmarks.benchmark_comparison.data.DataFetcher")
@patch("modules.xgboost_LTS.benchmarks.benchmark_comparison.data.ExchangeManager")
def test_fetch_symbols_data_rejects_when_below_min_bars(
    mock_em: MagicMock,
    mock_df_class: MagicMock,
) -> None:
    """Symbols with bars < MIN_BARS_FOR_BENCHMARK are not included."""
    symbols = ["BTC/USDT", "ETH/USDT"]
    mock_fetcher = MagicMock()
    mock_fetcher.list_binance_futures_symbols.return_value = symbols
    # Return fewer than MIN_BARS_FOR_BENCHMARK
    mock_fetcher.fetch_ohlcv_with_fallback_exchange.side_effect = [
        (_make_ohlcv_df(400), "binance"),
        (_make_ohlcv_df(499), "binance"),
    ]
    mock_df_class.return_value = mock_fetcher

    result = fetch_symbols_data(num_symbols=2, bars=2000, timeframe="1h")

    assert len(result) == 0


@patch("modules.xgboost_LTS.benchmarks.benchmark_comparison.data.DataFetcher")
@patch("modules.xgboost_LTS.benchmarks.benchmark_comparison.data.ExchangeManager")
def test_fetch_symbols_data_accepts_exactly_min_bars(
    mock_em: MagicMock,
    mock_df_class: MagicMock,
) -> None:
    """Symbol with exactly MIN_BARS_FOR_BENCHMARK bars is accepted."""
    symbols = ["BTC/USDT"]
    mock_fetcher = MagicMock()
    mock_fetcher.list_binance_futures_symbols.return_value = symbols
    mock_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (
        _make_ohlcv_df(MIN_BARS_FOR_BENCHMARK),
        "binance",
    )
    mock_df_class.return_value = mock_fetcher

    result = fetch_symbols_data(num_symbols=1, bars=5000, timeframe="1h")

    assert len(result) == 1
    assert len(result["BTC/USDT"]) == MIN_BARS_FOR_BENCHMARK


@patch("modules.xgboost_LTS.benchmarks.benchmark_comparison.data.DataFetcher")
@patch("modules.xgboost_LTS.benchmarks.benchmark_comparison.data.ExchangeManager")
def test_fetch_symbols_data_handles_fetch_exception(
    mock_em: MagicMock,
    mock_df_class: MagicMock,
) -> None:
    """When fetch raises, symbol is counted as failed and not in result."""
    symbols = ["BTC/USDT", "ETH/USDT"]
    mock_fetcher = MagicMock()
    mock_fetcher.list_binance_futures_symbols.return_value = symbols
    mock_fetcher.fetch_ohlcv_with_fallback_exchange.side_effect = [
        (_make_ohlcv_df(1000), "binance"),
        Exception("rate limit"),
    ]
    mock_df_class.return_value = mock_fetcher

    result = fetch_symbols_data(num_symbols=2, bars=2000, timeframe="1h")

    assert len(result) == 1
    assert "BTC/USDT" in result
    assert "ETH/USDT" not in result


@patch("modules.xgboost_LTS.benchmarks.benchmark_comparison.data.DataFetcher")
@patch("modules.xgboost_LTS.benchmarks.benchmark_comparison.data.ExchangeManager")
def test_fetch_symbols_data_handles_none_df(
    mock_em: MagicMock,
    mock_df_class: MagicMock,
) -> None:
    """When fetch returns (None, exchange), symbol is not in result."""
    symbols = ["BTC/USDT"]
    mock_fetcher = MagicMock()
    mock_fetcher.list_binance_futures_symbols.return_value = symbols
    mock_fetcher.fetch_ohlcv_with_fallback_exchange.return_value = (None, None)
    mock_df_class.return_value = mock_fetcher

    result = fetch_symbols_data(num_symbols=1, bars=1000, timeframe="1h")

    assert len(result) == 0
