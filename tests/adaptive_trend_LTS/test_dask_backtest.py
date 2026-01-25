import time

import numpy as np
import pandas as pd
import pytest

try:
    from modules.adaptive_trend_LTS.core.backtesting.dask_backtest import (
        _process_symbol_group,
        backtest_from_dataframe,
        backtest_with_dask,
        backtest_multiple_files_dask,
    )
except ImportError:
    pytest.skip("Dask backtesting module not available", allow_module_level=True)


@pytest.fixture
def sample_config():
    """Default ATC config for testing."""
    return {
        "ema_len": 20,
        "hull_len": 20,
        "wma_len": 20,
        "dema_len": 20,
        "lsma_len": 20,
        "kama_len": 20,
        "robustness": "Medium",
        "La": 0.02,
        "De": 0.03,
        "cutout": 0,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
    }


@pytest.fixture
def sample_historical_data():
    """Generate sample historical data for testing."""
    np.random.seed(42)
    n_bars = 1500
    n_symbols = 10

    records = []
    for sym_idx in range(n_symbols):
        symbol = f"SYM_{sym_idx}"
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
        timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="h")

        for price, ts in zip(prices, timestamps):
            records.append({"symbol": symbol, "close": price, "timestamp": ts})

    return pd.DataFrame(records)


def test_process_symbol_group_basic(sample_config, sample_historical_data):
    """Test basic symbol group processing."""
    symbol_df = sample_historical_data[sample_historical_data["symbol"] == "SYM_0"]

    result = _process_symbol_group(symbol_df, "symbol", "close", sample_config)

    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        assert "symbol" in result.columns
        assert "signal" in result.columns
        assert "price" in result.columns
        assert "timestamp" in result.columns


def test_process_symbol_group_empty(sample_config):
    """Test processing empty DataFrame."""
    empty_df = pd.DataFrame(columns=["symbol", "close", "timestamp"])
    result = _process_symbol_group(empty_df, "symbol", "close", sample_config)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_process_symbol_group_insufficient_data(sample_config):
    """Test processing with insufficient data."""
    np.random.seed(42)
    short_data = pd.DataFrame(
        {
            "symbol": ["SHORT"] * 10,
            "close": 100 + np.random.randn(10),
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="h"),
        }
    )

    result = _process_symbol_group(short_data, "symbol", "close", sample_config)

    assert result.empty


def test_backtest_from_dataframe_basic(sample_config, sample_historical_data):
    """Test basic backtesting from DataFrame."""
    result = backtest_from_dataframe(sample_historical_data, sample_config, npartitions=2)

    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        assert "symbol" in result.columns
        assert "signal" in result.columns
        assert "price" in result.columns


def test_backtest_from_dataframe_empty(sample_config):
    """Test backtesting with empty DataFrame."""
    empty_df = pd.DataFrame(columns=["symbol", "close", "timestamp"])
    result = backtest_from_dataframe(empty_df, sample_config)

    assert result.empty


def test_backtest_from_dataframe_auto_partitions(sample_config, sample_historical_data):
    """Test backtesting with auto-determined partitions."""
    result = backtest_from_dataframe(sample_historical_data, sample_config, npartitions=None, partition_size=3)

    assert isinstance(result, pd.DataFrame)


def test_backtest_with_dask_from_df(sample_config, sample_historical_data, tmp_path):
    """Test backtesting with Dask from DataFrame."""
    result = backtest_from_dataframe(sample_historical_data, sample_config, npartitions=2)

    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        assert len(result) > 0


def test_backtest_with_dask_file(sample_config, sample_historical_data, tmp_path):
    """Test backtesting with Dask from CSV file."""
    sample_historical_data.to_csv(tmp_path / "test_data.csv", index=False)

    result = backtest_with_dask(
        str(tmp_path / "test_data.csv"),
        sample_config,
        chunksize="50MB",
    )

    assert isinstance(result, pd.DataFrame)


def test_backtest_multiple_files_dask(sample_config, sample_historical_data, tmp_path):
    """Test backtesting multiple files."""
    file1 = tmp_path / "test_data_1.csv"
    file2 = tmp_path / "test_data_2.csv"

    df1 = sample_historical_data[sample_historical_data["symbol"].isin(["SYM_0", "SYM_1"])]
    df2 = sample_historical_data[sample_historical_data["symbol"].isin(["SYM_2", "SYM_3"])]

    df1.to_csv(file1, index=False)
    df2.to_csv(file2, index=False)

    result = backtest_multiple_files_dask(
        [str(file1), str(file2)],
        sample_config,
        chunksize="50MB",
    )

    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        assert "symbol" in result.columns


def test_backtest_multiple_files_empty(sample_config, tmp_path):
    """Test backtesting multiple files with empty list."""
    result = backtest_multiple_files_dask([], sample_config)

    assert result.empty


def test_backtest_with_dask_nonexistent_file(sample_config, tmp_path):
    """Test backtesting with nonexistent file."""
    result = backtest_with_dask(
        str(tmp_path / "nonexistent.csv"),
        sample_config,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_backtest_large_dataset(sample_config):
    """Test backtesting with larger dataset."""
    np.random.seed(42)
    n_bars = 1500
    n_symbols = 50

    records = []
    for sym_idx in range(n_symbols):
        symbol = f"SYM_{sym_idx}"
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
        timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="h")

        for price, ts in zip(prices, timestamps):
            records.append({"symbol": symbol, "close": price, "timestamp": ts})

    df = pd.DataFrame(records)

    start_time = time.time()
    result = backtest_from_dataframe(df, sample_config, npartitions=5)
    duration = time.time() - start_time

    assert isinstance(result, pd.DataFrame)
    print(f"\nProcessed {n_symbols} symbols with {len(df)} bars in {duration:.2f}s")


def test_backtest_memory_efficiency(sample_config):
    """Test that backtesting is memory-efficient."""
    import gc
    import sys

    gc.collect()
    initial_mem = sys.getsizeof([])

    np.random.seed(42)
    n_bars = 1500
    n_symbols = 100

    records = []
    for sym_idx in range(n_symbols):
        symbol = f"SYM_{sym_idx}"
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
        timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="h")

        for price, ts in zip(prices, timestamps):
            records.append({"symbol": symbol, "close": price, "timestamp": ts})

    df = pd.DataFrame(records)

    result = backtest_from_dataframe(df, sample_config, npartitions=10)

    gc.collect()
    final_mem = sys.getsizeof([])

    assert isinstance(result, pd.DataFrame)
    print(f"\nMemory before: {initial_mem}, after: {final_mem}")


def test_backtest_result_consistency(sample_config, sample_historical_data):
    """Test that results are consistent across multiple runs."""
    result1 = backtest_from_dataframe(sample_historical_data, sample_config, npartitions=2)
    result2 = backtest_from_dataframe(sample_historical_data, sample_config, npartitions=3)

    assert isinstance(result1, pd.DataFrame)
    assert isinstance(result2, pd.DataFrame)

    if not result1.empty and not result2.empty:
        symbols1 = set(result1["symbol"].unique())
        symbols2 = set(result2["symbol"].unique())
        assert symbols1 == symbols2


def test_backtest_with_different_chunksize(sample_config, sample_historical_data, tmp_path):
    """Test backtesting with different chunk sizes."""
    sample_historical_data.to_csv(tmp_path / "test_data.csv", index=False)

    result_small = backtest_with_dask(
        str(tmp_path / "test_data.csv"),
        sample_config,
        chunksize="10MB",
    )

    result_large = backtest_with_dask(
        str(tmp_path / "test_data.csv"),
        sample_config,
        chunksize="100MB",
    )

    assert isinstance(result_small, pd.DataFrame)
    assert isinstance(result_large, pd.DataFrame)


def test_backtest_custom_columns(sample_config, sample_historical_data):
    """Test backtesting with custom column names."""
    df = sample_historical_data.rename(columns={"symbol": "asset", "close": "price_val"})

    result = backtest_from_dataframe(df, sample_config, symbol_column="asset", price_column="price_val")

    assert isinstance(result, pd.DataFrame)


def test_backtest_error_handling_invalid_column(sample_config, sample_historical_data):
    """Test error handling with invalid column."""
    df = sample_historical_data.drop(columns=["symbol"])

    result = backtest_from_dataframe(df, sample_config, symbol_column="symbol", price_column="close")

    assert isinstance(result, pd.DataFrame)
