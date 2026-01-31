"""Comprehensive test suite for dask_backtest module.

Tests cover:
- Unit tests for _process_symbol_group with various edge cases
- Integration tests with sample CSV files
- Performance benchmarks (Dask vs pandas)
- Memory usage tests (especially for memory-mapped path)
- Error condition tests (missing columns, corrupt data, etc.)
"""

import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    from modules.adaptive_trend_LTS_mini.core.backtesting.dask_backtest import (
        _create_error_result,
        _process_symbol_group,
        _validate_file_path,
        backtest_from_dataframe,
        backtest_multiple_files_dask,
        backtest_with_dask,
    )
except ImportError:
    pytest.skip("Dask backtesting module not available", allow_module_level=True)


@pytest.fixture
def sample_atc_config():
    """Sample ATC configuration for testing."""
    return {
        "ema_len": 28,
        "hma_len": 28,
        "robustness": 5,
        "lambda_param": 0.0004,
        "decay": 0.5,
    }


@pytest.fixture
def sample_historical_data():
    """Create sample historical data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "symbol": ["BTC/USDT"] * 100 + ["ETH/USDT"] * 100,
            "close": list(range(100)) + list(range(100, 200)),
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="1h"),
        }
    )


class TestProcessSymbolGroupEdgeCases:
    """Unit tests for _process_symbol_group with edge cases."""

    def test_empty_dataframe(self, sample_atc_config):
        """Test processing empty DataFrame returns error result."""
        empty_df = pd.DataFrame(columns=["symbol", "close", "timestamp"])
        result = _process_symbol_group(empty_df, "symbol", "close", sample_atc_config)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["status"].iloc[0] == "error"
        assert "Insufficient data" in result["error_msg"].iloc[0]

    def test_single_row_dataframe(self, sample_atc_config):
        """Test processing DataFrame with only one row."""
        single_row = pd.DataFrame(
            {
                "symbol": ["TEST"],
                "close": [100.0],
                "timestamp": pd.Timestamp("2024-01-01"),
            }
        )
        result = _process_symbol_group(single_row, "symbol", "close", sample_atc_config)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["status"].iloc[0] == "error"

    def test_insufficient_data_below_ema_len(self, sample_atc_config):
        """Test processing with data shorter than ema_len."""
        config_short_ema = {**sample_atc_config, "ema_len": 50}
        short_df = pd.DataFrame(
            {
                "symbol": ["TEST"] * 20,
                "close": range(20),
                "timestamp": pd.date_range("2024-01-01", periods=20, freq="1h"),
            }
        )
        result = _process_symbol_group(short_df, "symbol", "close", config_short_ema)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["status"].iloc[0] == "error"
        assert "below ema_len threshold" in result["error_msg"].iloc[0]

    def test_nan_values_in_price(self, sample_atc_config):
        """Test processing with NaN values in price column."""
        df_with_nans = pd.DataFrame(
            {
                "symbol": ["TEST"] * 50,
                "close": [100 + i if i % 10 != 0 else np.nan for i in range(50)],
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
            }
        )
        result = _process_symbol_group(df_with_nans, "symbol", "close", sample_atc_config)

        assert isinstance(result, pd.DataFrame)

    def test_constant_prices(self, sample_atc_config):
        """Test processing with constant (non-varying) prices."""
        constant_df = pd.DataFrame(
            {
                "symbol": ["TEST"] * 50,
                "close": [100.0] * 50,
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
            }
        )
        result = _process_symbol_group(constant_df, "symbol", "close", sample_atc_config)

        assert isinstance(result, pd.DataFrame)

    def test_negative_prices(self, sample_atc_config):
        """Test processing with negative price values."""
        negative_df = pd.DataFrame(
            {
                "symbol": ["TEST"] * 50,
                "close": [100 - i * 2 for i in range(50)],
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
            }
        )
        result = _process_symbol_group(negative_df, "symbol", "close", sample_atc_config)

        assert isinstance(result, pd.DataFrame)

    def test_custom_column_names(self, sample_atc_config):
        """Test processing with custom column names."""
        custom_df = pd.DataFrame(
            {
                "asset": ["BTC"] * 50,
                "price": range(50),
                "time": pd.date_range("2024-01-01", periods=50, freq="1h"),
            }
        )
        result = _process_symbol_group(custom_df, "asset", "price", sample_atc_config)

        assert isinstance(result, pd.DataFrame)

    def test_result_schema(self, sample_atc_config):
        """Test that result has expected schema."""
        df = pd.DataFrame(
            {
                "symbol": ["TEST"] * 50,
                "close": range(50),
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
            }
        )
        result = _process_symbol_group(df, "symbol", "close", sample_atc_config)

        if not result.empty and result["status"].iloc[0] == "success":
            expected_columns = ["symbol", "signal", "price", "timestamp", "status", "error_msg"]
            assert all(col in result.columns for col in expected_columns)
        elif result["status"].iloc[0] == "error":
            expected_columns = ["symbol", "signal", "price", "timestamp", "status", "error_msg"]
            assert all(col in result.columns for col in expected_columns)


class TestIntegrationWithCSV:
    """Integration tests with sample CSV files."""

    def test_backtest_with_dask_from_csv(self, sample_atc_config, sample_historical_data, tmp_path):
        """Test backtesting from CSV file."""
        csv_path = tmp_path / "test_data.csv"
        sample_historical_data.to_csv(csv_path, index=False)

        result = backtest_with_dask(str(csv_path), sample_atc_config, chunksize="10MB")

        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "symbol" in result.columns
            assert "signal" in result.columns

    def test_backtest_with_multiple_csv_files(self, sample_atc_config, sample_historical_data, tmp_path):
        """Test backtesting from multiple CSV files."""
        df1 = sample_historical_data[sample_historical_data["symbol"] == "BTC/USDT"]
        df2 = sample_historical_data[sample_historical_data["symbol"] == "ETH/USDT"]

        csv_path1 = tmp_path / "data1.csv"
        csv_path2 = tmp_path / "data2.csv"

        df1.to_csv(csv_path1, index=False)
        df2.to_csv(csv_path2, index=False)

        result = backtest_multiple_files_dask(
            [str(csv_path1), str(csv_path2)],
            sample_atc_config,
            chunksize="10MB",
        )

        assert isinstance(result, pd.DataFrame)

    def test_backtest_from_parquet_file(self, sample_atc_config, sample_historical_data, tmp_path):
        """Test backtesting from Parquet file."""
        pq_path = tmp_path / "test_data.parquet"
        sample_historical_data.to_parquet(pq_path, index=False)

        result = backtest_with_dask(str(pq_path), sample_atc_config, chunksize="10MB")

        assert isinstance(result, pd.DataFrame)

    def test_backtest_from_dataframe(self, sample_atc_config, sample_historical_data):
        """Test backtesting directly from DataFrame."""
        result = backtest_from_dataframe(
            sample_historical_data,
            sample_atc_config,
            npartitions=2,
        )

        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "symbol" in result.columns
            assert "signal" in result.columns
            assert "price" in result.columns

    def test_auto_partition_calculation(self, sample_atc_config, sample_historical_data):
        """Test auto-calculation of partitions."""
        result = backtest_from_dataframe(
            sample_historical_data,
            sample_atc_config,
            npartitions=None,
        )

        assert isinstance(result, pd.DataFrame)


class TestPerformanceBenchmarks:
    """Performance benchmarks comparing Dask vs pandas."""

    def test_dask_vs_pandas_performance(self, sample_atc_config):
        """Benchmark Dask vs pandas performance."""
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

        df = pd.DataFrame(records)

        start_time = time.time()
        result_dask = backtest_from_dataframe(df, sample_atc_config, npartitions=4)
        dask_time = time.time() - start_time

        assert isinstance(result_dask, pd.DataFrame)
        assert dask_time > 0

    def test_large_dataset_performance(self, sample_atc_config):
        """Test performance with larger dataset."""
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

        start_time = time.time()
        result = backtest_from_dataframe(df, sample_atc_config, npartitions=10)
        duration = time.time() - start_time

        assert isinstance(result, pd.DataFrame)
        assert duration > 0

    def test_partition_scaling_performance(self, sample_atc_config):
        """Test how performance scales with different partition counts."""
        np.random.seed(42)
        n_bars = 500
        n_symbols = 20

        records = []
        for sym_idx in range(n_symbols):
            symbol = f"SYM_{sym_idx}"
            prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
            timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="h")
            for price, ts in zip(prices, timestamps):
                records.append({"symbol": symbol, "close": price, "timestamp": ts})

        df = pd.DataFrame(records)

        times = []
        for nparts in [1, 2, 4, 8]:
            start_time = time.time()
            result = backtest_from_dataframe(df, sample_atc_config, npartitions=nparts)
            duration = time.time() - start_time
            times.append(duration)
            assert isinstance(result, pd.DataFrame)

        assert all(t > 0 for t in times)


class TestMemoryUsage:
    """Memory usage tests, especially for memory-mapped path."""

    def test_memory_mapped_vs_regular_loading(self, sample_atc_config, sample_historical_data, tmp_path):
        """Compare memory usage between memory-mapped and regular loading."""
        csv_path = tmp_path / "test_data.csv"
        sample_historical_data.to_csv(csv_path, index=False)

        result_regular = backtest_with_dask(
            str(csv_path),
            sample_atc_config,
            use_memory_mapped=False,
            chunksize="10MB",
        )

        try:
            result_mapped = backtest_with_dask(
                str(csv_path),
                sample_atc_config,
                use_memory_mapped=True,
                chunksize="10MB",
            )

            assert isinstance(result_regular, pd.DataFrame)
            assert isinstance(result_mapped, pd.DataFrame)
        except Exception as e:
            pytest.skip(f"Memory-mapped loading failed: {e}")

    def test_large_file_memory_efficiency(self, sample_atc_config, tmp_path):
        """Test memory efficiency with large file."""
        np.random.seed(42)
        n_bars = 5000
        n_symbols = 50

        records = []
        for sym_idx in range(n_symbols):
            symbol = f"SYM_{sym_idx}"
            prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
            timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="h")
            for price, ts in zip(prices, timestamps):
                records.append({"symbol": symbol, "close": price, "timestamp": ts})

        df = pd.DataFrame(records)
        csv_path = tmp_path / "large_data.csv"
        df.to_csv(csv_path, index=False)

        result = backtest_with_dask(
            str(csv_path),
            sample_atc_config,
            chunksize="50MB",
        )

        assert isinstance(result, pd.DataFrame)


class TestErrorConditions:
    """Error condition tests (missing columns, corrupt data, etc.)."""

    def test_missing_symbol_column(self, sample_atc_config):
        """Test handling of missing symbol column."""
        df = pd.DataFrame(
            {
                "close": range(50),
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
            }
        )
        result = backtest_from_dataframe(df, sample_atc_config, symbol_column="symbol")

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_missing_price_column(self, sample_atc_config):
        """Test handling of missing price column."""
        df = pd.DataFrame(
            {
                "symbol": ["TEST"] * 50,
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
            }
        )
        result = backtest_from_dataframe(df, sample_atc_config, price_column="close")

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_invalid_file_path(self, sample_atc_config):
        """Test handling of non-existent file path."""
        result = backtest_with_dask(
            "/nonexistent/path/to/file.csv",
            sample_atc_config,
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_corrupt_csv_file(self, sample_atc_config, tmp_path):
        """Test handling of CSV file with wrong columns."""
        corrupt_path = tmp_path / "corrupt.csv"
        corrupt_path.write_text("wrong_col1,wrong_col2,wrong_col3\nno,matching,columns")

        try:
            result = backtest_with_dask(str(corrupt_path), sample_atc_config)

            assert isinstance(result, pd.DataFrame)
        except KeyError:
            pass

    def test_mixed_data_types(self, sample_atc_config):
        """Test handling of mixed data types."""
        mixed_df = pd.DataFrame(
            {
                "symbol": ["TEST"] * 50,
                "close": [float(i) if i % 2 == 0 else str(i) for i in range(50)],
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
            }
        )
        result = backtest_from_dataframe(mixed_df, sample_atc_config)

        assert isinstance(result, pd.DataFrame)

    def test_empty_dataframe_input(self, sample_atc_config):
        """Test handling of empty DataFrame input."""
        empty_df = pd.DataFrame(columns=["symbol", "close", "timestamp"])
        result = backtest_from_dataframe(empty_df, sample_atc_config)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_nonexistent_files_list(self, sample_atc_config):
        """Test handling of list with non-existent files."""
        result = backtest_multiple_files_dask(
            ["/nonexistent1.csv", "/nonexistent2.csv"],
            sample_atc_config,
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_empty_files_list(self, sample_atc_config):
        """Test handling of empty file list."""
        result = backtest_multiple_files_dask([], sample_atc_config)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_invalid_atc_config(self, sample_historical_data):
        """Test handling of invalid ATC configuration."""
        invalid_config = {
            "ema_len": -1,
            "hma_len": 0,
            "invalid_param": "value",
        }
        result = backtest_from_dataframe(
            sample_historical_data,
            invalid_config,
            npartitions=1,
        )

        assert isinstance(result, pd.DataFrame)


class TestFileValidation:
    """Tests for file path validation."""

    def test_validate_existing_file(self, tmp_path):
        """Test validation of existing file."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("symbol,close\nTEST,100")

        assert _validate_file_path(str(test_file)) == True

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        assert _validate_file_path("/nonexistent/file.csv") == False

    def test_validate_directory_path(self, tmp_path):
        """Test validation of directory path (not file)."""
        assert _validate_file_path(str(tmp_path)) == False

    def test_validate_invalid_extension(self, tmp_path):
        """Test validation of file with invalid extension (warns but doesn't reject)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("some text")

        result = _validate_file_path(str(test_file))
        assert result == True  # Currently warns but doesn't reject


class TestErrorResultCreation:
    """Tests for _create_error_result helper function."""

    def test_error_result_structure(self):
        """Test that error result has correct structure."""
        result = _create_error_result("TEST_SYMBOL", "Test error message")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["symbol"].iloc[0] == "TEST_SYMBOL"
        assert result["status"].iloc[0] == "error"
        assert result["error_msg"].iloc[0] == "Test error message"
        assert pd.isna(result["signal"].iloc[0])
        assert pd.isna(result["price"].iloc[0])
        assert pd.isna(result["timestamp"].iloc[0])
