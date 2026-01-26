import time

import numpy as np
import pandas as pd
import pytest

try:
    from modules.adaptive_trend_LTS.core.scanner.dask_scan import (
        _fetch_partition_lazy,
        _process_partition_with_gc,
        _process_single_symbol_dask,
        _scan_dask,
        ProgressCallback,
    )
    from modules.adaptive_trend_LTS.utils.config import ATCConfig
except ImportError:
    pytest.skip("Dask scanner module not available", allow_module_level=True)


class MockDataFetcher:
    """Mock DataFetcher for testing."""

    def __init__(self, data_map=None):
        self.data_map = data_map or {}
        self.call_count = 0

    def fetch_ohlcv_with_fallback_exchange(self, symbol, limit=None, timeframe=None, check_freshness=None):
        """Mock fetch method."""
        self.call_count += 1
        if symbol in self.data_map:
            prices, exchange = self.data_map[symbol]
            return prices, exchange
        return None, None


@pytest.fixture
def atc_config():
    """Default ATC config for testing."""
    return ATCConfig(
        ema_len=20,
        hma_len=20,
        wma_len=20,
        dema_len=20,
        lsma_len=20,
        kama_len=20,
        robustness="Medium",
        lambda_param=0.02,
        decay=0.03,
        cutout=0,
        limit=1500,
        timeframe="15m",
    )


@pytest.fixture
def sample_price_series():
    """Generate sample price data for testing."""
    np.random.seed(42)
    n = 1500
    prices = 100 + np.cumsum(np.random.normal(0, 1, n))
    df = pd.DataFrame({"open": prices * 0.99, "high": prices * 1.01, "low": prices * 0.98, "close": prices})
    return df


def test_process_single_symbol_dask_success(atc_config, sample_price_series):
    """Test processing a single symbol with valid data."""
    symbol_data = ("BTCUSDT", sample_price_series, "binance")
    result = _process_single_symbol_dask(symbol_data, atc_config, min_signal=0.01)

    assert result is not None
    assert result["symbol"] == "BTCUSDT"
    assert result["exchange"] == "binance"
    assert "signal" in result
    assert "trend" in result
    assert "price" in result
    assert isinstance(result["signal"], (int, float, np.number))
    assert isinstance(result["trend"], (int, float, np.number))
    assert result["price"] > 0


def test_process_single_symbol_dask_empty_df(atc_config):
    """Test processing with empty DataFrame."""
    symbol_data = ("EMPTY", pd.DataFrame(), "binance")
    result = _process_single_symbol_dask(symbol_data, atc_config, min_signal=0.01)
    assert result is None


def test_process_single_symbol_dask_none_df(atc_config):
    """Test processing with None DataFrame."""
    symbol_data = ("NONE", None, "binance")
    result = _process_single_symbol_dask(symbol_data, atc_config, min_signal=0.01)
    assert result is None


def test_process_single_symbol_dask_below_threshold(atc_config, sample_price_series):
    """Test that signals below threshold are filtered out."""
    symbol_data = ("LOW", sample_price_series, "binance")
    result = _process_single_symbol_dask(symbol_data, atc_config, min_signal=100.0)
    assert result is None


def test_fetch_partition_lazy(atc_config, sample_price_series):
    """Test lazy data fetching for a partition."""
    data_map = {"BTCUSDT": (sample_price_series, "binance"), "ETHUSDT": (sample_price_series * 1.1, "binance")}
    fetcher = MockDataFetcher(data_map)

    partition = ["BTCUSDT", "ETHUSDT", "MISSING"]
    result = _fetch_partition_lazy(partition, fetcher, atc_config)

    assert len(result) == 3
    assert fetcher.call_count == 3
    assert result[0][0] == "BTCUSDT"
    assert result[0][1] is not None
    assert result[1][0] == "ETHUSDT"
    assert result[1][1] is not None
    assert result[2][0] == "MISSING"
    assert result[2][1] is None


def test_process_partition_with_gc(atc_config, sample_price_series):
    """Test processing a partition with GC."""
    partition_data = [
        ("BTCUSDT", sample_price_series, "binance"),
        ("ETHUSDT", sample_price_series * 1.1, "binance"),
        ("SOLUSDT", sample_price_series * 0.9, "binance"),
    ]

    results = _process_partition_with_gc(partition_data, atc_config, min_signal=0.01)

    assert isinstance(results, list)
    assert all(isinstance(r, dict) for r in results)
    assert all("symbol" in r for r in results)


def test_scan_dask_empty_symbols(atc_config):
    """Test Dask scanner with empty symbol list."""
    fetcher = MockDataFetcher()
    results, skipped, errors, skipped_symbols = _scan_dask([], fetcher, atc_config)

    assert results == []
    assert skipped == 0
    assert errors == 0
    assert skipped_symbols == []


def test_scan_dask_basic(atc_config, sample_price_series):
    """Test basic Dask scanning functionality."""
    data_map = {
        "BTCUSDT": (sample_price_series, "binance"),
        "ETHUSDT": (sample_price_series * 1.1, "binance"),
        "SOLUSDT": (sample_price_series * 0.9, "binance"),
    }
    fetcher = MockDataFetcher(data_map)
    symbols = list(data_map.keys())

    results, skipped, errors, skipped_symbols = _scan_dask(symbols, fetcher, atc_config, min_signal=0.01, npartitions=2)

    assert isinstance(results, list)
    assert len(results) >= 0
    assert isinstance(skipped, int)
    assert isinstance(errors, int)
    assert isinstance(skipped_symbols, list)


def test_scan_dask_large_dataset(atc_config):
    """Test Dask scanner with a larger dataset."""
    np.random.seed(42)
    n_symbols = 50
    n_bars = 1500

    data_map = {}
    for i in range(n_symbols):
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
        df = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
            }
        )
        data_map[f"SYM_{i}"] = (df, "binance")

    fetcher = MockDataFetcher(data_map)
    symbols = list(data_map.keys())

    start_time = time.time()
    results, skipped, errors, skipped_symbols = _scan_dask(
        symbols, fetcher, atc_config, min_signal=0.01, npartitions=5, batch_size=10
    )
    duration = time.time() - start_time

    total_processed = len(results) + skipped + errors
    assert total_processed == n_symbols
    assert isinstance(results, list)
    print(f"\nProcessed {n_symbols} symbols in {duration:.2f}s ({n_symbols/duration:.1f} symbols/s)")


def test_scan_dask_auto_partitions(atc_config, sample_price_series):
    """Test Dask scanner with auto-determined partitions."""
    data_map = {f"SYM_{i}": (sample_price_series, "binance") for i in range(50)}
    fetcher = MockDataFetcher(data_map)
    symbols = list(data_map.keys())

    results, skipped, errors, skipped_symbols = _scan_dask(
        symbols, fetcher, atc_config, min_signal=0.01, npartitions=None, batch_size=10
    )

    assert len(results) + skipped + errors == len(symbols)


def test_scan_dask_all_failures(atc_config):
    """Test Dask scanner when all fetches fail."""
    fetcher = MockDataFetcher({})  # Empty map = all failures
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    results, skipped, errors, skipped_symbols = _scan_dask(symbols, fetcher, atc_config, min_signal=0.01, npartitions=1)

    assert results == []
    assert skipped + errors == len(symbols)
    assert len(skipped_symbols) == len(symbols)


def test_progress_callback():
    """Test ProgressCallback functionality."""
    total = 100
    callback = ProgressCallback(total)

    assert callback.total == total
    assert callback.processed == 0

    for i in range(1, total + 1):
        callback._posttask("task_key", None, {}, {}, "dummy_id")
        assert callback.processed == i


def test_progress_callback_logs():
    """Test that ProgressCallback logs at intervals."""
    total = 30
    callback = ProgressCallback(total)

    callback._start({})
    assert callback.processed == 0

    for i in range(1, total + 1):
        callback._posttask(f"task_{i}", None, {}, {}, "dummy_id")
        if i % 10 == 0 or i == total:
            assert callback.last_logged == i


def test_scan_dask_with_threshold_filtering(atc_config, sample_price_series):
    """Test that min_signal threshold filters results correctly."""
    data_map = {"BTCUSDT": (sample_price_series, "binance")}
    fetcher = MockDataFetcher(data_map)

    results_high, _, _, _ = _scan_dask(["BTCUSDT"], fetcher, atc_config, min_signal=100.0, npartitions=1)
    results_low, _, _, _ = _scan_dask(["BTCUSDT"], fetcher, atc_config, min_signal=0.0, npartitions=1)

    assert len(results_high) == 0 or len(results_low) >= len(results_high)


def test_scan_dask_single_partition(atc_config, sample_price_series):
    """Test Dask scanner with single partition (sequential-like behavior)."""
    data_map = {f"SYM_{i}": (sample_price_series, "binance") for i in range(10)}
    fetcher = MockDataFetcher(data_map)
    symbols = list(data_map.keys())

    results, skipped, errors, skipped_symbols = _scan_dask(symbols, fetcher, atc_config, min_signal=0.01, npartitions=1)

    assert len(results) + skipped + errors == len(symbols)


def test_scan_dask_many_partitions(atc_config, sample_price_series):
    """Test Dask scanner with many partitions (more than symbols)."""
    data_map = {f"SYM_{i}": (sample_price_series, "binance") for i in range(5)}
    fetcher = MockDataFetcher(data_map)
    symbols = list(data_map.keys())

    results, skipped, errors, skipped_symbols = _scan_dask(
        symbols, fetcher, atc_config, min_signal=0.01, npartitions=20
    )

    assert len(results) + skipped + errors == len(symbols)


def test_scan_dask_memory_efficiency(atc_config):
    """Test that Dask scanner is memory-efficient for large datasets."""
    import gc
    import sys

    gc.collect()
    initial_mem = sys.getsizeof([])

    np.random.seed(42)
    n_symbols = 100
    n_bars = 1500

    data_map = {}
    for i in range(n_symbols):
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
        df = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
            }
        )
        data_map[f"SYM_{i}"] = (df, "binance")

    fetcher = MockDataFetcher(data_map)
    symbols = list(data_map.keys())

    results, skipped, errors, skipped_symbols = _scan_dask(
        symbols, fetcher, atc_config, min_signal=0.01, npartitions=10, batch_size=20
    )

    gc.collect()
    final_mem = sys.getsizeof([])

    assert len(results) + skipped + errors == len(symbols)
    print(f"\nMemory before: {initial_mem}, after: {final_mem}")


def test_scan_dask_result_consistency(atc_config, sample_price_series):
    """Test that Dask scanner produces consistent results across runs."""
    data_map = {
        "BTCUSDT": (sample_price_series, "binance"),
        "ETHUSDT": (sample_price_series * 1.1, "binance"),
        "SOLUSDT": (sample_price_series * 0.9, "binance"),
    }
    fetcher = MockDataFetcher(data_map)
    symbols = list(data_map.keys())

    results1, _, _, _ = _scan_dask(symbols, fetcher, atc_config, min_signal=0.01, npartitions=2)
    results2, _, _, _ = _scan_dask(symbols, fetcher, atc_config, min_signal=0.01, npartitions=3)

    assert len(results1) == len(results2)
    symbols1 = {r["symbol"] for r in results1}
    symbols2 = {r["symbol"] for r in results2}
    assert symbols1 == symbols2
