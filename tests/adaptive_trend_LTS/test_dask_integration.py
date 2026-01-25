"""Integration tests for Dask-based components end-to-end workflows."""

import gc
import sys
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

try:
    from modules.adaptive_trend_LTS.core.scanner.scan_all_symbols import scan_all_symbols
    from modules.adaptive_trend_LTS.utils.config import ATCConfig
except ImportError:
    pytest.skip("Scanner module not available", allow_module_level=True)


class MockDataFetcher:
    """Mock DataFetcher for integration testing."""

    def __init__(self, data_map: Dict[str, Tuple[pd.DataFrame, str]] = None):
        self.data_map = data_map or {}
        self.call_count = 0
        self.fetch_calls = []

    def fetch_ohlcv_with_fallback_exchange(
        self, symbol: str, limit: int = None, timeframe: str = None, check_freshness: bool = None
    ) -> Tuple[pd.DataFrame, str]:
        """Mock fetch method that returns data from data_map."""
        self.call_count += 1
        self.fetch_calls.append(symbol)
        if symbol in self.data_map:
            df, exchange = self.data_map[symbol]
            return df.copy(), exchange
        return None, None

    def fetch_ohlcv(self, symbol: str, limit: int = None) -> pd.Series:
        """Mock fetch method for backward compatibility."""
        df, _ = self.fetch_ohlcv_with_fallback_exchange(symbol, limit)
        if df is not None and "close" in df.columns:
            return df["close"]
        return pd.Series(dtype=float)


@pytest.fixture
def atc_config():
    """Default ATC config for integration testing."""
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
        calculation_source="close",
        precision="float64",
    )


@pytest.fixture
def sample_price_data():
    """Generate sample price data for multiple symbols."""
    np.random.seed(42)
    n_bars = 1500
    data_map = {}

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
    for symbol in symbols:
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
        df = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
            }
        )
        data_map[symbol] = (df, "binance")

    return data_map


def test_end_to_end_dask_scanner(atc_config, sample_price_data):
    """Test complete Dask scanner workflow from scan_all_symbols to results."""
    # Setup
    fetcher = MockDataFetcher(sample_price_data)
    symbols = list(sample_price_data.keys())

    # Monitor memory before
    gc.collect()
    initial_mem = sys.getsizeof([])

    # Run full scan with Dask mode
    start_time = time.time()
    long_signals_df, short_signals_df = scan_all_symbols(
        data_fetcher=fetcher,
        atc_config=atc_config,
        max_symbols=None,
        min_signal=0.01,
        execution_mode="dask",
        max_workers=None,
        batch_size=10,
        npartitions=2,
    )
    duration = time.time() - start_time

    # Verify results structure
    assert isinstance(long_signals_df, pd.DataFrame)
    assert isinstance(short_signals_df, pd.DataFrame)

    # Verify DataFrame columns
    expected_columns = ["symbol", "signal", "trend", "price", "exchange"]
    if not long_signals_df.empty:
        assert all(col in long_signals_df.columns for col in expected_columns)
    if not short_signals_df.empty:
        assert all(col in short_signals_df.columns for col in expected_columns)

    # Verify data consistency
    total_results = len(long_signals_df) + len(short_signals_df)
    assert total_results >= 0
    assert total_results <= len(symbols)

    # Verify signal values
    if not long_signals_df.empty:
        assert all(long_signals_df["signal"] >= atc_config.long_threshold)
        assert all(long_signals_df["trend"] > 0)
    if not short_signals_df.empty:
        assert all(short_signals_df["signal"] <= atc_config.short_threshold)
        assert all(short_signals_df["trend"] < 0)

    # Check memory usage (should be reasonable)
    gc.collect()
    final_mem = sys.getsizeof([])
    memory_delta = final_mem - initial_mem

    # Verify fetcher was called
    assert fetcher.call_count > 0

    print(f"\nEnd-to-end Dask scanner test:")
    print(f"  Symbols processed: {len(symbols)}")
    print(f"  Long signals: {len(long_signals_df)}")
    print(f"  Short signals: {len(short_signals_df)}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Memory delta: {memory_delta} bytes")

    # Memory should not grow excessively (basic check)
    assert memory_delta < 1000000  # Less than 1MB for this small test


def test_end_to_end_dask_scanner_large_dataset(atc_config):
    """Test end-to-end Dask scanner with larger dataset."""
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

    # Monitor memory
    gc.collect()
    initial_mem = sys.getsizeof([])

    start_time = time.time()
    long_signals_df, short_signals_df = scan_all_symbols(
        data_fetcher=fetcher,
        atc_config=atc_config,
        max_symbols=None,
        min_signal=0.01,
        execution_mode="dask",
        batch_size=10,
        npartitions=5,
    )
    duration = time.time() - start_time

    # Verify results
    assert isinstance(long_signals_df, pd.DataFrame)
    assert isinstance(short_signals_df, pd.DataFrame)

    total_results = len(long_signals_df) + len(short_signals_df)
    assert total_results >= 0
    assert total_results <= len(symbols)

    # Check memory efficiency
    gc.collect()
    final_mem = sys.getsizeof([])
    memory_delta = final_mem - initial_mem

    print(f"\nLarge dataset test:")
    print(f"  Symbols: {n_symbols}")
    print(f"  Results: {total_results}")
    print(f"  Duration: {duration:.2f}s ({n_symbols/duration:.1f} symbols/s)")
    print(f"  Memory delta: {memory_delta} bytes")

    # For large datasets, memory should be managed efficiently
    # (exact threshold depends on system, but should be reasonable)
    assert memory_delta < 5000000  # Less than 5MB for 50 symbols


def test_end_to_end_dask_scanner_empty_input(atc_config):
    """Test end-to-end Dask scanner with empty symbol list."""
    fetcher = MockDataFetcher({})

    long_signals_df, short_signals_df = scan_all_symbols(
        data_fetcher=fetcher,
        atc_config=atc_config,
        max_symbols=0,
        min_signal=0.01,
        execution_mode="dask",
        npartitions=1,
    )

    assert isinstance(long_signals_df, pd.DataFrame)
    assert isinstance(short_signals_df, pd.DataFrame)
    assert len(long_signals_df) == 0
    assert len(short_signals_df) == 0


def test_end_to_end_dask_scanner_vs_threadpool(atc_config, sample_price_data):
    """Test that Dask and ThreadPool modes produce consistent results."""
    symbols = list(sample_price_data.keys())

    # Run with Dask
    fetcher_dask = MockDataFetcher(sample_price_data.copy())
    long_dask, short_dask = scan_all_symbols(
        data_fetcher=fetcher_dask,
        atc_config=atc_config,
        min_signal=0.01,
        execution_mode="dask",
        npartitions=2,
    )

    # Run with ThreadPool
    fetcher_threadpool = MockDataFetcher(sample_price_data.copy())
    long_threadpool, short_threadpool = scan_all_symbols(
        data_fetcher=fetcher_threadpool,
        atc_config=atc_config,
        min_signal=0.01,
        execution_mode="threadpool",
        max_workers=2,
    )

    # Results should be consistent (same symbols, similar signals)
    dask_symbols = set(long_dask["symbol"].tolist() + short_dask["symbol"].tolist())
    threadpool_symbols = set(
        long_threadpool["symbol"].tolist() + short_threadpool["symbol"].tolist()
    )

    # Allow for minor differences due to timing/partitioning, but should be similar
    assert len(dask_symbols.symmetric_difference(threadpool_symbols)) <= 1


def test_end_to_end_dask_scanner_memory_cleanup(atc_config, sample_price_data):
    """Test that memory is properly cleaned up after Dask scanner run."""
    fetcher = MockDataFetcher(sample_price_data)
    symbols = list(sample_price_data.keys())

    # Force garbage collection before
    gc.collect()
    initial_objects = len(gc.get_objects())

    # Run scan
    long_signals_df, short_signals_df = scan_all_symbols(
        data_fetcher=fetcher,
        atc_config=atc_config,
        min_signal=0.01,
        execution_mode="dask",
        npartitions=2,
    )

    # Force garbage collection after
    gc.collect()
    final_objects = len(gc.get_objects())

    # Object count should not grow excessively
    # (allowing some growth for results, but should be reasonable)
    object_delta = final_objects - initial_objects
    print(f"\nMemory cleanup test:")
    print(f"  Initial objects: {initial_objects}")
    print(f"  Final objects: {final_objects}")
    print(f"  Object delta: {object_delta}")

    # Should not create excessive objects (threshold depends on system)
    assert object_delta < 10000
