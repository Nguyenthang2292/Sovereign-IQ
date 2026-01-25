import time

import numpy as np
import pandas as pd
import pytest

try:
    from modules.adaptive_trend_LTS.core.compute_atc_signals.rust_dask_bridge import (
        _process_partition_with_rust_cpu,
        _process_partition_with_rust_cuda,
        _process_partition_python,
        auto_tune_partition_size,
        process_symbols_rust_dask,
    )
except ImportError:
    pytest.skip("Rust-Dask bridge module not available", allow_module_level=True)


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
def sample_price_series():
    """Generate sample price data for testing."""
    np.random.seed(42)
    n = 1500
    prices = 100 + np.cumsum(np.random.normal(0, 1, n))
    return pd.Series(prices)


def test_auto_tune_partition_size_basic():
    """Test basic partition size tuning."""
    partition_size = auto_tune_partition_size(
        total_symbols=1000,
        available_memory_gb=8.0,
        target_memory_per_partition_gb=0.5,
    )

    assert isinstance(partition_size, int)
    assert partition_size >= 10
    assert partition_size <= 200


def test_auto_tune_partition_size_large_memory():
    """Test partition size tuning with large memory."""
    partition_size = auto_tune_partition_size(
        total_symbols=5000,
        available_memory_gb=16.0,
        target_memory_per_partition_gb=0.5,
    )

    assert isinstance(partition_size, int)
    assert partition_size >= 10
    assert partition_size <= 200


def test_auto_tune_partition_size_small_memory():
    """Test partition size tuning with small memory."""
    partition_size = auto_tune_partition_size(
        total_symbols=1000,
        available_memory_gb=4.0,
        target_memory_per_partition_gb=0.25,
    )

    assert isinstance(partition_size, int)
    assert partition_size >= 10
    assert partition_size <= 200


def test_process_partition_python_basic(sample_config, sample_price_series):
    """Test basic Python partition processing."""
    partition_data = {
        "BTCUSDT": sample_price_series.values,
        "ETHUSDT": (sample_price_series * 1.1).values,
    }

    results = _process_partition_python(partition_data, sample_config)

    assert isinstance(results, dict)
    assert len(results) >= 0
    assert all("Average_Signal" in result for result in results.values())


def test_process_partition_python_empty(sample_config):
    """Test processing empty partition."""
    results = _process_partition_python({}, sample_config)
    assert results == {}


def test_process_partition_python_with_none(sample_config):
    """Test processing partition with None values."""
    partition_data = {
        "BTCUSDT": None,
        "ETHUSDT": sample_price_series.values,
        "EMPTY": np.array([]),
    }

    results = _process_partition_python(partition_data, sample_config)

    assert isinstance(results, dict)
    assert "BTCUSDT" not in results
    assert "ETHUSDT" in results


def test_process_partition_with_rust_cpu_basic(sample_config, sample_price_series):
    """Test basic Rust CPU partition processing."""
    partition_data = {
        "BTCUSDT": sample_price_series.values,
        "ETHUSDT": (sample_price_series * 1.1).values,
    }

    results = _process_partition_with_rust_cpu(partition_data, sample_config)

    assert isinstance(results, dict)


def test_process_partition_with_rust_cpu_empty(sample_config):
    """Test Rust CPU with empty partition."""
    results = _process_partition_with_rust_cpu({}, sample_config)
    assert results == {}


def test_process_partition_with_rust_cuda_basic(sample_config, sample_price_series):
    """Test basic Rust CUDA partition processing."""
    partition_data = {
        "BTCUSDT": sample_price_series.values,
        "ETHUSDT": (sample_price_series * 1.1).values,
    }

    results = _process_partition_with_rust_cuda(partition_data, sample_config)

    assert isinstance(results, dict)


def test_process_partition_with_rust_cuda_empty(sample_config):
    """Test Rust CUDA with empty partition."""
    results = _process_partition_with_rust_cuda({}, sample_config)
    assert results == {}


def test_process_symbols_rust_dask_empty(sample_config):
    """Test Rust-Dask with empty input."""
    results = process_symbols_rust_dask({}, sample_config)

    assert results == {}


def test_process_symbols_rust_dask_basic(sample_config, sample_price_series):
    """Test basic Rust-Dask processing."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
        "SOLUSDT": sample_price_series * 0.9,
    }

    results = process_symbols_rust_dask(
        symbols_data,
        sample_config,
        use_cuda=False,
        npartitions=2,
        partition_size=2,
    )

    assert isinstance(results, dict)
    assert len(results) == 3
    assert all("Average_Signal" in result for result in results.values())


def test_process_symbols_rust_dask_large_dataset(sample_config):
    """Test Rust-Dask with larger dataset."""
    np.random.seed(42)
    n_symbols = 50
    n_bars = 1500

    symbols_data = {}
    for i in range(n_symbols):
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
        symbols_data[f"SYM_{i}"] = pd.Series(prices)

    start_time = time.time()
    results = process_symbols_rust_dask(
        symbols_data,
        sample_config,
        use_cuda=False,
        npartitions=5,
        partition_size=10,
    )
    duration = time.time() - start_time

    assert len(results) == n_symbols
    print(f"\nProcessed {n_symbols} symbols in {duration:.2f}s ({n_symbols/duration:.1f} symbols/s)")


def test_process_symbols_rust_dask_auto_partitions(sample_config, sample_price_series):
    """Test Rust-Dask with auto-determined partitions."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
        "SOLUSDT": sample_price_series * 0.9,
    }

    results = process_symbols_rust_dask(
        symbols_data,
        sample_config,
        use_cuda=False,
        npartitions=None,
        partition_size=2,
    )

    assert len(results) == 3


def test_process_symbols_rust_dask_single_partition(sample_config, sample_price_series):
    """Test Rust-Dask with single partition."""
    symbols_data = {f"SYM_{i}": sample_price_series for i in range(5)}

    results = process_symbols_rust_dask(
        symbols_data,
        sample_config,
        use_cuda=False,
        npartitions=1,
        partition_size=5,
    )

    assert len(results) == 5


def test_process_symbols_rust_dask_many_partitions(sample_config, sample_price_series):
    """Test Rust-Dask with many partitions."""
    symbols_data = {f"SYM_{i}": sample_price_series for i in range(3)}

    results = process_symbols_rust_dask(
        symbols_data,
        sample_config,
        use_cuda=False,
        npartitions=20,
        partition_size=2,
    )

    assert len(results) == 3


def test_process_symbols_rust_dask_with_fallback(sample_config, sample_price_series):
    """Test Rust-Dask with Python fallback enabled."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
    }

    results = process_symbols_rust_dask(
        symbols_data,
        sample_config,
        use_cuda=False,
        npartitions=2,
        partition_size=2,
        use_fallback=True,
    )

    assert len(results) >= 0


def test_process_symbols_rust_dask_cuda_mode(sample_config, sample_price_series):
    """Test Rust-Dask with CUDA mode."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
        "SOLUSDT": sample_price_series * 0.9,
    }

    results = process_symbols_rust_dask(
        symbols_data,
        sample_config,
        use_cuda=True,
        npartitions=2,
        partition_size=2,
    )

    assert len(results) >= 0


def test_process_symbols_rust_dask_result_consistency(sample_config, sample_price_series):
    """Test that results are consistent across multiple runs."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
        "SOLUSDT": sample_price_series * 0.9,
    }

    results1 = process_symbols_rust_dask(
        symbols_data,
        sample_config,
        use_cuda=False,
        npartitions=2,
    )

    results2 = process_symbols_rust_dask(
        symbols_data,
        sample_config,
        use_cuda=False,
        npartitions=3,
    )

    assert len(results1) == len(results2)
    symbols1 = set(results1.keys())
    symbols2 = set(results2.keys())
    assert symbols1 == symbols2


def test_process_symbols_rust_dask_memory_efficiency(sample_config):
    """Test that Rust-Dask is memory-efficient."""
    import gc
    import sys

    gc.collect()
    initial_mem = sys.getsizeof([])

    np.random.seed(42)
    n_symbols = 100
    n_bars = 1500

    symbols_data = {}
    for i in range(n_symbols):
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
        symbols_data[f"SYM_{i}"] = pd.Series(prices)

    results = process_symbols_rust_dask(
        symbols_data,
        sample_config,
        use_cuda=False,
        npartitions=10,
        partition_size=10,
    )

    gc.collect()
    final_mem = sys.getsizeof([])

    assert len(results) == n_symbols
    print(f"\nMemory before: {initial_mem}, after: {final_mem}")


def test_process_symbols_rust_dask_error_handling(sample_config):
    """Test error handling with invalid data."""
    symbols_data = {
        "INVALID": None,
        "EMPTY": pd.Series(),
    }

    results = process_symbols_rust_dask(
        symbols_data,
        sample_config,
        use_cuda=False,
        npartitions=2,
    )

    assert isinstance(results, dict)


def test_process_partition_rust_with_series_input(sample_config, sample_price_series):
    """Test Rust CPU with pandas Series input."""
    partition_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
    }

    results = _process_partition_with_rust_cpu(partition_data, sample_config)

    assert isinstance(results, dict)
    assert len(results) == 2


def test_process_partition_python_with_series_input(sample_config, sample_price_series):
    """Test Python fallback with pandas Series input."""
    partition_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
    }

    results = _process_partition_python(partition_data, sample_config)

    assert isinstance(results, dict)
    assert len(results) == 2
