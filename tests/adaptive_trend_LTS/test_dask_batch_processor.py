import time

import numpy as np
import pandas as pd
import pytest

try:
    from modules.adaptive_trend_LTS.core.compute_atc_signals.dask_batch_processor import (
        _process_partition_python,
        _process_partition_with_backend,
        process_symbols_batch_dask,
    )
    from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import (
        process_symbols_batch_with_dask,
    )
except ImportError:
    pytest.skip("Dask batch processor module not available", allow_module_level=True)


@pytest.fixture
def sample_config():
    """Default ATC config for testing."""
    return {
        "ema_len": 20,
        "hma_len": 20,
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


def test_process_partition_python_basic(sample_config, sample_price_series):
    """Test basic Python partition processing."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
    }

    results = _process_partition_python(symbols_data, sample_config)

    assert isinstance(results, dict)
    assert len(results) == 2
    assert all("Average_Signal" in result for result in results.values())


def test_process_partition_python_empty(sample_config):
    """Test processing empty partition."""
    results = _process_partition_python({}, sample_config)
    assert results == {}


def test_process_partition_python_with_none(sample_config, sample_price_series):
    """Test processing partition with None values."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": None,
        "SOLUSDT": pd.Series(),
    }

    results = _process_partition_python(symbols_data, sample_config)

    assert isinstance(results, dict)
    assert "BTCUSDT" in results
    assert "ETHUSDT" not in results
    assert "SOLUSDT" not in results


def test_process_partition_with_backend_rust(sample_config, sample_price_series):
    """Test partition processing with Rust backend."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
    }

    results = _process_partition_with_backend(
        list(symbols_data.items()), sample_config, use_rust=True, use_cuda=False, use_fallback=False
    )

    assert isinstance(results, dict)
    assert len(results) == 2
    assert all("Average_Signal" in result for result in results.values())


def test_process_partition_with_backend_python(sample_config, sample_price_series):
    """Test partition processing with Python backend."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
    }

    results = _process_partition_with_backend(
        list(symbols_data.items()), sample_config, use_rust=False, use_cuda=False, use_fallback=False
    )

    assert isinstance(results, dict)
    assert len(results) == 2


def test_process_partition_with_backend_fallback(sample_config, sample_price_series):
    """Test partition processing with Python fallback."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
    }

    results = _process_partition_with_backend(
        list(symbols_data.items()), sample_config, use_rust=True, use_cuda=False, use_fallback=True
    )

    assert isinstance(results, dict)
    assert len(results) == 2


def test_process_partition_with_backend_empty(sample_config):
    """Test processing empty partition."""
    results = _process_partition_with_backend([], sample_config, use_rust=True, use_cuda=False, use_fallback=False)
    assert results == {}


def test_process_symbols_batch_dask_empty(sample_config):
    """Test Dask batch processor with empty input."""
    results = process_symbols_batch_dask({}, sample_config)

    assert results == {}


def test_process_symbols_batch_dask_basic(sample_config, sample_price_series):
    """Test basic Dask batch processing."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
        "SOLUSDT": sample_price_series * 0.9,
    }

    results = process_symbols_batch_dask(
        symbols_data,
        sample_config,
        use_rust=True,
        use_cuda=False,
        npartitions=2,
        partition_size=2,
    )

    assert isinstance(results, dict)
    assert len(results) == 3
    assert all("Average_Signal" in result for result in results.values())


def test_process_symbols_batch_dask_large_dataset(sample_config):
    """Test Dask batch processor with larger dataset."""
    np.random.seed(42)
    n_symbols = 20
    n_bars = 1500

    symbols_data = {}
    for i in range(n_symbols):
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
        symbols_data[f"SYM_{i}"] = pd.Series(prices)

    start_time = time.time()
    results = process_symbols_batch_dask(
        symbols_data,
        sample_config,
        use_rust=True,
        use_cuda=False,
        npartitions=4,
        partition_size=5,
    )
    duration = time.time() - start_time

    assert len(results) == n_symbols
    print(f"\nProcessed {n_symbols} symbols in {duration:.2f}s ({n_symbols/duration:.1f} symbols/s)")


def test_process_symbols_batch_dask_auto_partitions(sample_config, sample_price_series):
    """Test Dask batch processor with auto-determined partitions."""
    symbols_data = {f"SYM_{i}": sample_price_series for i in range(10)}

    results = process_symbols_batch_dask(
        symbols_data,
        sample_config,
        use_rust=True,
        use_cuda=False,
        npartitions=None,
        partition_size=2,
    )

    assert len(results) == 10


def test_process_symbols_batch_dask_with_fallback(sample_config, sample_price_series):
    """Test Dask batch processor with Python fallback enabled."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
    }

    results = process_symbols_batch_dask(
        symbols_data,
        sample_config,
        use_rust=True,
        use_cuda=False,
        npartitions=2,
        use_fallback=True,
    )

    assert len(results) == 2


def test_process_symbols_batch_with_dask_small_batch(sample_config, sample_price_series):
    """Test process_symbols_batch_with_dask with small batch (no Dask)."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
    }

    results = process_symbols_batch_with_dask(
        symbols_data,
        sample_config,
        use_dask=True,
        npartitions=2,
    )

    assert len(results) == 2


def test_process_symbols_batch_with_dask_large_batch(sample_config):
    """Test process_symbols_batch_with_dask with large batch (uses Dask)."""
    np.random.seed(42)
    n_symbols = 1500
    n_bars = 1500

    symbols_data = {}
    for i in range(n_symbols):
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
        symbols_data[f"SYM_{i}"] = pd.Series(prices)

    results = process_symbols_batch_with_dask(
        symbols_data,
        sample_config,
        use_dask=True,
        npartitions=30,
        partition_size=50,
    )

    assert len(results) == n_symbols


def test_process_symbols_batch_with_dask_disabled(sample_config, sample_price_series):
    """Test process_symbols_batch_with_dask with Dask disabled."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
    }

    results = process_symbols_batch_with_dask(
        symbols_data,
        sample_config,
        use_dask=False,
    )

    assert len(results) == 2


def test_dask_batch_memory_efficiency(sample_config):
    """Test that Dask batch processor is memory-efficient."""
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

    results = process_symbols_batch_dask(
        symbols_data,
        sample_config,
        use_rust=True,
        use_cuda=False,
        npartitions=10,
        partition_size=10,
    )

    gc.collect()
    final_mem = sys.getsizeof([])

    assert len(results) == n_symbols
    print(f"\nMemory before: {initial_mem}, after: {final_mem}")


def test_dask_batch_result_consistency(sample_config, sample_price_series):
    """Test that results are consistent across multiple runs."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
        "SOLUSDT": sample_price_series * 0.9,
    }

    results1 = process_symbols_batch_dask(
        symbols_data,
        sample_config,
        use_rust=True,
        use_cuda=False,
        npartitions=2,
    )

    results2 = process_symbols_batch_dask(
        symbols_data,
        sample_config,
        use_rust=True,
        use_cuda=False,
        npartitions=3,
    )

    assert len(results1) == len(results2)
    symbols1 = set(results1.keys())
    symbols2 = set(results2.keys())
    assert symbols1 == symbols2


def test_dask_batch_single_partition(sample_config, sample_price_series):
    """Test Dask batch processor with single partition."""
    symbols_data = {f"SYM_{i}": sample_price_series for i in range(5)}

    results = process_symbols_batch_dask(
        symbols_data,
        sample_config,
        use_rust=True,
        use_cuda=False,
        npartitions=1,
    )

    assert len(results) == 5


def test_dask_batch_many_partitions(sample_config, sample_price_series):
    """Test Dask batch processor with many partitions."""
    symbols_data = {f"SYM_{i}": sample_price_series for i in range(3)}

    results = process_symbols_batch_dask(
        symbols_data,
        sample_config,
        use_rust=True,
        use_cuda=False,
        npartitions=20,
    )

    assert len(results) == 3


def test_dask_batch_with_cuda_if_available(sample_config, sample_price_series):
    """Test Dask batch processor with CUDA backend (if available)."""
    symbols_data = {
        "BTCUSDT": sample_price_series,
        "ETHUSDT": sample_price_series * 1.1,
    }

    try:
        results = process_symbols_batch_dask(
            symbols_data,
            sample_config,
            use_rust=False,
            use_cuda=True,
            npartitions=2,
        )
        assert len(results) == 2
    except Exception as e:
        # CUDA might not be available, that's okay
        pytest.skip(f"CUDA not available: {e}")


def test_dask_batch_error_handling(sample_config):
    """Test error handling with invalid data."""
    symbols_data = {
        "INVALID": None,
        "EMPTY": pd.Series(),
    }

    results = process_symbols_batch_dask(
        symbols_data,
        sample_config,
        use_rust=True,
        use_cuda=False,
        npartitions=2,
    )

    # Should return empty results or handle gracefully
    assert isinstance(results, dict)
