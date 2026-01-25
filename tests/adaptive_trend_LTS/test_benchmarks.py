import time

import numpy as np
import pandas as pd
import pytest

try:
    from modules.adaptive_trend_LTS.benchmarks.benchmark_dask_memory import (
        benchmark_memory_usage,
        generate_test_data,
        get_memory_usage_mb,
    )
    from modules.adaptive_trend_LTS.benchmarks.benchmark_dask_speed import (
        benchmark_speed,
        benchmark_speed_quick,
    )
except ImportError:
    pytest.skip("Benchmark modules not available", allow_module_level=True)


@pytest.fixture
def sample_config():
    """Default ATC config for benchmarking."""
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


def test_generate_test_data_basic():
    """Test basic test data generation."""
    data = generate_test_data(n_symbols=10, n_bars=1500)

    assert isinstance(data, dict)
    assert len(data) == 10
    assert all(isinstance(v, pd.Series) for v in data.values())
    assert all(len(v) == 1500 for v in data.values())


def test_generate_test_data_reproducibility():
    """Test that test data generation is reproducible."""
    data1 = generate_test_data(n_symbols=5, n_bars=100)
    data2 = generate_test_data(n_symbols=5, n_bars=100)

    assert data1.keys() == data2.keys()
    for symbol in data1:
        pd.testing.assert_series_equal(data1[symbol], data2[symbol])


def test_generate_test_data_sizes():
    """Test test data generation with different sizes."""
    data_small = generate_test_data(n_symbols=1, n_bars=100)
    data_large = generate_test_data(n_symbols=50, n_bars=1500)

    assert len(data_small) == 1
    assert len(data_large) == 50


def test_get_memory_usage_mb():
    """Test memory usage measurement."""
    mem = get_memory_usage_mb()

    assert isinstance(mem, float)
    assert mem >= 0


def test_benchmark_memory_usage_small(sample_config):
    """Test memory benchmark with small dataset."""
    results = benchmark_memory_usage(
        dataset_sizes=[10, 20],
        config=sample_config,
    )

    assert isinstance(results, list)


def test_benchmark_memory_usage_empty(sample_config):
    """Test memory benchmark with empty dataset."""
    results = benchmark_memory_usage(
        dataset_sizes=[],
        config=sample_config,
    )

    assert results == []


def test_benchmark_memory_usage_config(sample_config):
    """Test memory benchmark with custom config."""
    custom_config = sample_config.copy()
    custom_config["ema_len"] = 10

    results = benchmark_memory_usage(
        dataset_sizes=[10],
        config=custom_config,
    )

    assert isinstance(results, list)


def test_benchmark_speed_small(sample_config):
    """Test speed benchmark with small dataset."""
    results = benchmark_speed(
        dataset_sizes=[10, 20],
        config=sample_config,
    )

    assert isinstance(results, list)


def test_benchmark_speed_empty(sample_config):
    """Test speed benchmark with empty dataset."""
    results = benchmark_speed(
        dataset_sizes=[],
        config=sample_config,
    )

    assert results == []


def test_benchmark_speed_quick(sample_config):
    """Test quick speed benchmark."""
    benchmark_speed_quick(n_symbols=10, n_runs=1)


def test_benchmark_speed_quick_multiple_runs(sample_config):
    """Test quick speed benchmark with multiple runs."""
    start_time = time.time()
    benchmark_speed_quick(n_symbols=5, n_runs=2)
    duration = time.time() - start_time

    assert duration < 60


def test_benchmark_speed_quick_custom_config(sample_config):
    """Test quick benchmark with custom config."""
    custom_config = sample_config.copy()
    custom_config["ema_len"] = 10

    results = benchmark_speed_quick(n_symbols=5, n_runs=1)

    assert results is None


def test_benchmark_memory_dask_function(sample_config):
    """Test Dask memory benchmarking function."""
    from modules.adaptive_trend_LTS.benchmarks.benchmark_dask_memory import (
        benchmark_memory_dask,
    )

    data = generate_test_data(n_symbols=5, n_bars=1000)
    mem_delta, count = benchmark_memory_dask(data, sample_config, npartitions=2)

    assert isinstance(mem_delta, float)
    assert isinstance(count, int)
    assert count >= 0


def test_benchmark_memory_rust_function(sample_config):
    """Test Rust memory benchmarking function."""
    from modules.adaptive_trend_LTS.benchmarks.benchmark_dask_memory import (
        benchmark_memory_rust,
    )

    data = generate_test_data(n_symbols=5, n_bars=1000)
    mem_delta, count = benchmark_memory_rust(data, sample_config)

    assert isinstance(mem_delta, float)
    assert isinstance(count, int)
    assert count >= 0


def test_benchmark_speed_dask_function(sample_config):
    """Test Dask speed benchmarking function."""
    from modules.adaptive_trend_LTS.benchmarks.benchmark_dask_speed import (
        benchmark_dask,
    )

    data = generate_test_data(n_symbols=5, n_bars=1000)
    duration, count = benchmark_dask(data, sample_config, npartitions=2)

    assert isinstance(duration, float)
    assert isinstance(count, int)
    assert count >= 0
    assert duration >= 0


def test_benchmark_speed_rust_function(sample_config):
    """Test Rust speed benchmarking function."""
    from modules.adaptive_trend_LTS.benchmarks.benchmark_dask_speed import (
        benchmark_rust,
    )

    data = generate_test_data(n_symbols=5, n_bars=1000)
    duration, count = benchmark_rust(data, sample_config)

    assert isinstance(duration, float)
    assert isinstance(count, int)
    assert count >= 0
    assert duration >= 0


def test_benchmark_speed_rust_dask_hybrid_function(sample_config):
    """Test Rust-Dask hybrid speed benchmarking function."""
    from modules.adaptive_trend_LTS.benchmarks.benchmark_dask_speed import (
        benchmark_rust_dask_hybrid,
    )

    data = generate_test_data(n_symbols=5, n_bars=1000)
    duration, count = benchmark_rust_dask_hybrid(data, sample_config, npartitions=2)

    assert isinstance(duration, float)
    assert isinstance(count, int)
    assert count >= 0
    assert duration >= 0


def test_benchmark_integration_memory_speed(sample_config):
    """Test that both memory and speed benchmarks run."""
    data = generate_test_data(n_symbols=10, n_bars=1000)

    from modules.adaptive_trend_LTS.benchmarks.benchmark_dask_memory import (
        benchmark_memory_dask,
    )
    from modules.adaptive_trend_LTS.benchmarks.benchmark_dask_speed import (
        benchmark_dask,
    )

    mem_delta, mem_count = benchmark_memory_dask(data, sample_config, npartitions=2)
    duration, speed_count = benchmark_dask(data, sample_config, npartitions=2)

    assert mem_count == speed_count
    assert mem_count == 10
