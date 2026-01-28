"""Benchmark memory usage of Dask vs non-Dask implementations."""

import gc
import sys
from typing import Dict

import numpy as np
import pandas as pd

try:
    from modules.adaptive_trend_LTS.core.compute_atc_signals.dask_batch_processor import (
        process_symbols_batch_dask,
    )
    from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import (
        process_symbols_batch_rust,
    )
except ImportError:
    print("Warning: Some modules not available")

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available, memory tracking limited")


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    else:
        return sys.getsizeof([]) / 1024 / 1024


def generate_test_data(n_symbols: int, n_bars: int = 1500) -> Dict[str, pd.Series]:
    """Generate test data for benchmarking."""
    np.random.seed(42)

    symbols_data = {}
    for i in range(n_symbols):
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
        symbols_data[f"SYM_{i}"] = pd.Series(prices)

    return symbols_data


def benchmark_memory_dask(
    symbols_data: Dict[str, pd.Series],
    config: dict,
    npartitions: int,
) -> tuple:
    """Benchmark Dask implementation memory usage."""
    gc.collect()
    initial_memory = get_memory_usage_mb()

    results = process_symbols_batch_dask(
        symbols_data,
        config,
        use_rust=False,
        use_cuda=False,
        npartitions=npartitions,
        partition_size=len(symbols_data) // npartitions,
    )

    gc.collect()
    peak_memory = get_memory_usage_mb()
    memory_delta = peak_memory - initial_memory

    return memory_delta, len(results)


def benchmark_memory_rust(
    symbols_data: Dict[str, pd.Series],
    config: dict,
) -> tuple:
    """Benchmark Rust implementation memory usage."""
    gc.collect()
    initial_memory = get_memory_usage_mb()

    results = process_symbols_batch_rust(symbols_data, config)

    gc.collect()
    peak_memory = get_memory_usage_mb()
    memory_delta = peak_memory - initial_memory

    return memory_delta, len(results)


def benchmark_memory_usage(
    dataset_sizes: list[int] = [100, 500, 1000, 5000],
    config: dict = None,
) -> list:
    """Compare memory usage between implementations.

    Args:
        dataset_sizes: List of symbol counts to test
        config: ATC configuration
    """
    if config is None:
        config = {
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

    print("=" * 80)
    print("MEMORY USAGE BENCHMARK: Dask vs Rust")
    print("=" * 80)

    results = []

    for n_symbols in dataset_sizes:
        print(f"\n--- Testing with {n_symbols} symbols ---")

        symbols_data = generate_test_data(n_symbols)

        print(f"  Dataset size: {sum(len(s) for s in symbols_data.values()):,} data points")

        npartitions = max(1, n_symbols // 50)

        print(f"  Benchmarking Dask (npartitions={npartitions})...")
        dask_memory, dask_count = benchmark_memory_dask(symbols_data, config, npartitions)
        print(f"    Memory delta: {dask_memory:.2f} MB")
        print(f"    Symbols processed: {dask_count}")

        print(f"  Benchmarking Rust...")
        try:
            rust_memory, rust_count = benchmark_memory_rust(symbols_data, config)
            print(f"    Memory delta: {rust_memory:.2f} MB")
            print(f"    Symbols processed: {rust_count}")
        except Exception as e:
            print(f"    Rust not available: {e}")
            rust_memory, rust_count = None, 0

        if rust_memory is not None:
            memory_reduction = (1 - dask_memory / rust_memory) * 100
            print(f"    Memory reduction: {memory_reduction:+.1f}%")

        results.append(
            {
                "n_symbols": n_symbols,
                "dask_memory_mb": dask_memory,
                "rust_memory_mb": rust_memory,
                "memory_reduction_percent": ((1 - dask_memory / rust_memory) * 100 if rust_memory else None),
            }
        )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Symbols':>10} {'Dask (MB)':>12} {'Rust (MB)':>12} {'Reduction %':>12}")
    print("-" * 80)
    for result in results:
        dask_mem = result["dask_memory_mb"]
        rust_mem = result["rust_memory_mb"]
        reduction = result["memory_reduction_percent"]

        rust_str = f"{rust_mem:>12.2f}" if rust_mem is not None else f"{'N/A':>12}"
        reduction_str = f"{reduction:>12.1f}" if reduction is not None else f"{'N/A':>12}"
        print(f"{result['n_symbols']:>10} {dask_mem:>12.2f} {rust_str} {reduction_str}")

    return results


if __name__ == "__main__":
    benchmark_memory_usage()
