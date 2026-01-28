"""Benchmark speed of Dask vs Rust vs Hybrid."""

import time
from typing import Dict

import numpy as np
import pandas as pd

try:
    from modules.adaptive_trend_LTS.core.compute_atc_signals.batch_processor import (
        process_symbols_batch_rust,
    )
    from modules.adaptive_trend_LTS.core.compute_atc_signals.dask_batch_processor import (
        process_symbols_batch_dask,
    )
    from modules.adaptive_trend_LTS.core.compute_atc_signals.rust_dask_bridge import (
        process_symbols_rust_dask,
    )
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")


def generate_test_data(n_symbols: int, n_bars: int = 1500) -> Dict[str, pd.Series]:
    """Generate test data for benchmarking."""
    np.random.seed(42)

    symbols_data = {}
    for i in range(n_symbols):
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_bars))
        symbols_data[f"SYM_{i}"] = pd.Series(prices)

    return symbols_data


def benchmark_dask(
    symbols_data: Dict[str, pd.Series],
    config: dict,
    npartitions: int,
) -> tuple:
    """Benchmark Dask implementation speed."""
    start_time = time.time()

    results = process_symbols_batch_dask(
        symbols_data,
        config,
        use_rust=False,
        use_cuda=False,
        npartitions=npartitions,
        partition_size=len(symbols_data) // npartitions,
    )

    duration = time.time() - start_time
    return duration, len(results)


def benchmark_rust(
    symbols_data: Dict[str, pd.Series],
    config: dict,
) -> tuple:
    """Benchmark Rust implementation speed."""
    start_time = time.time()

    results = process_symbols_batch_rust(symbols_data, config)

    duration = time.time() - start_time
    return duration, len(results)


def benchmark_rust_dask_hybrid(
    symbols_data: Dict[str, pd.Series],
    config: dict,
    npartitions: int,
) -> tuple:
    """Benchmark Rust-Dask hybrid implementation speed."""
    start_time = time.time()

    results = process_symbols_rust_dask(
        symbols_data,
        config,
        use_cuda=False,
        npartitions=npartitions,
        partition_size=len(symbols_data) // npartitions,
    )

    duration = time.time() - start_time
    return duration, len(results)


def benchmark_speed(
    dataset_sizes: list[int] = [100, 500, 1000],
    config: dict = None,
) -> list:
    """Compare processing speed between implementations.

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

    print("=" * 100)
    print("SPEED BENCHMARK: Dask vs Rust vs Rust-Dask Hybrid")
    print("=" * 100)

    results = []

    for n_symbols in dataset_sizes:
        print(f"\n--- Testing with {n_symbols} symbols ---")

        symbols_data = generate_test_data(n_symbols)

        print(f"  Dataset size: {sum(len(s) for s in symbols_data.values()):,} data points")

        npartitions = max(1, n_symbols // 50)

        print(f"  Benchmarking Dask (npartitions={npartitions})...")
        dask_time, dask_count = benchmark_dask(symbols_data, config, npartitions)
        dask_throughput = dask_count / dask_time if dask_time > 0 else 0
        print(f"    Time: {dask_time:.2f}s")
        print(f"    Symbols: {dask_count}")
        print(f"    Throughput: {dask_throughput:.1f} symbols/s")

        print("  Benchmarking Rust...")
        try:
            rust_time, rust_count = benchmark_rust(symbols_data, config)
            rust_throughput = rust_count / rust_time if rust_time > 0 else 0
            print(f"    Time: {rust_time:.2f}s")
            print(f"    Symbols: {rust_count}")
            print(f"    Throughput: {rust_throughput:.1f} symbols/s")
        except Exception as e:
            print(f"    Rust not available: {e}")
            rust_time, rust_count = None, 0

        print("  Benchmarking Rust-Dask Hybrid...")
        try:
            hybrid_time, hybrid_count = benchmark_rust_dask_hybrid(symbols_data, config, npartitions)
            hybrid_throughput = hybrid_count / hybrid_time if hybrid_time > 0 else 0
            print(f"    Time: {hybrid_time:.2f}s")
            print(f"    Symbols: {hybrid_count}")
            print(f"    Throughput: {hybrid_throughput:.1f} symbols/s")
        except Exception as e:
            print(f"    Rust-Dask not available: {e}")
            hybrid_time, hybrid_count = None, 0

        if rust_time is not None and dask_time > 0:
            speedup = rust_time / dask_time
            slowdown = (dask_time / rust_time - 1) * 100
            print(
                f"    Dask vs Rust: {speedup:.2f}x ({slowdown:+.1f}% {'slower' if dask_time > rust_time else 'faster'})"
            )

        if hybrid_time is not None and dask_time > 0:
            speedup = dask_time / hybrid_time
            slowdown = (dask_time / hybrid_time - 1) * 100
            print(
                f"    Dask vs Hybrid: {speedup:.2f}x ({slowdown:+.1f}% {'slower' if dask_time > hybrid_time else 'faster'})"
            )

        if rust_time is not None and hybrid_time is not None:
            speedup = hybrid_time / rust_time
            slowdown = (hybrid_time / rust_time - 1) * 100
            print(
                f"    Hybrid vs Rust: {speedup:.2f}x ({slowdown:+.1f}% {'slower' if hybrid_time > rust_time else 'faster'})"
            )

        results.append(
            {
                "n_symbols": n_symbols,
                "dask_time_s": dask_time,
                "dask_throughput_sps": dask_throughput,
                "rust_time_s": rust_time,
                "rust_throughput_sps": rust_throughput if rust_time else None,
                "hybrid_time_s": hybrid_time,
                "hybrid_throughput_sps": hybrid_throughput if hybrid_time else None,
            }
        )

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(
        f"{'Symbols':>10} {'Dask (s)':>12} {'Dask (/s)':>12} "
        f"{'Rust (s)':>12} {'Rust (/s)':>12} {'Hybrid (s)':>12} {'Hybrid (/s)':>12}"
    )
    print("-" * 100)
    for result in results:
        n = result["n_symbols"]
        dask_t = result["dask_time_s"]
        dask_tp = result["dask_throughput_sps"]
        rust_t = result["rust_time_s"] or 0
        rust_tp = result["rust_throughput_sps"] or 0
        hybrid_t = result["hybrid_time_s"] or 0
        hybrid_tp = result["hybrid_throughput_sps"] or 0

        rust_t_str = f"{rust_t:>12.2f}" if rust_t else f"{'N/A':>12}"
        rust_tp_str = f"{rust_tp:>12.1f}" if rust_tp else f"{'N/A':>12}"
        hybrid_t_str = f"{hybrid_t:>12.2f}" if hybrid_t else f"{'N/A':>12}"
        hybrid_tp_str = f"{hybrid_tp:>12.1f}" if hybrid_tp else f"{'N/A':>12}"
        print(f"{n:>10} {dask_t:>12.2f} {dask_tp:>12.1f} {rust_t_str} {rust_tp_str} {hybrid_t_str} {hybrid_tp_str}")

    return results


def benchmark_speed_quick(
    n_symbols: int = 100,
    n_runs: int = 3,
) -> None:
    """Quick speed benchmark for single dataset size with multiple runs."""
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

    print(f"\nQuick benchmark: {n_symbols} symbols x {n_runs} runs")
    print("=" * 80)

    symbols_data = generate_test_data(n_symbols)
    npartitions = max(1, n_symbols // 50)

    for run in range(1, n_runs + 1):
        print(f"\nRun {run}/{n_runs}:")

        print("  Dask...")
        dask_time, _ = benchmark_dask(symbols_data, config, npartitions)
        print(f"    Time: {dask_time:.2f}s")

        print("  Rust...")
        try:
            rust_time, _ = benchmark_rust(symbols_data, config)
            print(f"    Time: {rust_time:.2f}s")
        except Exception:
            rust_time = None

        print("  Hybrid...")
        try:
            hybrid_time, _ = benchmark_rust_dask_hybrid(symbols_data, config, npartitions)
            print(f"    Time: {hybrid_time:.2f}s")
        except Exception:
            hybrid_time = None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        benchmark_speed_quick(n_symbols=100, n_runs=3)
    else:
        benchmark_speed()
