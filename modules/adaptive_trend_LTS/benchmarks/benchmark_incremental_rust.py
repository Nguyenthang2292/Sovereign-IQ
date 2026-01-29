"""Benchmark runner for Rust incremental ATC backend vs Python.

This module benchmarks performance improvement of Rust backend
compared to Python implementation for incremental ATC updates.
"""

import argparse
import time
from typing import Dict, Any

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.core.compute_atc_signals import IncrementalATC
from modules.adaptive_trend_LTS.core.incremental_backend import check_rust_available


def create_sample_config(use_rust: bool = True) -> Dict[str, Any]:
    """Create standard ATC configuration for benchmarking."""
    return {
        "ema_len": 28,
        "hma_len": 28,
        "wma_len": 28,
        "dema_len": 28,
        "lsma_len": 28,
        "kama_len": 28,
        "La": 0.02,
        "De": 0.03,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
        "use_rust_incremental": use_rust,
        "use_o1_mas": True,
    }


def benchmark_incremental_atc_rust(iterations: int = 1000) -> dict:
    """Benchmark IncrementalATC with Rust backend.

    Args:
        iterations: Number of update iterations to benchmark

    Returns:
        Dictionary with timing results
    """
    print(f"\nBenchmarking IncrementalATC with Rust backend (iterations={iterations})...")

    # Generate test data
    np.random.seed(42)
    init_prices = pd.Series(100 + np.random.randn(200) * 10)
    update_prices = pd.Series(100 + np.random.randn(iterations) * 10)

    config = create_sample_config(use_rust=True)
    atc = IncrementalATC(config)
    atc.initialize(init_prices)

    # Warmup
    for _ in range(10):
        atc.update(100.0)

    # Benchmark
    start = time.perf_counter()
    for price in update_prices:
        atc.update(price)
    rust_time = time.perf_counter() - start

    return {
        "backend": "Rust",
        "iterations": iterations,
        "time_ms": rust_time * 1000,
        "time_per_update_us": (rust_time / iterations) * 1_000_000,
    }


def benchmark_incremental_atc_python(iterations: int = 1000) -> dict:
    """Benchmark IncrementalATC with Python backend.

    Args:
        iterations: Number of update iterations to benchmark

    Returns:
        Dictionary with timing results
    """
    print(f"\nBenchmarking IncrementalATC with Python backend (iterations={iterations})...")

    # Generate test data
    np.random.seed(42)
    init_prices = pd.Series(100 + np.random.randn(200) * 10)
    update_prices = pd.Series(100 + np.random.randn(iterations) * 10)

    config = create_sample_config(use_rust=False)
    atc = IncrementalATC(config)
    atc.initialize(init_prices)

    # Warmup
    for _ in range(10):
        atc.update(100.0)

    # Benchmark
    start = time.perf_counter()
    for price in update_prices:
        atc.update(price)
    python_time = time.perf_counter() - start

    return {
        "backend": "Python",
        "iterations": iterations,
        "time_ms": python_time * 1000,
        "time_per_update_us": (python_time / iterations) * 1_000_000,
    }


def benchmark_consistency_check(iterations: int = 100) -> dict:
    """Check that Rust and Python produce identical results.

    Args:
        iterations: Number of updates to compare

    Returns:
        Dictionary with consistency results
    """
    print(f"\nChecking Rust vs Python consistency (iterations={iterations})...")

    # Generate test data
    np.random.seed(42)
    init_prices = pd.Series(100 + np.random.randn(200) * 10)
    update_prices = pd.Series(100 + np.random.randn(iterations) * 10)

    # Rust backend
    config_rust = create_sample_config(use_rust=True)
    atc_rust = IncrementalATC(config_rust)
    atc_rust.initialize(init_prices)

    signals_rust = []
    for price in update_prices:
        signal = atc_rust.update(price)
        signals_rust.append(signal)

    # Python backend
    config_python = create_sample_config(use_rust=False)
    atc_python = IncrementalATC(config_python)
    atc_python.initialize(init_prices)

    signals_python = []
    for price in update_prices:
        signal = atc_python.update(price)
        signals_python.append(signal)

    # Compare results
    max_diff = max(abs(sr - sp) for sr, sp in zip(signals_rust, signals_python))
    avg_diff = np.mean([abs(sr - sp) for sr, sp in zip(signals_rust, signals_python)])

    return {
        "iterations": iterations,
        "max_diff": max_diff,
        "avg_diff": avg_diff,
        "max_acceptable_diff": 1e-3,
        "consistent": max_diff < 1e-3,
    }


def benchmark_variable_iterations():
    """Benchmark with varying iteration counts."""
    print("\n" + "=" * 80)
    print("BENCHMARK: Variable Iterations")
    print("=" * 80)

    iteration_counts = [100, 1000, 10000]

    for iterations in iteration_counts:
        rust_result = benchmark_incremental_atc_rust(iterations)
        python_result = benchmark_incremental_atc_python(iterations)

        speedup = python_result["time_ms"] / rust_result["time_ms"] if rust_result["time_ms"] > 0 else 0

        print(f"\nIterations: {iterations}")
        print(f"  Rust:    {rust_result['time_ms']:.4f} ms ({rust_result['time_per_update_us']:.2f} μs/update)")
        print(f"  Python:  {python_result['time_ms']:.4f} ms ({python_result['time_per_update_us']:.2f} μs/update)")
        print(f"  Speedup: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Rust incremental ATC backend")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations to benchmark")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "rust", "python", "consistency"], help="Benchmark mode")
    args = parser.parse_args()

    print("=" * 80)
    print("Rust Incremental ATC Benchmark Suite")
    print("=" * 80)
    print(f"Iterations: {args.iterations}")
    print(f"Python: {__import__('sys').version}")
    print(f"NumPy: {np.__version__}")
    print(f"Rust Available: {check_rust_available()}")
    print("=" * 80)

    results = []

    if args.mode in ["all", "rust"]:
        if not check_rust_available():
            print("\nWARNING: Rust backend not available. Run 'maturin develop --release' first.")
        else:
            rust_result = benchmark_incremental_atc_rust(args.iterations)
            results.append(rust_result)

    if args.mode in ["all", "python"]:
        python_result = benchmark_incremental_atc_python(args.iterations)
        results.append(python_result)

    if args.mode == "consistency":
        consistency_result = benchmark_consistency_check(args.iterations)
        results.append(consistency_result)

    if args.mode in ["all", "rust", "python"]:
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        print(f"{'Backend':<15} {'Iterations':<15} {'Time (ms)':<15} {'Time/Update (μs)':<20}")
        print("-" * 80)

        for result in results:
            print(f"{result['backend']:<15} {result['iterations']:<15} {result['time_ms']:<15.4f} {result['time_per_update_us']:<20.2f}")

        if len(results) == 2:
            rust_result = results[0]
            python_result = results[1]
            speedup = python_result["time_ms"] / rust_result["time_ms"] if rust_result["time_ms"] > 0 else 0
            print("-" * 80)
            print(f"{'Speedup':<15} {speedup:<15.2f}x")
            print("=" * 80)

            # Analysis
            print("\nAnalysis:")
            print(f"  - Rust time:    {rust_result['time_ms']:.4f} ms")
            print(f"  - Python time:  {python_result['time_ms']:.4f} ms")
            print(f"  - Speedup:      {speedup:.2f}x")
            print(f"  - Expected:     2-3x speedup")

            if speedup >= 2.0:
                print(f"  ✓ Target met: {speedup:.2f}x speedup")
            elif speedup >= 1.5:
                print(f"  ⚠ Partial speedup: {speedup:.2f}x (target: 2x)")
            else:
                print(f"  ✗ Target not met: {speedup:.2f}x (expected ≥ 2x)")

            # Efficiency metrics
            rust_ops_per_sec = (rust_result["iterations"] * 6) / rust_result["time_ms"] * 1000  # 6 MAs per update
            python_ops_per_sec = (python_result["iterations"] * 6) / python_result["time_ms"] * 1000
            print(f"  - Rust ops/sec:    {rust_ops_per_sec:.0f}")
            print(f"  - Python ops/sec:  {python_ops_per_sec:.0f}")

    if args.mode == "consistency":
        print("\n" + "=" * 80)
        print("CONSISTENCY CHECK RESULTS")
        print("=" * 80)

        for result in results:
            print(f"Iterations: {result['iterations']}")
            print(f"Max diff:   {result['max_diff']:.10f}")
            print(f"Avg diff:   {result['avg_diff']:.10f}")
            print(f"Consistent:  {result['consistent']}")

            if result['consistent']:
                print(f"✓ Rust and Python outputs are consistent")
            else:
                print(f"✗ Rust and Python outputs differ")

    # Run variable iterations benchmark in 'all' mode
    if args.mode == "all":
        benchmark_variable_iterations()

    # Environment info
    print("\nEnvironment:")
    import platform
    print(f"  - Platform: {platform.platform()}")
    print(f"  - Processor: {platform.processor()}")
    print(f"  - CPU Count: {__import__('os').cpu_count()}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
