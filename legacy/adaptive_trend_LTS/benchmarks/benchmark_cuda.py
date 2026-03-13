"""
Performance benchmarks for CUDA kernels vs CPU implementations.

Run with: python benchmarks/benchmark_cuda.py
"""

import time

import numpy as np
import pandas as pd

try:
    from atc_rust import (
        calculate_ema_cuda,
        calculate_ema_rust,
        calculate_equity_cuda,
        calculate_equity_rust,
        calculate_hma_cuda,
        calculate_hma_rust,
        calculate_kama_cuda,
        calculate_kama_rust,
        calculate_wma_cuda,
        calculate_wma_rust,
    )

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠️  CUDA extensions not available. Skipping benchmarks.")


def benchmark_function(func, *args, n_runs=10, warmup=2):
    """Benchmark a function with warmup runs."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def benchmark_equity():
    """Benchmark equity calculation."""
    print("\n" + "=" * 60)
    print("EQUITY CALCULATION BENCHMARK")
    print("=" * 60)

    sizes = [1000, 5000, 10000]
    results = []

    for n in sizes:
        r_values = np.random.randn(n) * 0.02
        sig_prev = np.random.choice([-1.0, 0.0, 1.0], size=n)

        cpu_stats = benchmark_function(calculate_equity_rust, r_values, sig_prev, 100.0, 0.97, 0)

        cuda_stats = benchmark_function(calculate_equity_cuda, r_values, sig_prev, 100.0, 0.97, 0)

        speedup = cpu_stats["mean_ms"] / cuda_stats["mean_ms"]

        results.append(
            {"size": n, "cpu_ms": cpu_stats["mean_ms"], "cuda_ms": cuda_stats["mean_ms"], "speedup": speedup}
        )

        print(f"\nSize: {n:,} bars")
        print(f"  CPU:  {cpu_stats['mean_ms']:.3f} ± {cpu_stats['std_ms']:.3f} ms")
        print(f"  CUDA: {cuda_stats['mean_ms']:.3f} ± {cuda_stats['std_ms']:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")

    return results


def benchmark_ma():
    """Benchmark moving average calculations."""
    print("\n" + "=" * 60)
    print("MOVING AVERAGE BENCHMARK")
    print("=" * 60)

    sizes = [1000, 5000, 10000]
    ma_types = [
        ("EMA", calculate_ema_rust, calculate_ema_cuda),
        ("KAMA", calculate_kama_rust, calculate_kama_cuda),
        ("WMA", calculate_wma_rust, calculate_wma_cuda),
        ("HMA", calculate_hma_rust, calculate_hma_cuda),
    ]

    results = []

    for ma_name, cpu_func, cuda_func in ma_types:
        print(f"\n{ma_name}:")
        for n in sizes:
            prices = np.random.randn(n).cumsum() + 100
            length = 20

            cpu_stats = benchmark_function(cpu_func, prices, length)
            cuda_stats = benchmark_function(cuda_func, prices, length)

            speedup = cpu_stats["mean_ms"] / cuda_stats["mean_ms"]

            results.append(
                {
                    "ma_type": ma_name,
                    "size": n,
                    "cpu_ms": cpu_stats["mean_ms"],
                    "cuda_ms": cuda_stats["mean_ms"],
                    "speedup": speedup,
                }
            )

            print(
                f"  {n:,} bars: CPU={cpu_stats['mean_ms']:.3f}ms, "
                f"CUDA={cuda_stats['mean_ms']:.3f}ms, Speedup={speedup:.2f}x"
            )

    return results


def benchmark_signal():
    """Benchmark signal classification."""
    print("\n" + "=" * 60)
    print("SIGNAL CLASSIFICATION BENCHMARK")
    print("=" * 60)

    from atc_rust import calculate_and_classify_cuda, calculate_average_signal_cuda

    sizes = [1000, 5000, 10000]
    n_mas = 5

    results = []

    for n_bars in sizes:
        signals = np.random.randn(n_mas, n_bars)
        equities = np.abs(np.random.randn(n_mas, n_bars)) + 1.0

        # Benchmark weighted average
        avg_stats = benchmark_function(calculate_average_signal_cuda, signals, equities, 0.5, -0.5)

        # Benchmark fused kernel
        fused_stats = benchmark_function(calculate_and_classify_cuda, signals, equities, 0.5, -0.5)

        results.append(
            {
                "size": n_bars,
                "avg_ms": avg_stats["mean_ms"],
                "fused_ms": fused_stats["mean_ms"],
            }
        )

        print(f"\n{n_bars:,} bars:")
        print(f"  Weighted Avg: {avg_stats['mean_ms']:.3f} ms")
        print(f"  Fused (Avg+Classify): {fused_stats['mean_ms']:.3f} ms")

    return results


def main():
    """Run all benchmarks."""
    if not CUDA_AVAILABLE:
        return

    print("\n" + "=" * 60)
    print("CUDA KERNEL PERFORMANCE BENCHMARKS")
    print("=" * 60)
    print(f"NumPy version: {np.__version__}")
    print(f"Running on: {np.show_config()}")

    # Run benchmarks
    equity_results = benchmark_equity()
    ma_results = benchmark_ma()
    signal_results = benchmark_signal()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nEquity Calculation:")
    df_equity = pd.DataFrame(equity_results)
    print(df_equity.to_string(index=False))
    print(f"Average Speedup: {df_equity['speedup'].mean():.2f}x")

    print("\nMoving Averages:")
    df_ma = pd.DataFrame(ma_results)
    for ma_type in df_ma["ma_type"].unique():
        subset = df_ma[df_ma["ma_type"] == ma_type]
        avg_speedup = subset["speedup"].mean()
        print(f"  {ma_type}: {avg_speedup:.2f}x average speedup")

    print("\nSignal Classification:")
    df_signal = pd.DataFrame(signal_results)
    print(df_signal.to_string(index=False))


if __name__ == "__main__":
    main()
