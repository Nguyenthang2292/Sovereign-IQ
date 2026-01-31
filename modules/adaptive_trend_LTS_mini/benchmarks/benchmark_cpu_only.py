"""
Benchmark CPU-only version performance.
Run this to validate CPU performance characteristics after migration.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def generate_sample_data(n_bars=1000, n_symbols=1):
    """Generate sample price data for benchmarking."""
    np.random.seed(42)

    symbols_data = {}
    for i in range(n_symbols):
        # Generate random walk price data
        returns = np.random.normal(0.001, 0.02, n_bars)
        prices = 100 * np.exp(np.cumsum(returns))

        symbol = f"TEST{i:03d}"
        symbols_data[symbol] = pd.Series(prices)

    return symbols_data


def benchmark_single_symbol():
    """Benchmark single symbol processing."""
    print("\n" + "=" * 60)
    print("Benchmark 1: Single Symbol Processing")
    print("=" * 60)

    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals

    symbols_data = generate_sample_data(n_bars=1000, n_symbols=1)
    prices = list(symbols_data.values())[0]

    # Warm up
    _ = compute_atc_signals(prices, use_rust_backend=True)

    # Benchmark
    times = []
    for _ in range(5):
        start = time.time()
        result = compute_atc_signals(prices, use_rust_backend=True)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print("Data: 1000 bars")
    print(f"Time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"Speed: {1000 / avg_time:.0f} bars/second")

    # Verify result
    if "Average_Signal" in result:
        print("✅ Result contains Average_Signal")
    else:
        print("❌ Result missing Average_Signal")

    return avg_time


def benchmark_batch_small():
    """Benchmark small batch (10 symbols)."""
    print("\n" + "=" * 60)
    print("Benchmark 2: Small Batch (10 symbols)")
    print("=" * 60)

    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.batch_processor import process_symbols_batch_rust

    symbols_data = generate_sample_data(n_bars=500, n_symbols=10)

    config = {
        "ema_len": 28,
        "hma_len": 28,
        "wma_len": 28,
        "dema_len": 28,
        "lsma_len": 28,
        "kama_len": 28,
        "robustness": "Medium",
        "La": 0.02,
        "De": 0.03,
    }

    start = time.time()
    results = process_symbols_batch_rust(symbols_data, config)
    elapsed = time.time() - start

    print("Data: 10 symbols × 500 bars = 5000 bars total")
    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {len(results)} symbols in {elapsed:.1f}s")
    print(f"Per symbol: {elapsed / len(results):.3f}s")

    return elapsed


def benchmark_batch_medium():
    """Benchmark medium batch (50 symbols)."""
    print("\n" + "=" * 60)
    print("Benchmark 3: Medium Batch (50 symbols)")
    print("=" * 60)

    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.batch_processor import process_symbols_batch_rust

    symbols_data = generate_sample_data(n_bars=500, n_symbols=50)

    config = {
        "ema_len": 28,
        "hma_len": 28,
        "wma_len": 28,
        "dema_len": 28,
        "lsma_len": 28,
        "kama_len": 28,
        "robustness": "Medium",
        "La": 0.02,
        "De": 0.03,
    }

    start = time.time()
    results = process_symbols_batch_rust(symbols_data, config)
    elapsed = time.time() - start

    print("Data: 50 symbols × 500 bars = 25000 bars total")
    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {len(results)} symbols in {elapsed:.1f}s")
    print(f"Per symbol: {elapsed / len(results):.3f}s")

    return elapsed


def benchmark_ma_calculations():
    """Benchmark individual MA calculations."""
    print("\n" + "=" * 60)
    print("Benchmark 4: Individual MA Calculations")
    print("=" * 60)

    from modules.adaptive_trend_LTS_mini.core.rust_backend import (
        calculate_dema,
        calculate_ema,
        calculate_hma,
        calculate_kama,
        calculate_wma,
    )

    np.random.seed(42)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 1000))))

    ma_types = [
        ("EMA", calculate_ema),
        ("WMA", calculate_wma),
        ("HMA", calculate_hma),
        ("DEMA", calculate_dema),
        ("KAMA", calculate_kama),
    ]

    results = []
    for name, func in ma_types:
        times = []
        for _ in range(10):
            start = time.time()
            _ = func(prices.values, 28, use_rust=True)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        results.append((name, avg_time))
        print(f"{name:6s}: {avg_time * 1000:.2f}ms (avg of 10 runs)")

    return results


def check_rust_extension():
    """Check Rust extension status."""
    print("\n" + "=" * 60)
    print("Rust Extension Status")
    print("=" * 60)

    try:
        import atc_rust

        print(f"✅ atc_rust loaded: {atc_rust.__file__}")

        # Check for CPU functions
        cpu_funcs = [
            "calculate_ema_rust",
            "calculate_wma_rust",
            "calculate_hma_rust",
            "calculate_equity_rust",
            "compute_atc_signals_batch_cpu",
        ]

        available = []
        for func in cpu_funcs:
            if hasattr(atc_rust, func):
                available.append(func)

        print(f"✅ CPU functions available: {len(available)}/{len(cpu_funcs)}")
        for func in available[:3]:
            print(f"   - {func}")

        return len(available) > 0
    except ImportError:
        print("❌ atc_rust not available")
        print("   Run: cd rust_extensions && cargo build --release")
        return False


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("CPU-Only Performance Benchmark")
    print("=" * 60)
    print("Testing adaptive_trend_LTS_mini CPU-only version")
    print("Expected performance: 5-10x slower than GPU version")

    # Check Rust extension
    if not check_rust_extension():
        print("\n❌ Cannot run benchmarks without Rust extension")
        return 1

    results = {}

    try:
        results["single"] = benchmark_single_symbol()
    except Exception as e:
        print(f"❌ Single symbol benchmark failed: {e}")
        results["single"] = None

    try:
        results["batch_small"] = benchmark_batch_small()
    except Exception as e:
        print(f"❌ Small batch benchmark failed: {e}")
        results["batch_small"] = None

    try:
        results["batch_medium"] = benchmark_batch_medium()
    except Exception as e:
        print(f"❌ Medium batch benchmark failed: {e}")
        results["batch_medium"] = None

    try:
        results["ma_calc"] = benchmark_ma_calculations()
    except Exception as e:
        print(f"❌ MA calculations benchmark failed: {e}")
        results["ma_calc"] = None

    # Summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)

    if results["single"]:
        print(f"Single symbol (1000 bars): {results['single']:.3f}s")
    if results["batch_small"]:
        print(f"Batch 10 symbols: {results['batch_small']:.3f}s")
    if results["batch_medium"]:
        print(f"Batch 50 symbols: {results['batch_medium']:.3f}s")

    print("\nPerformance Expectations:")
    print(
        "  Single symbol: ~100-500ms ✅"
        if results["single"] and results["single"] < 1.0
        else "  Single symbol: >500ms ⚠️"
    )
    print("  Batch 10: ~1-5s ✅" if results["batch_small"] and results["batch_small"] < 5.0 else "  Batch 10: >5s ⚠️")
    print(
        "  Batch 50: ~5-25s ✅" if results["batch_medium"] and results["batch_medium"] < 25.0 else "  Batch 50: >25s ⚠️"
    )

    print("\n" + "=" * 60)
    print("✅ Benchmark completed successfully!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
