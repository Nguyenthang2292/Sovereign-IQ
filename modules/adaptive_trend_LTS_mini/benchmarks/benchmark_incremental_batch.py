"""Benchmark script for batch_update() vs sequential update() in IncrementalATC.

This script measures the performance difference between:
1. Sequential update() calls (updating one price at a time)
2. Batch update() calls (updating multiple prices at once)

Target: 1.5-2x speedup for batch size >= 10.
"""

import time

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.core.compute_atc_signals import IncrementalATC


def generate_price_series(base_price: float, bars: int, volatility: float = 0.01) -> pd.Series:
    """Generate a synthetic price series for testing.

    Args:
        base_price: Starting price
        bars: Number of price bars
        volatility: Price volatility

    Returns:
        Price series
    """
    np.random.seed(42)
    returns = np.random.normal(0, volatility, bars)
    prices = base_price * (1 + returns).cumprod()
    return pd.Series(prices)


def benchmark_sequential_update(config, init_prices, test_prices):
    """Benchmark sequential update() calls.

    Args:
        config: ATC configuration
        init_prices: Initial price series for initialization
        test_prices: Price series for updates

    Returns:
        Total time for all updates
    """
    atc = IncrementalATC(config)
    atc.initialize(init_prices)

    start_time = time.time()
    for price in test_prices:
        atc.update(price)
    elapsed_time = time.time() - start_time

    return elapsed_time


def benchmark_batch_update(config, init_prices, test_prices):
    """Benchmark batch_update() call.

    Args:
        config: ATC configuration
        init_prices: Initial price series for initialization
        test_prices: Price series for updates

    Returns:
        Total time for all updates
    """
    atc = IncrementalATC(config)
    atc.initialize(init_prices)

    start_time = time.time()
    atc.batch_update(test_prices.tolist())
    elapsed_time = time.time() - start_time

    return elapsed_time


def run_benchmark():
    """Run comprehensive benchmark comparing batch vs sequential updates."""
    config = {
        "ema_len": 28,
        "hma_len": 28,
        "wma_len": 28,
        "dema_len": 28,
        "lsma_len": 28,
        "kama_len": 28,
        "De": 0.03,
        "La": 0.02,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
    }

    batch_sizes = [1, 10, 50, 100]
    num_iterations = 10
    total_test_bars = 2000

    results = []

    print("=" * 80)
    print("Batch Update vs Sequential Update Benchmark")
    print("=" * 80)
    print()

    for batch_size in batch_sizes:
        # Generate test data
        np.random.seed(42)
        base_price = 100.0
        test_prices = generate_price_series(base_price, total_test_bars)
        init_prices = test_prices[: total_test_bars - batch_size]

        print(f"Benchmarking with batch_size={batch_size}, {batch_size * num_iterations} total updates...")
        print("-" * 80)

        # Benchmark sequential updates
        sequential_times = []
        for _ in range(num_iterations):
            seq_time = benchmark_sequential_update(config, init_prices, test_prices[-batch_size:])
            sequential_times.append(seq_time)

        avg_sequential = np.mean(sequential_times)
        std_sequential = np.std(sequential_times)

        print("  Sequential Updates:")
        print(f"    Avg Time:     {avg_sequential * 1000:.4f}ms ± {std_sequential * 1000:.4f}ms")
        print(f"    Per Update:   {avg_sequential * 1000 / batch_size:.4f}ms")

        # Benchmark batch updates
        batch_times = []
        for _ in range(num_iterations):
            batch_time = benchmark_batch_update(config, init_prices, test_prices[-batch_size:])
            batch_times.append(batch_time)

        avg_batch = np.mean(batch_times)
        std_batch = np.std(batch_times)

        print("  Batch Updates:")
        print(f"    Avg Time:     {avg_batch * 1000:.4f}ms ± {std_batch * 1000:.4f}ms")
        print(f"    Per Update:   {avg_batch * 1000 / batch_size:.4f}ms")

        # Calculate speedup
        speedup = avg_sequential / avg_batch if avg_batch > 0 else 0
        time_saved = ((avg_sequential - avg_batch) / avg_sequential) * 100

        print("  Results:")
        print(f"    Speedup:      {speedup:.2f}x")
        print(f"    Time Saved:   {time_saved:.1f}%")
        print()

        results.append(
            {
                "batch_size": batch_size,
                "sequential_time": avg_sequential,
                "batch_time": avg_batch,
                "speedup": speedup,
                "time_saved_percent": time_saved,
            }
        )

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print(f"{'Batch Size':>10} | {'Sequential':>12} | {'Batch':>12} | {'Speedup':>8} | {'Saved':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['batch_size']:>10} | {r['sequential_time']:>12.4f}s | {r['batch_time']:>12.4f}s | {r['speedup']:>8.2f}x | {r['time_saved_percent']:>7.1f}%"
        )

    print()
    print("Conclusion:")
    print("-" * 80)

    # Filter results for batch_size >= 10
    large_batch_results = [r for r in results if r["batch_size"] >= 10]

    if large_batch_results:
        avg_speedup_large = np.mean([r["speedup"] for r in large_batch_results])
        print(f"Average speedup for batch_size >= 10: {avg_speedup_large:.2f}x")

        if avg_speedup_large >= 2.0:
            print("✅ Batch mode shows 2-5x speedup as expected (target met)")
        elif avg_speedup_large >= 1.5:
            print("⚠️  Batch mode shows moderate speedup (below target but beneficial)")
        else:
            print("❌ Batch mode does not show significant speedup (needs optimization)")
    else:
        print("⚠️  No large batch results (batch_size < 10)")

    print()
    print("Performance observations:")
    print("-" * 80)
    speedup_by_size = {r["batch_size"]: r["speedup"] for r in results}
    for size in sorted(speedup_by_size.keys()):
        print(f"  Batch size {size}: {speedup_by_size[size]:.2f}x speedup")

    print()
    print("Note: Small batch sizes (1) show no speedup because the overhead")
    print("      of the batch call is larger than the update itself.")


if __name__ == "__main__":
    run_benchmark()
