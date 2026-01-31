"""Benchmark runner for O(1) incremental MAs vs legacy implementations.

This module benchmarks the performance improvement of O(1) MA implementations
compared to legacy O(n) implementations for WMA, HMA, LSMA, and KAMA.
"""

import argparse
import time
from collections import deque
from typing import Deque

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental_mas_o1 import (
    TrueO1HMA,
    TrueO1KAMA,
    TrueO1LSMA,
    TrueO1WMA,
)


def legacy_wma_update(price: float, length: int, price_window: Deque[float]) -> float:
    """Legacy WMA update (O(n) per update)."""
    price_window.append(price)
    if len(price_window) < length:
        return price

    window = list(price_window)[-length:]
    weights = np.arange(1, length + 1)
    return np.dot(window, weights) / weights.sum()


def legacy_hma_update(price: float, length: int, price_window: Deque[float], hma_window: Deque[float]) -> float:
    """Legacy HMA update (O(n) per update)."""
    half_len = max(1, length // 2)
    sqrt_len = max(1, int(np.sqrt(length)))

    price_window.append(price)
    if len(price_window) < length:
        return price

    # WMA(n/2)
    half_window = list(price_window)[-half_len:]
    half_weights = np.arange(1, half_len + 1)
    wma_half = np.dot(half_window, half_weights) / half_weights.sum()

    # WMA(n)
    full_window = list(price_window)[-length:]
    full_weights = np.arange(1, length + 1)
    wma_full = np.dot(full_window, full_weights) / full_weights.sum()

    # Intermediate
    intermediate = 2.0 * wma_half - wma_full
    hma_window.append(intermediate)

    if len(hma_window) < sqrt_len:
        return intermediate

    # WMA(sqrt(n))
    final_window = list(hma_window)[-sqrt_len:]
    final_weights = np.arange(1, sqrt_len + 1)
    return np.dot(final_window, final_weights) / final_weights.sum()


def legacy_lsma_update(price: float, length: int, price_window: Deque[float]) -> float:
    """Legacy LSMA update (O(n) per update)."""
    price_window.append(price)
    if len(price_window) < length:
        return price

    window = list(price_window)[-length:]
    x = np.arange(length)
    y = np.array(window)

    n = length
    sum_x = n * (n - 1) / 2
    sum_x2 = n * (n - 1) * (2 * n - 1) / 6
    sum_y = np.sum(y)
    sum_xy = np.dot(x, y)

    denom = n * sum_x2 - sum_x**2
    if denom == 0:
        return price

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return intercept + slope * (n - 1)


def legacy_kama_update(price: float, length: int, price_window: Deque[float], prev_kama: float) -> tuple[float, float]:
    """Legacy KAMA update (O(n) per update)."""
    price_window.append(price)
    if len(price_window) < length + 1:
        return price, price

    window = list(price_window)[- (length + 1):]
    change = abs(window[-1] - window[0])
    volatility = sum(abs(window[i] - window[i - 1]) for i in range(1, len(window)))

    er = change / volatility if volatility != 0 else 0
    fast_sc = 2.0 / (2.0 + 1)
    slow_sc = 2.0 / (30.0 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    new_kama = prev_kama + sc * (price - prev_kama)
    return new_kama, new_kama


def benchmark_wma(iterations: int = 1000, length: int = 28) -> dict:
    """Benchmark WMA: O(1) vs O(n)."""
    print(f"\nBenchmarking WMA (length={length}, iterations={iterations})...")

    # Generate test data
    np.random.seed(42)
    prices = np.random.randn(iterations + length) * 10 + 100

    # Benchmark O(1) WMA
    wma_o1 = TrueO1WMA(length)
    start = time.perf_counter()
    for price in prices:
        wma_o1.update(price)
    o1_time = time.perf_counter() - start

    # Benchmark legacy WMA
    price_window_legacy = deque(maxlen=length)
    start = time.perf_counter()
    for price in prices:
        legacy_wma_update(price, length, price_window_legacy)
    legacy_time = time.perf_counter() - start

    speedup = legacy_time / o1_time if o1_time > 0 else 0

    return {
        "ma": "WMA",
        "length": length,
        "iterations": iterations,
        "o1_time_ms": o1_time * 1000,
        "legacy_time_ms": legacy_time * 1000,
        "speedup": speedup,
    }


def benchmark_hma(iterations: int = 1000, length: int = 28) -> dict:
    """Benchmark HMA: O(1) vs O(n)."""
    print(f"\nBenchmarking HMA (length={length}, iterations={iterations})...")

    # Generate test data
    np.random.seed(42)
    prices = np.random.randn(iterations + length + int(np.sqrt(length))) * 10 + 100

    # Benchmark O(1) HMA
    hma_o1 = TrueO1HMA(length)
    start = time.perf_counter()
    for price in prices:
        hma_o1.update(price)
    o1_time = time.perf_counter() - start

    # Benchmark legacy HMA
    price_window_legacy = deque(maxlen=length)
    hma_window_legacy = deque(maxlen=int(np.sqrt(length)))
    start = time.perf_counter()
    for price in prices:
        legacy_hma_update(price, length, price_window_legacy, hma_window_legacy)
    legacy_time = time.perf_counter() - start

    speedup = legacy_time / o1_time if o1_time > 0 else 0

    return {
        "ma": "HMA",
        "length": length,
        "iterations": iterations,
        "o1_time_ms": o1_time * 1000,
        "legacy_time_ms": legacy_time * 1000,
        "speedup": speedup,
    }


def benchmark_lsma(iterations: int = 1000, length: int = 28) -> dict:
    """Benchmark LSMA: O(1) vs O(n)."""
    print(f"\nBenchmarking LSMA (length={length}, iterations={iterations})...")

    # Generate test data
    np.random.seed(42)
    prices = np.random.randn(iterations + length) * 10 + 100

    # Benchmark O(1) LSMA
    lsma_o1 = TrueO1LSMA(length)
    start = time.perf_counter()
    for price in prices:
        lsma_o1.update(price)
    o1_time = time.perf_counter() - start

    # Benchmark legacy LSMA
    price_window_legacy = deque(maxlen=length)
    start = time.perf_counter()
    for price in prices:
        legacy_lsma_update(price, length, price_window_legacy)
    legacy_time = time.perf_counter() - start

    speedup = legacy_time / o1_time if o1_time > 0 else 0

    return {
        "ma": "LSMA",
        "length": length,
        "iterations": iterations,
        "o1_time_ms": o1_time * 1000,
        "legacy_time_ms": legacy_time * 1000,
        "speedup": speedup,
    }


def benchmark_kama(iterations: int = 1000, length: int = 28) -> dict:
    """Benchmark KAMA: O(1) vs O(n)."""
    print(f"\nBenchmarking KAMA (length={length}, iterations={iterations})...")

    # Generate test data
    np.random.seed(42)
    prices = np.random.randn(iterations + length + 1) * 10 + 100

    # Benchmark O(1) KAMA
    kama_o1 = TrueO1KAMA(length)
    start = time.perf_counter()
    for price in prices:
        kama_o1.update(price)
    o1_time = time.perf_counter() - start

    # Benchmark legacy KAMA
    price_window_legacy = deque(maxlen=length + 1)
    prev_kama = prices[0]
    start = time.perf_counter()
    for price in prices:
        prev_kama, _ = legacy_kama_update(price, length, price_window_legacy, prev_kama)
    legacy_time = time.perf_counter() - start

    speedup = legacy_time / o1_time if o1_time > 0 else 0

    return {
        "ma": "KAMA",
        "length": length,
        "iterations": iterations,
        "o1_time_ms": o1_time * 1000,
        "legacy_time_ms": legacy_time * 1000,
        "speedup": speedup,
    }


def benchmark_incremental_atc(iterations: int = 1000) -> dict:
    """Benchmark IncrementalATC with O(1) vs legacy MAs."""
    from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import IncrementalATC

    print(f"\nBenchmarking IncrementalATC (iterations={iterations})...")

    # Generate test data
    np.random.seed(42)
    init_prices = pd.Series(100 + np.random.randn(200) * 10)
    update_prices = pd.Series(100 + np.random.randn(iterations) * 10)

    config = {
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
    }

    # Benchmark with O(1) MAs
    config_o1 = config.copy()
    config_o1["use_o1_mas"] = True
    atc_o1 = IncrementalATC(config_o1)
    atc_o1.initialize(init_prices)

    start = time.perf_counter()
    for price in update_prices:
        atc_o1.update(price)
    o1_time = time.perf_counter() - start

    # Benchmark with legacy MAs
    config_legacy = config.copy()
    config_legacy["use_o1_mas"] = False
    atc_legacy = IncrementalATC(config_legacy)
    atc_legacy.initialize(init_prices)

    start = time.perf_counter()
    for price in update_prices:
        atc_legacy.update(price)
    legacy_time = time.perf_counter() - start

    speedup = legacy_time / o1_time if o1_time > 0 else 0

    return {
        "ma": "IncrementalATC",
        "length": 28,
        "iterations": iterations,
        "o1_time_ms": o1_time * 1000,
        "legacy_time_ms": legacy_time * 1000,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark O(1) MAs vs legacy implementations")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations to benchmark")
    parser.add_argument("--ma", type=str, default="all", choices=["all", "wma", "hma", "lsma", "kama", "atc"], help="MA to benchmark")
    parser.add_argument("--length", type=int, default=28, help="MA window length")
    args = parser.parse_args()

    print("=" * 80)
    print("O(1) MA Benchmark Suite")
    print("=" * 80)
    print(f"Iterations: {args.iterations}")
    print(f"MA Length: {args.length}")
    print(f"Python: {__import__('sys').version}")
    print(f"NumPy: {np.__version__}")
    print("=" * 80)

    results = []

    if args.ma in ["all", "wma"]:
        results.append(benchmark_wma(args.iterations, args.length))

    if args.ma in ["all", "hma"]:
        results.append(benchmark_hma(args.iterations, args.length))

    if args.ma in ["all", "lsma"]:
        results.append(benchmark_lsma(args.iterations, args.length))

    if args.ma in ["all", "kama"]:
        results.append(benchmark_kama(args.iterations, args.length))

    if args.ma in ["all", "atc"]:
        results.append(benchmark_incremental_atc(args.iterations))

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'MA':<15} {'Length':<10} {'O(1) Time (ms)':<20} {'Legacy Time (ms)':<20} {'Speedup':<10}")
    print("-" * 80)

    total_o1_time = 0
    total_legacy_time = 0

    for result in results:
        print(f"{result['ma']:<15} {result['length']:<10} {result['o1_time_ms']:<20.4f} {result['legacy_time_ms']:<20.4f} {result['speedup']:<10.2f}x")
        total_o1_time += result['o1_time_ms']
        total_legacy_time += result['legacy_time_ms']

    print("-" * 80)
    print(f"{'Total':<15} {'-':<10} {total_o1_time:<20.4f} {total_legacy_time:<20.4f} {total_legacy_time/total_o1_time:<10.2f}x")
    print("=" * 80)

    # Analysis
    print("\nAnalysis:")
    avg_speedup = np.mean([r['speedup'] for r in results])
    print(f"  - Average speedup: {avg_speedup:.2f}x")
    print("  - All O(1) implementations use constant-time updates (no window iteration)")
    print("  - Expected speedup should be 2-5x for affected MAs (WMA, HMA, LSMA, KAMA)")

    if avg_speedup >= 2.0:
        print(f"  ✓ Target met: {avg_speedup:.2f}x average speedup")
    else:
        print(f"  ✗ Target not met: {avg_speedup:.2f}x average speedup (expected ≥ 2x)")

    print("\nEnvironment:")
    import platform
    print(f"  - Platform: {platform.platform()}")
    print(f"  - Processor: {platform.processor()}")
    print(f"  - CPU Count: {__import__('os').cpu_count()}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
