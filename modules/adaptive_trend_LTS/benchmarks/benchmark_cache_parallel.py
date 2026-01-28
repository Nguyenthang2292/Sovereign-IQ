"""
Benchmark for Phase 8.1: Intelligent Cache Warming & Parallelism.
Compares different execution modes to demonstrate performance gains.
"""

import asyncio
import time
from typing import Dict

import numpy as np
import pandas as pd
from tabulate import tabulate

from modules.adaptive_trend_LTS.core.async_io.async_compute import run_batch_atc_async
from modules.adaptive_trend_LTS.core.compute_atc_signals.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_LTS.utils.cache_manager import get_cache_manager, reset_cache_manager
from modules.common.ui.logging import log_info, log_success


def generate_test_data(num_symbols: int, bars: int) -> Dict[str, pd.Series]:
    """Generate mock price data for multiple symbols."""
    data = {}
    for i in range(num_symbols):
        symbol = f"SYM_{i}"
        np.random.seed(i)
        prices = 100 + np.cumsum(np.random.randn(bars) * 0.5)
        data[symbol] = pd.Series(prices)
    return data


async def run_benchmark():
    num_symbols = 20
    bars = 2000
    config = {
        "ema_len": 28,
        "hull_len": 28,
        "wma_len": 28,
        "dema_len": 28,
        "lsma_len": 28,
        "kama_len": 28,
        "fast_mode": True,
    }

    log_info(f"Starting Phase 8.1 Benchmark: {num_symbols} symbols, {bars} bars")
    symbols_data = generate_test_data(num_symbols, bars)
    results = []

    # 1. Baseline: No Warming, Synchronous
    log_info("Running Mode 1: Baseline (Sync, No Warming)...")
    reset_cache_manager()
    cache_mgr = get_cache_manager()
    cache_mgr.clear()  # Ensure clean state

    start = time.time()
    for symbol, prices in symbols_data.items():
        compute_atc_signals(prices, **config, use_cache=True)
    baseline_time = time.time() - start
    baseline_stats = cache_mgr.get_stats()
    results.append(["Baseline", f"{baseline_time:.4f}s", "1.0x", f"{baseline_stats['hit_rate_percent']:.1f}%"])

    # 2. Warmed Only: With Warming, Synchronous
    log_info("Running Mode 2: Warmed Only (Sync, Warmed)...")
    # Warm the cache
    cache_mgr.warm_cache(symbols_data, [config])

    # Reset hit/miss counters for the test run
    cache_mgr._misses = 0
    cache_mgr._hits_l1 = 0
    cache_mgr._hits_l2 = 0

    start = time.time()
    for symbol, prices in symbols_data.items():
        compute_atc_signals(prices, **config, use_cache=True)
    warmed_time = time.time() - start
    warmed_stats = cache_mgr.get_stats()
    results.append(
        [
            "Warmed Only",
            f"{warmed_time:.4f}s",
            f"{baseline_time / warmed_time:.2f}x",
            f"{warmed_stats['hit_rate_percent']:.1f}%",
        ]
    )

    # 3. Parallel Only: No Warming, Asynchronous
    log_info("Running Mode 3: Parallel Only (Async, No Warming)...")
    reset_cache_manager()
    cache_mgr = get_cache_manager()
    cache_mgr.clear()

    start = time.time()
    await run_batch_atc_async(symbols_data, **config, use_cache=True)
    parallel_time = time.time() - start
    parallel_stats = cache_mgr.get_stats()
    results.append(
        [
            "Parallel Only",
            f"{parallel_time:.4f}s",
            f"{baseline_time / parallel_time:.2f}x",
            f"{parallel_stats['hit_rate_percent']:.1f}%",
        ]
    )

    # 4. Warmed + Parallel: After Warming, Asynchronous
    log_info("Running Mode 4: Warmed + Parallel (Async, Warmed)...")
    cache_mgr.warm_cache(symbols_data, [config])

    cache_mgr._misses = 0
    cache_mgr._hits_l1 = 0
    cache_mgr._hits_l2 = 0

    start = time.time()
    await run_batch_atc_async(symbols_data, **config, use_cache=True)
    total_time = time.time() - start
    total_stats = cache_mgr.get_stats()
    results.append(
        [
            "Warmed + Parallel",
            f"{total_time:.4f}s",
            f"{baseline_time / total_time:.2f}x",
            f"{total_stats['hit_rate_percent']:.1f}%",
        ]
    )

    # Print Report
    headers = ["Execution Mode", "Time", "Speedup", "Hit Rate"]
    print("\n" + "=" * 70)
    print("PHASE 8.1 BENCHMARK REPORT: CACHE & PARALLELISM")
    print("=" * 70)
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print("=" * 70)

    log_success("Benchmark completed successfully.")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
