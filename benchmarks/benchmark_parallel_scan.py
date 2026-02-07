"""
Benchmark: Sequential vs Parallel Scanning for ATC

This script benchmarks the performance difference between sequential and
parallel (threadpool) execution modes of the ATC scanner.

Usage:
    python benchmarks/benchmark_parallel_scan.py [--symbols 100] [--workers 10]

Examples:
    # Default: 100 symbols with 10 workers
    python benchmarks/benchmark_parallel_scan.py

    # Custom symbol count
    python benchmarks/benchmark_parallel_scan.py --symbols 50

    # Custom worker count
    python benchmarks/benchmark_parallel_scan.py --symbols 100 --workers 20
"""

import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from typing import List

from modules.adaptive_trend_LTS_mini.core.scanner.scan_all_symbols import scan_all_symbols
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager

# Configure logging to avoid noise during benchmark
logging.getLogger().setLevel(logging.INFO)

def run_benchmark(num_symbols: int, max_workers: int):
    """Run the benchmark."""
    print(f"Preparing benchmark with {num_symbols} symbols...")

    # Initialize components
    exchange_manager = ExchangeManager(testnet=False) # Use real exchange or mock?
    data_fetcher = DataFetcher(exchange_manager)
    config = ATCConfig(
        timeframe="1h",
        limit=100, # Keep it light
        ema_len=20,
    )

    # Get symbols list first to ensure fairness
    print("Fetching symbol list...")
    all_symbols = data_fetcher.list_binance_futures_symbols()
    if not all_symbols:
        print("Error: Could not fetch symbols from Binance.")
        return

    test_symbols = all_symbols[:num_symbols]
    print(f"Selected {len(test_symbols)} symbols for testing.")

    # 1. Sequential Benchmark
    print("\n[1/2] Running Sequential Scan...")
    start_seq = time.perf_counter()
    scan_all_symbols(
        data_fetcher=data_fetcher,
        atc_config=config,
        symbols=test_symbols,
        execution_mode="sequential",
        min_signal=0.0 # Capture all
    )
    end_seq = time.perf_counter()
    time_seq = end_seq - start_seq
    print(f"Sequential time: {time_seq:.2f}s")

    # 2. Parallel Benchmark
    print(f"\n[2/2] Running Parallel Scan (Threadpool, workers={max_workers})...")
    start_par = time.perf_counter()
    scan_all_symbols(
        data_fetcher=data_fetcher,
        atc_config=config,
        symbols=test_symbols,
        execution_mode="threadpool",
        max_workers=max_workers,
        min_signal=0.0
    )
    end_par = time.perf_counter()
    time_par = end_par - start_par
    print(f"Parallel time:   {time_par:.2f}s")

    # Results
    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print("="*40)
    print(f"Symbols:    {num_symbols}")
    print(f"Sequential: {time_seq:.2f}s ({(time_seq/num_symbols):.3f}s/symbol)")
    print(f"Parallel:   {time_par:.2f}s ({(time_par/num_symbols):.3f}s/symbol)")

    if time_par < time_seq:
        speedup = time_seq / time_par
        print(f"Speedup:    {speedup:.2f}x faster")
    else:
        print("Parallel was slower (likely overhead or rate limits dominant for small batch)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ATC Scanner modes")
    parser.add_argument("--symbols", type=int, default=100, help="Number of symbols to scan (default: 100)")
    parser.add_argument("--workers", type=int, default=10, help="Number of worker threads (default: 10)")
    args = parser.parse_args()

    try:
        run_benchmark(args.symbols, args.workers)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
