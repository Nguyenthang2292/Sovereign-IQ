"""Benchmark script comparing parallel vs sequential data fetching.

This script compares the performance of parallel batch data fetching vs
sequential fetching for 100 symbols to validate the performance improvement.
"""

import sys
from pathlib import Path

# Add project root to sys.path for module imports
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
from typing import List

from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.common.ui.logging import log_error, log_info, log_success, log_warn


def benchmark_sequential_fetch(
    data_fetcher: DataFetcher,
    symbols: List[str],
    limit: int = 500,
    timeframe: str = "1h",
) -> float:
    """
    Benchmark sequential fetching of OHLCV data.

    Args:
        data_fetcher: DataFetcher instance
        symbols: List of symbols to fetch
        limit: Number of candles to fetch
        timeframe: Timeframe string

    Returns:
        Time taken in seconds
    """
    log_info(f"\n{'='*60}")
    log_info("SEQUENTIAL FETCH BENCHMARK")
    log_info(f"{'='*60}")
    log_info(f"Fetching {len(symbols)} symbols sequentially...")

    start_time = time.time()
    success_count = 0

    for i, symbol in enumerate(symbols, 1):
        try:
            df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol=symbol,
                limit=limit,
                timeframe=timeframe,
                check_freshness=False,
            )
            if df is not None:
                success_count += 1

            # Progress update every 10 symbols
            if i % 10 == 0 or i == len(symbols):
                elapsed = time.time() - start_time
                log_info(f"Progress: {i}/{len(symbols)} symbols ({success_count} successful) - {elapsed:.2f}s elapsed")
        except Exception as e:
            log_warn(f"Error fetching {symbol}: {type(e).__name__}: {e}")

    elapsed_time = time.time() - start_time

    log_success(f"\nSequential fetch complete:")
    log_info(f"  - Total time: {elapsed_time:.2f}s")
    log_info(f"  - Successful: {success_count}/{len(symbols)}")
    log_info(f"  - Average time per symbol: {elapsed_time/len(symbols):.3f}s")

    return elapsed_time


def benchmark_parallel_fetch(
    data_fetcher: DataFetcher,
    symbols: List[str],
    limit: int = 500,
    timeframe: str = "1h",
    max_workers: int = None,
) -> float:
    """
    Benchmark parallel batch fetching of OHLCV data.

    Args:
        data_fetcher: DataFetcher instance
        symbols: List of symbols to fetch
        limit: Number of candles to fetch
        timeframe: Timeframe string
        max_workers: Maximum number of worker threads

    Returns:
        Time taken in seconds
    """
    log_info(f"\n{'='*60}")
    log_info("PARALLEL FETCH BENCHMARK")
    log_info(f"{'='*60}")
    log_info(f"Fetching {len(symbols)} symbols in parallel (max_workers={max_workers or 'auto'})...")

    start_time = time.time()

    results = data_fetcher.fetch_ohlcv_batch_parallel(
        symbols=symbols,
        limit=limit,
        timeframe=timeframe,
        check_freshness=False,
        max_workers=max_workers,
    )

    elapsed_time = time.time() - start_time

    success_count = sum(1 for df, _ in results.values() if df is not None)

    log_success(f"\nParallel fetch complete:")
    log_info(f"  - Total time: {elapsed_time:.2f}s")
    log_info(f"  - Successful: {success_count}/{len(symbols)}")
    log_info(f"  - Average time per symbol: {elapsed_time/len(symbols):.3f}s")

    return elapsed_time


def benchmark_prefetch(
    data_fetcher: DataFetcher,
    symbols: List[str],
    limit: int = 500,
    timeframe: str = "1h",
    max_workers: int = None,
) -> float:
    """
    Benchmark prefetch functionality.

    Args:
        data_fetcher: DataFetcher instance
        symbols: List of symbols to prefetch
        limit: Number of candles to fetch
        timeframe: Timeframe string
        max_workers: Maximum number of worker threads

    Returns:
        Time taken in seconds
    """
    log_info(f"\n{'='*60}")
    log_info("PREFETCH BENCHMARK")
    log_info(f"{'='*60}")
    log_info(f"Prefetching {len(symbols)} symbols (max_workers={max_workers or 'auto'})...")

    start_time = time.time()

    success_count = data_fetcher.prefetch_symbols_data(
        symbols=symbols,
        limit=limit,
        timeframe=timeframe,
        check_freshness=False,
        max_workers=max_workers,
    )

    elapsed_time = time.time() - start_time

    log_success(f"\nPrefetch complete:")
    log_info(f"  - Total time: {elapsed_time:.2f}s")
    log_info(f"  - Successful: {success_count}/{len(symbols)}")
    log_info(f"  - Average time per symbol: {elapsed_time/len(symbols):.3f}s")

    # Test that cached data is used
    log_info("\nTesting cache hit (fetching first symbol again)...")
    test_start = time.time()
    df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
        symbol=symbols[0],
        limit=limit,
        timeframe=timeframe,
        check_freshness=False,
    )
    cache_hit_time = time.time() - test_start
    log_info(f"  - Cache hit time: {cache_hit_time:.4f}s (should be < 0.001s)")

    return elapsed_time


def run_benchmark(num_symbols: int = 100, limit: int = 500, timeframe: str = "1h"):
    """
    Run complete benchmark comparing sequential vs parallel fetching.

    Args:
        num_symbols: Number of symbols to fetch (default: 100)
        limit: Number of candles to fetch per symbol (default: 500)
        timeframe: Timeframe string (default: '1h')
    """
    log_info("="*60)
    log_info("DATA FETCHING BENCHMARK")
    log_info("="*60)
    log_info(f"Configuration:")
    log_info(f"  - Number of symbols: {num_symbols}")
    log_info(f"  - Candles per symbol: {limit}")
    log_info(f"  - Timeframe: {timeframe}")

    # Initialize DataFetcher
    log_info("\nInitializing DataFetcher...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)

    # Get symbols to test
    log_info(f"\nFetching {num_symbols} futures symbols from Binance...")
    try:
        all_symbols = data_fetcher.list_binance_futures_symbols(
            max_candidates=num_symbols,
            progress_label="Symbol Discovery",
        )
        if not all_symbols:
            log_error("No symbols found. Cannot run benchmark.")
            return

        symbols = all_symbols[:num_symbols]
        log_success(f"Found {len(symbols)} symbols to test")
    except Exception as e:
        log_error(f"Error fetching symbols: {type(e).__name__}: {e}")
        return

    # Clear cache to ensure fair comparison
    data_fetcher._ohlcv_dataframe_cache.clear()

    # Benchmark 1: Sequential fetch
    try:
        sequential_time = benchmark_sequential_fetch(data_fetcher, symbols, limit, timeframe)
    except Exception as e:
        log_error(f"Sequential benchmark failed: {type(e).__name__}: {e}")
        sequential_time = None

    # Clear cache between benchmarks
    data_fetcher._ohlcv_dataframe_cache.clear()

    # Benchmark 2: Parallel fetch with default workers
    try:
        parallel_time = benchmark_parallel_fetch(data_fetcher, symbols, limit, timeframe, max_workers=None)
    except Exception as e:
        log_error(f"Parallel benchmark failed: {type(e).__name__}: {e}")
        parallel_time = None

    # Clear cache between benchmarks
    data_fetcher._ohlcv_dataframe_cache.clear()

    # Benchmark 3: Parallel fetch with 10 workers
    try:
        parallel_time_10 = benchmark_parallel_fetch(data_fetcher, symbols, limit, timeframe, max_workers=10)
    except Exception as e:
        log_error(f"Parallel benchmark (10 workers) failed: {type(e).__name__}: {e}")
        parallel_time_10 = None

    # Clear cache between benchmarks
    data_fetcher._ohlcv_dataframe_cache.clear()

    # Benchmark 4: Parallel fetch with 20 workers
    try:
        parallel_time_20 = benchmark_parallel_fetch(data_fetcher, symbols, limit, timeframe, max_workers=20)
    except Exception as e:
        log_error(f"Parallel benchmark (20 workers) failed: {type(e).__name__}: {e}")
        parallel_time_20 = None

    # Clear cache between benchmarks
    data_fetcher._ohlcv_dataframe_cache.clear()

    # Benchmark 5: Prefetch
    try:
        prefetch_time = benchmark_prefetch(data_fetcher, symbols, limit, timeframe, max_workers=10)
    except Exception as e:
        log_error(f"Prefetch benchmark failed: {type(e).__name__}: {e}")
        prefetch_time = None

    # Print summary
    log_info(f"\n{'='*60}")
    log_info("BENCHMARK SUMMARY")
    log_info(f"{'='*60}")

    if sequential_time:
        log_info(f"Sequential fetch:              {sequential_time:.2f}s")

    if parallel_time:
        log_info(f"Parallel fetch (auto workers): {parallel_time:.2f}s")
        if sequential_time:
            speedup = sequential_time / parallel_time
            improvement = ((sequential_time - parallel_time) / sequential_time) * 100
            log_success(f"  -> Speedup: {speedup:.2f}x ({improvement:.1f}% faster)")

    if parallel_time_10:
        log_info(f"Parallel fetch (10 workers):   {parallel_time_10:.2f}s")
        if sequential_time:
            speedup = sequential_time / parallel_time_10
            improvement = ((sequential_time - parallel_time_10) / sequential_time) * 100
            log_success(f"  -> Speedup: {speedup:.2f}x ({improvement:.1f}% faster)")

    if parallel_time_20:
        log_info(f"Parallel fetch (20 workers):   {parallel_time_20:.2f}s")
        if sequential_time:
            speedup = sequential_time / parallel_time_20
            improvement = ((sequential_time - parallel_time_20) / sequential_time) * 100
            log_success(f"  -> Speedup: {speedup:.2f}x ({improvement:.1f}% faster)")

    if prefetch_time:
        log_info(f"Prefetch (10 workers):         {prefetch_time:.2f}s")
        if sequential_time:
            speedup = sequential_time / prefetch_time
            improvement = ((sequential_time - prefetch_time) / sequential_time) * 100
            log_success(f"  -> Speedup: {speedup:.2f}x ({improvement:.1f}% faster)")

    log_info(f"{'='*60}")

    # Determine if benchmark passed
    if sequential_time and parallel_time:
        if parallel_time < sequential_time:
            log_success("\nBENCHMARK PASSED: Parallel fetch is faster than sequential!")
            log_info(f"Improvement: {((sequential_time - parallel_time) / sequential_time) * 100:.1f}%")
        else:
            log_warn("\nBENCHMARK FAILED: Parallel fetch is not faster than sequential.")
            log_warn("This may be due to network conditions or API rate limits.")
    else:
        log_warn("\nBENCHMARK INCOMPLETE: Could not complete all benchmarks.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark parallel vs sequential data fetching")
    parser.add_argument(
        "--symbols",
        type=int,
        default=100,
        help="Number of symbols to fetch (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of candles to fetch per symbol (default: 500)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Timeframe string (default: 1h)",
    )

    args = parser.parse_args()

    run_benchmark(num_symbols=args.symbols, limit=args.limit, timeframe=args.timeframe)
