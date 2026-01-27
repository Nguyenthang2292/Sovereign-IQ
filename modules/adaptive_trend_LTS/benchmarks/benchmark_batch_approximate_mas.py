"""Benchmark for BatchApproximateMAScanner.

Compares batch vs individual calculation speed and measures memory usage.
"""

import time
import tracemalloc
from typing import Dict, List

import numpy as np
import pandas as pd

from modules.adaptive_trend_LTS.core.compute_moving_averages.batch_approximate_mas import (
    BatchApproximateMAScanner,
)
from modules.adaptive_trend_LTS.core.compute_moving_averages.approximate_mas import (
    fast_ema_approx,
    fast_hma_approx,
)


def generate_sample_prices(n_points: int = 100) -> pd.Series:
    """Generate sample price data.

    Args:
        n_points: Number of price points

    Returns:
        Price series
    """
    np.random.seed(42)
    return pd.Series(
        np.random.randn(n_points).cumsum() + 100,
        index=pd.date_range("2024-01-01", periods=n_points),
    )


def benchmark_batch_vs_individual(num_symbols: int = 100, ma_type: str = "EMA", length: int = 20) -> Dict[str, float]:
    """Benchmark batch processing vs individual calculations.

    Args:
        num_symbols: Number of symbols to process
        ma_type: MA type to calculate
        length: MA length

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmark: Batch vs Individual Calculation")
    print(f"Symbols: {num_symbols}, MA Type: {ma_type}, Length: {length}")
    print(f"{'=' * 60}")

    # Generate sample data
    symbols: Dict[str, pd.Series] = {}
    for i in range(num_symbols):
        symbols[f"SYMBOL{i}"] = generate_sample_prices(100)

    # Individual calculation
    tracemalloc.start()
    start_time = time.time()

    individual_results = {}
    for symbol, prices in symbols.items():
        if ma_type == "EMA":
            result = fast_ema_approx(prices, length)
        elif ma_type == "HMA":
            result = fast_hma_approx(prices, length)
        else:
            continue
        individual_results[symbol] = result

    individual_time = time.time() - start_time
    _, individual_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Individual calculation:")
    print(f"  Time: {individual_time:.4f}s")
    print(f"  Memory: {individual_memory / 1024 / 1024:.2f} MB")

    # Batch calculation (parallel)
    scanner = BatchApproximateMAScanner(num_threads=4)

    tracemalloc.start()
    start_time = time.time()

    for symbol, prices in symbols.items():
        scanner.add_symbol(symbol, prices)

    batch_results = scanner.calculate_all(ma_type, length, use_parallel=True)

    batch_time = time.time() - start_time
    _, batch_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Batch calculation (parallel):")
    print(f"  Time: {batch_time:.4f}s")
    print(f"  Memory: {batch_memory / 1024 / 1024:.2f} MB")

    # Batch calculation (serial)
    scanner_serial = BatchApproximateMAScanner(num_threads=4)

    tracemalloc.start()
    start_time = time.time()

    for symbol, prices in symbols.items():
        scanner_serial.add_symbol(symbol, prices)

    batch_serial_results = scanner_serial.calculate_all(ma_type, length, use_parallel=False)

    batch_serial_time = time.time() - start_time
    _, batch_serial_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Batch calculation (serial):")
    print(f"  Time: {batch_serial_time:.4f}s")
    print(f"  Memory: {batch_serial_memory / 1024 / 1024:.2f} MB")

    # Calculate speedup
    parallel_speedup = individual_time / batch_time if batch_time > 0 else 0
    serial_speedup = individual_time / batch_serial_time if batch_serial_time > 0 else 0

    print(f"\nSpeedup:")
    print(f"  Parallel vs Individual: {parallel_speedup:.2f}x")
    print(f"  Serial vs Individual: {serial_speedup:.2f}x")

    return {
        "individual_time": individual_time,
        "batch_parallel_time": batch_time,
        "batch_serial_time": batch_serial_time,
        "individual_memory_mb": individual_memory / 1024 / 1024,
        "batch_parallel_memory_mb": batch_memory / 1024 / 1024,
        "batch_serial_memory_mb": batch_serial_memory / 1024 / 1024,
        "parallel_speedup": parallel_speedup,
        "serial_speedup": serial_speedup,
    }


def benchmark_different_batch_sizes(
    batch_sizes: List[int] = [10, 50, 100, 500], ma_type: str = "EMA", length: int = 20
) -> Dict[str, List[float]]:
    """Benchmark different batch sizes.

    Args:
        batch_sizes: List of batch sizes to test
        ma_type: MA type to calculate
        length: MA length

    Returns:
        Dictionary with benchmark results for each batch size
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmark: Different Batch Sizes")
    print(f"MA Type: {ma_type}, Length: {length}")
    print(f"Batch Sizes: {batch_sizes}")
    print(f"{'=' * 60}")

    results = {
        "batch_sizes": batch_sizes,
        "individual_times": [],
        "batch_parallel_times": [],
        "parallel_speedups": [],
    }

    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")

        # Generate sample data
        symbols: Dict[str, pd.Series] = {}
        for i in range(batch_size):
            symbols[f"SYMBOL{i}"] = generate_sample_prices(100)

        # Individual calculation
        start_time = time.time()
        for symbol, prices in symbols.items():
            fast_ema_approx(prices, length)
        individual_time = time.time() - start_time

        # Batch calculation (parallel)
        scanner = BatchApproximateMAScanner(num_threads=4)
        for symbol, prices in symbols.items():
            scanner.add_symbol(symbol, prices)

        start_time = time.time()
        scanner.calculate_all(ma_type, length, use_parallel=True)
        batch_time = time.time() - start_time

        speedup = individual_time / batch_time if batch_time > 0 else 0

        print(f"Individual: {individual_time:.4f}s, Batch: {batch_time:.4f}s, Speedup: {speedup:.2f}x")

        results["individual_times"].append(individual_time)
        results["batch_parallel_times"].append(batch_time)
        results["parallel_speedups"].append(speedup)

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    for i, batch_size in enumerate(batch_sizes):
        print(f"  {batch_size:4d} symbols: {results['parallel_speedups'][i]:.2f}x speedup")
    print(f"{'=' * 60}")

    return results


def benchmark_all_ma_types(num_symbols: int = 50, length: int = 20) -> Dict[str, Dict[str, float]]:
    """Benchmark all MA types.

    Args:
        num_symbols: Number of symbols to process
        length: MA length

    Returns:
        Dictionary with benchmark results for each MA type
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmark: All MA Types")
    print(f"Symbols: {num_symbols}, Length: {length}")
    print(f"{'=' * 60}")

    ma_types = ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]
    results = {}

    for ma_type in ma_types:
        print(f"\n--- {ma_type} ---")

        # Generate sample data
        symbols: Dict[str, pd.Series] = {}
        for i in range(num_symbols):
            symbols[f"SYMBOL{i}"] = generate_sample_prices(100)

        # Individual calculation
        start_time = time.time()
        for symbol, prices in symbols.items():
            if ma_type == "EMA":
                fast_ema_approx(prices, length)
            elif ma_type == "HMA":
                fast_hma_approx(prices, length)
            individual_time = time.time() - start_time

        # Batch calculation
        scanner = BatchApproximateMAScanner(num_threads=4)
        for symbol, prices in symbols.items():
            scanner.add_symbol(symbol, prices)

        start_time = time.time()
        scanner.calculate_all(ma_type, length, use_parallel=True)
        batch_time = time.time() - start_time

        speedup = individual_time / batch_time if batch_time > 0 else 0

        print(f"Individual: {individual_time:.4f}s, Batch: {batch_time:.4f}s, Speedup: {speedup:.2f}x")

        results[ma_type] = {
            "individual_time": individual_time,
            "batch_time": batch_time,
            "speedup": speedup,
        }

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    for ma_type, result in results.items():
        print(f"  {ma_type}: {result['speedup']:.2f}x speedup")
    print(f"{'=' * 60}")

    return results


def benchmark_ma_sets(num_symbols: int = 50, base_length: int = 20, robustness: str = "Medium") -> Dict[str, float]:
    """Benchmark calculating sets of MAs (9 MAs per symbol).

    Args:
        num_symbols: Number of symbols to process
        base_length: Base MA length
        robustness: Robustness level

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmark: MA Sets (9 MAs per symbol)")
    print(f"Symbols: {num_symbols}, Base Length: {base_length}, Robustness: {robustness}")
    print(f"{'=' * 60}")

    # Generate sample data
    symbols: Dict[str, pd.Series] = {}
    for i in range(num_symbols):
        symbols[f"SYMBOL{i}"] = generate_sample_prices(100)

    # Batch calculation
    scanner = BatchApproximateMAScanner(num_threads=4)

    tracemalloc.start()
    start_time = time.time()

    for symbol, prices in symbols.items():
        scanner.add_symbol(symbol, prices)

    results = scanner.calculate_set_of_mas("EMA", base_length, robustness=robustness)

    batch_time = time.time() - start_time
    _, batch_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"MA Sets calculation:")
    print(f"  Time: {batch_time:.4f}s")
    print(f"  Memory: {batch_memory / 1024 / 1024:.2f} MB")
    print(f"  Symbols processed: {len(results) if results else 0}/{num_symbols}")

    return {
        "batch_time": batch_time,
        "memory_mb": batch_memory / 1024 / 1024,
        "symbols_processed": len(results) if results else 0,
    }


def benchmark_adaptive_mode(num_symbols: int = 50, ma_type: str = "EMA", length: int = 20) -> Dict[str, float]:
    """Benchmark adaptive mode vs standard mode.

    Args:
        num_symbols: Number of symbols to process
        ma_type: MA type to calculate
        length: MA length

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmark: Adaptive Mode vs Standard Mode")
    print(f"Symbols: {num_symbols}, MA Type: {ma_type}, Length: {length}")
    print(f"{'=' * 60}")

    # Generate sample data
    symbols: Dict[str, pd.Series] = {}
    for i in range(num_symbols):
        symbols[f"SYMBOL{i}"] = generate_sample_prices(100)

    # Standard mode
    scanner_standard = BatchApproximateMAScanner(use_adaptive=False, num_threads=4)

    tracemalloc.start()
    start_time = time.time()

    for symbol, prices in symbols.items():
        scanner_standard.add_symbol(symbol, prices)

    scanner_standard.calculate_all(ma_type, length, use_parallel=True)

    standard_time = time.time() - start_time
    _, standard_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Standard mode:")
    print(f"  Time: {standard_time:.4f}s")
    print(f"  Memory: {standard_memory / 1024 / 1024:.2f} MB")

    # Adaptive mode
    scanner_adaptive = BatchApproximateMAScanner(use_adaptive=True, num_threads=4)

    tracemalloc.start()
    start_time = time.time()

    for symbol, prices in symbols.items():
        scanner_adaptive.add_symbol(symbol, prices)

    scanner_adaptive.calculate_all(ma_type, length, use_parallel=True)

    adaptive_time = time.time() - start_time
    _, adaptive_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Adaptive mode:")
    print(f"  Time: {adaptive_time:.4f}s")
    print(f"  Memory: {adaptive_memory / 1024 / 1024:.2f} MB")

    overhead_ratio = adaptive_time / standard_time if standard_time > 0 else 0

    print(f"\nOverhead:")
    print(f"  Time overhead: {(overhead_ratio - 1) * 100:.2f}%")
    print(f"  Memory overhead: {(adaptive_memory / standard_memory - 1) * 100:.2f}%")

    return {
        "standard_time": standard_time,
        "adaptive_time": adaptive_time,
        "standard_memory_mb": standard_memory / 1024 / 1024,
        "adaptive_memory_mb": adaptive_memory / 1024 / 1024,
        "overhead_ratio": overhead_ratio,
    }


def run_all_benchmarks() -> None:
    """Run all benchmarks and print summary."""
    print("\n" + "=" * 60)
    print("BATCH APPROXIMATE MA SCANNER - BENCHMARK SUITE")
    print("=" * 60)

    # Benchmark 1: Batch vs Individual
    result1 = benchmark_batch_vs_individual(num_symbols=100, ma_type="EMA", length=20)

    # Benchmark 2: Different batch sizes
    result2 = benchmark_different_batch_sizes(batch_sizes=[10, 50, 100, 500], ma_type="EMA", length=20)

    # Benchmark 3: All MA types
    result3 = benchmark_all_ma_types(num_symbols=50, length=20)

    # Benchmark 4: MA Sets
    result4 = benchmark_ma_sets(num_symbols=50, base_length=20, robustness="Medium")

    # Benchmark 5: Adaptive mode
    result5 = benchmark_adaptive_mode(num_symbols=50, ma_type="EMA", length=20)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Batch vs Individual (100 symbols): {result1['parallel_speedup']:.2f}x speedup")
    print(f"Best batch size: {result2['batch_sizes'][np.argmax(result2['parallel_speedups'])]} symbols")
    print(f"Fastest MA type: {min(result3.keys(), key=lambda k: result3[k]['speedup'])}")
    print(f"MA Sets (50 symbols, 9 MAs each): {result4['batch_time']:.4f}s")
    print(f"Adaptive mode overhead: {(result5['overhead_ratio'] - 1) * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_all_benchmarks()
