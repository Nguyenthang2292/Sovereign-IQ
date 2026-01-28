"""Benchmark memory reduction with memory-mapped vs normal backtest path."""

import gc
import os
import sys
import tempfile
from typing import Dict

import numpy as np
import pandas as pd

try:
    from modules.adaptive_trend_LTS.core.backtesting.dask_backtest import backtest_with_dask
    from modules.adaptive_trend_LTS.utils.memory_mapped_data import (
        MemoryMappedDataManager,
        cleanup as cleanup_mmap,
    )
except ImportError:
    print("Warning: Some modules not available")

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available, memory tracking limited")


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    else:
        return sys.getsizeof([]) / 1024 / 1024


def get_peak_memory_mb() -> float:
    """Get peak memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process()
        mem_info = process.memory_info()
        return max(mem_info.rss, getattr(mem_info, "vms", 0)) / 1024 / 1024
    else:
        return get_memory_usage_mb()


def generate_test_csv(n_symbols: int, n_bars: int = 1500, output_path: str = None) -> str:
    """Generate test CSV file with synthetic data.

    Args:
        n_symbols: Number of symbols
        n_bars: Number of data bars per symbol
        output_path: Path to save CSV (creates temp file if None)

    Returns:
        Path to generated CSV file
    """
    if output_path is None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

    np.random.seed(42)

    print(f"Generating test data: {n_symbols} symbols x {n_bars} bars")

    with open(output_path, "w") as f:
        f.write("symbol,close,high,low,open,volume\n")

        for i in range(n_symbols):
            symbol = f"SYMBOL_{i}"
            base_price = 100 + i * 10

            for j in range(n_bars):
                close = base_price + np.random.normal(0, 5)
                high = close + abs(np.random.normal(0, 2))
                low = close - abs(np.random.normal(0, 2))
                open_price = close + np.random.normal(0, 1)
                volume = np.random.randint(1000, 10000)

                f.write(f"{symbol},{close:.2f},{high:.2f},{low:.2f},{open_price:.2f},{volume}\n")

    print(f"Generated test data: {output_path}")
    return output_path


def benchmark_normal_path(
    csv_path: str,
    atc_config: dict,
    chunksize: str = "100MB",
) -> tuple:
    """Benchmark normal backtest path memory usage.

    Args:
        csv_path: Path to CSV file
        atc_config: ATC configuration
        chunksize: Dask chunksize

    Returns:
        Tuple of (memory_delta_mb, peak_memory_mb, result_count)
    """
    gc.collect()
    initial_memory = get_memory_usage_mb()

    print(f"  Running normal path...")
    result = backtest_with_dask(
        historical_data_path=csv_path,
        atc_config=atc_config,
        chunksize=chunksize,
        use_memory_mapped=False,
    )

    gc.collect()
    peak_memory = get_peak_memory_mb()
    memory_delta = peak_memory - initial_memory
    result_count = len(result) if result is not None else 0

    print(f"    Memory delta: {memory_delta:.2f} MB")
    print(f"    Peak memory: {peak_memory:.2f} MB")
    print(f"    Results: {result_count} rows")

    return memory_delta, peak_memory, result_count


def benchmark_memmap_path(
    csv_path: str,
    atc_config: dict,
    chunksize: str = "100MB",
    cache_dir: str = ".cache/mmap",
) -> tuple:
    """Benchmark memory-mapped backtest path memory usage.

    Args:
        csv_path: Path to CSV file
        atc_config: ATC configuration
        chunksize: Dask chunksize
        cache_dir: Cache directory for memmap files

    Returns:
        Tuple of (memory_delta_mb, peak_memory_mb, result_count)
    """
    gc.collect()
    initial_memory = get_memory_usage_mb()

    print(f"  Running memory-mapped path...")
    result = backtest_with_dask(
        historical_data_path=csv_path,
        atc_config=atc_config,
        chunksize=chunksize,
        use_memory_mapped=True,
    )

    gc.collect()
    peak_memory = get_peak_memory_mb()
    memory_delta = peak_memory - initial_memory
    result_count = len(result) if result is not None else 0

    print(f"    Memory delta: {memory_delta:.2f} MB")
    print(f"    Peak memory: {peak_memory:.2f} MB")
    print(f"    Results: {result_count} rows")

    return memory_delta, peak_memory, result_count


def benchmark_memory_comparison(
    dataset_sizes: list[int] = [10, 50, 100, 500, 1000],
    n_bars: int = 1500,
    chunksize: str = "100MB",
) -> None:
    """Compare memory usage between normal and memory-mapped paths.

    Args:
        dataset_sizes: List of symbol counts to test
        n_bars: Number of data bars per symbol
        chunksize: Dask chunksize
    """
    atc_config = {
        "ema_len": 28,
        "atc_period": 14,
        "volatility_window": 20,
        "robustness": "Medium",
        "La": 0.02,
        "De": 0.03,
        "cutout": 0,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
    }

    print("=" * 80)
    print("MEMORY MAPPED ARRAYS BENCHMARK")
    print("Comparing normal vs memory-mapped backtest paths")
    print("=" * 80)

    results = []

    for n_symbols in dataset_sizes:
        print(f"\n{'=' * 80}")
        print(f"Testing with {n_symbols} symbols x {n_bars} bars")
        print(f"{'=' * 80}")

        total_rows = n_symbols * n_bars
        print(f"Total rows: {total_rows:,}")
        print(f"Estimated file size: {total_rows * 64 / 1024 / 1024:.2f} MB")

        csv_path = generate_test_csv(n_symbols, n_bars)

        try:
            print("\n[Normal Path]")
            normal_delta, normal_peak, normal_count = benchmark_normal_path(csv_path, atc_config, chunksize)

            print("\n[Memory-Mapped Path]")
            memmap_delta, memmap_peak, memmap_count = benchmark_memmap_path(csv_path, atc_config, chunksize)

            memory_reduction = (1 - memmap_delta / normal_delta) * 100 if normal_delta > 0 else 0

            print(f"\n[Comparison]")
            print(f"  Normal memory delta: {normal_delta:.2f} MB")
            print(f"  Memmap memory delta: {memmap_delta:.2f} MB")
            print(f"  Memory reduction: {memory_reduction:+.1f}%")
            print(f"  Peak reduction: {(1 - memmap_peak / normal_peak) * 100:+.1f}%")

            results.append(
                {
                    "n_symbols": n_symbols,
                    "total_rows": total_rows,
                    "normal_memory_mb": normal_delta,
                    "normal_peak_mb": normal_peak,
                    "memmap_memory_mb": memmap_delta,
                    "memmap_peak_mb": memmap_peak,
                    "memory_reduction_percent": memory_reduction,
                    "peak_reduction_percent": (1 - memmap_peak / normal_peak) * 100,
                }
            )

        except Exception as e:
            print(f"\nError during benchmark: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
            print(f"Cleaned up test file: {csv_path}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Symbols':>10} {'Rows':>12} {'Normal (MB)':>14} {'Memmap (MB)':>14} {'Reduction %':>14}")
    print("-" * 80)

    for result in results:
        print(
            f"{result['n_symbols']:>10} "
            f"{result['total_rows']:>12,} "
            f"{result['normal_memory_mb']:>14.2f} "
            f"{result['memmap_memory_mb']:>14.2f} "
            f"{result['memory_reduction_percent']:>+14.1f}"
        )

    print("\n" + "=" * 80)
    print("PEAK MEMORY SUMMARY")
    print("=" * 80)
    print(f"{'Symbols':>10} {'Normal Peak (MB)':>18} {'Memmap Peak (MB)':>18} {'Peak Reduction %':>18}")
    print("-" * 80)

    for result in results:
        print(
            f"{result['n_symbols']:>10} "
            f"{result['normal_peak_mb']:>18.2f} "
            f"{result['memmap_peak_mb']:>18.2f} "
            f"{result['peak_reduction_percent']:>+18.1f}"
        )

    print("\n" + "=" * 80)
    print("DISK USAGE")
    print("=" * 80)

    try:
        cache_dir = ".cache/mmap"
        if os.path.exists(cache_dir):
            total_size = 0
            file_count = 0
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                    file_count += 1

            print(f"Cache directory: {cache_dir}")
            print(f"Total files: {file_count}")
            print(f"Total disk usage: {total_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"Could not calculate disk usage: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark memory-mapped vs normal backtest paths")
    parser.add_argument(
        "--symbols",
        type=int,
        nargs="+",
        default=[10, 50, 100, 500, 1000],
        help="List of symbol counts to test",
    )
    parser.add_argument("--bars", type=int, default=1500, help="Number of bars per symbol")
    parser.add_argument("--chunksize", type=str, default="100MB", help="Dask chunksize")

    args = parser.parse_args()

    benchmark_memory_comparison(
        dataset_sizes=args.symbols,
        n_bars=args.bars,
        chunksize=args.chunksize,
    )
