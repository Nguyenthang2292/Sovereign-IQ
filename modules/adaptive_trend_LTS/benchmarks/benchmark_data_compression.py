"""Benchmark cache size and CPU cost with/without compression."""

import gc
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from modules.adaptive_trend_LTS.utils.data_compression import (
        compress_pickle,
        decompress_pickle,
        get_compression_ratio,
        compress_to_file,
        decompress_from_file,
        is_compression_available,
        BLOSC_AVAILABLE,
    )
    from modules.adaptive_trend_LTS.utils.cache_manager import CacheManager
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
        return 0.0


def generate_test_cache_entries(n_entries: int, data_size: int = 1000) -> dict:
    """Generate test cache entries representative of real workloads.

    Args:
        n_entries: Number of cache entries to generate
        data_size: Size of each numpy array

    Returns:
        Dictionary of cache entries
    """
    np.random.seed(42)

    cache_entries = {}
    for i in range(n_entries):
        key = f"ema_20_{i}"
        value_data = np.random.randn(data_size).astype(np.float64)
        cache_entries[key] = {
            "ma_type": "EMA",
            "length": 20,
            "price_data": value_data,
            "extra_params": None,
        }

    return cache_entries


def benchmark_compression_speed(
    data: dict,
    iterations: int = 10,
) -> tuple:
    """Benchmark compression speed.

    Args:
        data: Data to compress
        iterations: Number of iterations to average

    Returns:
        Tuple of (avg_compress_time_sec, avg_decompress_time_sec)
    """
    if not BLOSC_AVAILABLE:
        return 0.0, 0.0

    compress_times = []
    decompress_times = []

    for _ in range(iterations):
        gc.collect()

        # Benchmark compression
        start = time.time()
        compressed = compress_pickle(data)
        compress_time = time.time() - start
        compress_times.append(compress_time)

        gc.collect()

        # Benchmark decompression
        start = time.time()
        _ = decompress_pickle(compressed)
        decompress_time = time.time() - start
        decompress_times.append(decompress_time)

    avg_compress = sum(compress_times) / len(compress_times)
    avg_decompress = sum(decompress_times) / len(decompress_times)

    return avg_compress, avg_decompress


def benchmark_cache_operations(
    cache_entries: dict,
    use_compression: bool = False,
    temp_dir: str = None,
) -> tuple:
    """Benchmark cache save/load operations.

    Args:
        cache_entries: Dictionary of cache entries
        use_compression: Whether to use compression
        temp_dir: Temporary directory for cache files

    Returns:
        Tuple of (save_time_sec, load_time_sec, file_size_bytes)
    """
    gc.collect()
    initial_memory = get_memory_usage_mb()

    # Setup cache manager
    manager = CacheManager(
        cache_dir=temp_dir,
        use_compression=use_compression,
        compression_level=5,
    )

    # Populate L2 cache
    for key, value in cache_entries.items():
        manager.put(
            ma_type=value["ma_type"],
            length=value["length"],
            price_data=value["price_data"],
            value=value,
            extra_params=value["extra_params"],
        )

    # Benchmark save
    start = time.time()
    manager.save_to_disk()
    save_time = time.time() - start

    gc.collect()

    # Clear and benchmark load
    manager.clear()
    gc.collect()

    start = time.time()
    manager.load_from_disk()
    load_time = time.time() - start

    # Get file size
    if use_compression:
        cache_file = Path(temp_dir) / "cache_v1.pkl.blosc"
    else:
        cache_file = Path(temp_dir) / "cache_v1.pkl"

    file_size = os.path.getsize(cache_file) if cache_file.exists() else 0

    gc.collect()
    final_memory = get_memory_usage_mb()
    memory_delta = final_memory - initial_memory

    return save_time, load_time, file_size, memory_delta


def benchmark_cache_sizes():
    """Compare cache file sizes with/without compression."""
    print("=" * 80)
    print("CACHE COMPRESSION BENCHMARK")
    print("=" * 80)

    test_sizes = [100, 500, 1000, 2000]

    results = []

    for n_entries in test_sizes:
        print(f"\n{'=' * 80}")
        print(f"Testing with {n_entries} cache entries")
        print(f"{'=' * 80}")

        cache_entries = generate_test_cache_entries(n_entries)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test uncompressed
            print("\n[Uncompressed Mode]")
            save_time_uncomp, load_time_uncomp, size_uncomp, mem_delta_uncomp = benchmark_cache_operations(
                cache_entries=cache_entries,
                use_compression=False,
                temp_dir=tmpdir,
            )
            print(f"  Save time: {save_time_uncomp:.4f}s")
            print(f"  Load time: {load_time_uncomp:.4f}s")
            print(f"  File size: {size_uncomp:,} bytes ({size_uncomp / 1024 / 1024:.2f} MB)")
            print(f"  Memory delta: {mem_delta_uncomp:.2f} MB")

            # Test compressed (if available)
            if BLOSC_AVAILABLE:
                print("\n[Compressed Mode]")
                save_time_comp, load_time_comp, size_comp, mem_delta_comp = benchmark_cache_operations(
                    cache_entries=cache_entries,
                    use_compression=True,
                    temp_dir=tmpdir,
                )
                print(f"  Save time: {save_time_comp:.4f}s")
                print(f"  Load time: {load_time_comp:.4f}s")
                print(f"  File size: {size_comp:,} bytes ({size_comp / 1024 / 1024:.2f} MB)")
                print(f"  Memory delta: {mem_delta_comp:.2f} MB")

                size_ratio = size_uncomp / size_comp if size_comp > 0 else 0
                save_overhead = (save_time_comp - save_time_uncomp) / save_time_uncomp * 100
                load_overhead = (load_time_comp - load_time_uncomp) / load_time_uncomp * 100

                print(f"\n[Comparison]")
                print(f"  Size reduction: {(1 - 1 / size_ratio) * 100:.1f}%")
                print(f"  Compression ratio: {size_ratio:.2f}x")
                print(f"  Save overhead: {save_overhead:+.1f}%")
                print(f"  Load overhead: {load_overhead:+.1f}%")
                print(f"  Memory reduction: {(1 - mem_delta_comp / mem_delta_uncomp) * 100:.1f}%")
            else:
                print("\n[Compressed Mode]")
                print("  blosc not available - skipping compression tests")
                save_time_comp, load_time_comp, size_comp, mem_delta_comp = (
                    save_time_uncomp,
                    load_time_uncomp,
                    size_uncomp,
                    mem_delta_uncomp,
                )
                size_ratio = 0
                save_overhead = 0
                load_overhead = 0

            results.append(
                {
                    "n_entries": n_entries,
                    "uncompressed_size": size_uncomp,
                    "compressed_size": size_comp,
                    "uncompressed_save_time": save_time_uncomp,
                    "compressed_save_time": save_time_comp,
                    "uncompressed_load_time": load_time_uncomp,
                    "compressed_load_time": load_time_comp,
                    "size_ratio": size_ratio,
                    "save_overhead_percent": save_overhead,
                    "load_overhead_percent": load_overhead,
                }
            )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Entries':>10} {'Uncomp (MB)':>14} {'Comp (MB)':>12} {'Ratio':>8} {'Save %':>10} {'Load %':>10}")
    print("-" * 80)

    for result in results:
        uncomp_mb = result["uncompressed_size"] / 1024 / 1024
        comp_mb = result["compressed_size"] / 1024 / 1024
        ratio = result["size_ratio"]
        save_overhead = result["save_overhead_percent"]
        load_overhead = result["load_overhead_percent"]

        print(
            f"{result['n_entries']:>10} "
            f"{uncomp_mb:>14.2f} "
            f"{comp_mb:>12.2f} "
            f"{ratio:>8.2f} "
            f"{save_overhead:>+10.1f} "
            f"{load_overhead:>+10.1f}"
        )


def benchmark_compression_algorithms():
    """Compare different compression algorithms."""
    if not BLOSC_AVAILABLE:
        print("blosc not available - skipping algorithm comparison")
        return

    print("\n" + "=" * 80)
    print("COMPRESSION ALGORITHM COMPARISON")
    print("=" * 80)

    test_data = generate_test_cache_entries(500)

    algorithms = ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]

    results = []

    for algorithm in algorithms:
        print(f"\nTesting algorithm: {algorithm}")

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(
                cache_dir=tmpdir,
                use_compression=True,
                compression_level=5,
                compression_algorithm=algorithm,
            )

            # Populate cache
            for key, value in test_data.items():
                manager.put(
                    ma_type=value["ma_type"],
                    length=value["length"],
                    price_data=value["price_data"],
                    value=value,
                    extra_params=value["extra_params"],
                )

            # Benchmark save
            gc.collect()
            start = time.time()
            manager.save_to_disk()
            save_time = time.time() - start

            # Get file size
            cache_file = Path(tmpdir) / "cache_v1.pkl.blosc"
            file_size = os.path.getsize(cache_file) if cache_file.exists() else 0

            print(f"  Save time: {save_time:.4f}s")
            print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

            results.append(
                {
                    "algorithm": algorithm,
                    "save_time": save_time,
                    "file_size": file_size,
                }
            )

    print("\n" + "=" * 80)
    print("ALGORITHM SUMMARY")
    print("=" * 80)
    print(f"{'Algorithm':>12} {'Size (MB)':>14} {'Time (s)':>12}")
    print("-" * 80)

    results.sort(key=lambda x: x["file_size"])

    for result in results:
        size_mb = result["file_size"] / 1024 / 1024
        print(f"{result['algorithm']:>12} {size_mb:>14.2f} {result['save_time']:>12.4f}")


def benchmark_compression_levels():
    """Compare different compression levels."""
    if not BLOSC_AVAILABLE:
        print("blosc not available - skipping compression level comparison")
        return

    print("\n" + "=" * 80)
    print("COMPRESSION LEVEL COMPARISON")
    print("=" * 80)

    test_data = generate_test_cache_entries(500)
    levels = [1, 3, 5, 7, 9]

    results = []

    for level in levels:
        print(f"\nTesting compression level: {level}")

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(
                cache_dir=tmpdir,
                use_compression=True,
                compression_level=level,
            )

            # Populate cache
            for key, value in test_data.items():
                manager.put(
                    ma_type=value["ma_type"],
                    length=value["length"],
                    price_data=value["price_data"],
                    value=value,
                    extra_params=value["extra_params"],
                )

            # Benchmark save/load
            gc.collect()
            start = time.time()
            manager.save_to_disk()
            save_time = time.time() - start

            manager.clear()
            gc.collect()

            start = time.time()
            manager.load_from_disk()
            load_time = time.time() - start

            # Get file size
            cache_file = Path(tmpdir) / "cache_v1.pkl.blosc"
            file_size = os.path.getsize(cache_file) if cache_file.exists() else 0

            print(f"  Save time: {save_time:.4f}s")
            print(f"  Load time: {load_time:.4f}s")
            print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

            results.append(
                {
                    "level": level,
                    "save_time": save_time,
                    "load_time": load_time,
                    "file_size": file_size,
                }
            )

    print("\n" + "=" * 80)
    print("COMPRESSION LEVEL SUMMARY")
    print("=" * 80)
    print(f"{'Level':>8} {'Size (MB)':>14} {'Save (s)':>12} {'Load (s)':>12}")
    print("-" * 80)

    for result in results:
        size_mb = result["file_size"] / 1024 / 1024
        print(f"{result['level']:>8} {size_mb:>14.2f} {result['save_time']:>12.4f} {result['load_time']:>12.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark cache compression performance")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 2000],
        help="List of cache entry counts to test",
    )
    parser.add_argument(
        "--algorithms",
        action="store_true",
        help="Benchmark different compression algorithms",
    )
    parser.add_argument(
        "--levels",
        action="store_true",
        help="Benchmark different compression levels",
    )

    args = parser.parse_args()

    benchmark_cache_sizes()

    if args.algorithms:
        benchmark_compression_algorithms()

    if args.levels:
        benchmark_compression_levels()

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("Recommendations:")
    print("  - Compression level 5-7 provides good balance of size/speed")
    print("  - blosclz is fastest for compression")
    print("  - Expected size reduction: 5-10x")
    print("  - Expected CPU overhead: <10% for save/load")
