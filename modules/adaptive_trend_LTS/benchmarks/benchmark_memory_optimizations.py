import os
import sys
import time
import timeit
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import tempfile

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not installed. Memory measurements will be skipped.")

from modules.adaptive_trend_LTS.utils.memory_mapped_data import (
    create_memory_mapped_from_csv,
    load_memory_mapped_from_csv,
)
from modules.adaptive_trend_LTS.utils.data_compression import (
    compress_to_file,
    decompress_from_file,
    compress_pickle,
    decompress_pickle,
)


def generate_dummy_data(rows=1_000_000):
    """Generate dummy price data."""
    print(f"Generating {rows:,} rows of dummy data...")
    dates = pd.date_range(start="2020-01-01", periods=rows, freq="1min")
    np.random.seed(42)
    prices = 10000 + np.cumsum(np.random.randn(rows))

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["BTCUSDT"] * rows,
            "open": prices,
            "high": prices + np.random.rand(rows) * 10,
            "low": prices - np.random.rand(rows) * 10,
            "close": prices + np.random.randn(rows),
            "volume": np.abs(np.random.randn(rows) * 100),
        }
    )
    return df


def measure_memory():
    """Measure current process memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return 0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_memory_mapped(df, temp_dir):
    """Benchmark memory-mapped arrays."""
    print("\n" + "=" * 50)
    print("BENCHMARK: Memory-Mapped Arrays")
    print("=" * 50)

    csv_path = os.path.join(temp_dir, "test_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Created CSV: {os.path.getsize(csv_path) / 1024 / 1024:.2f} MB")

    # Measure In-Memory Load
    gc_collect()
    mem_before = measure_memory()
    start_time = time.time()
    df_loaded = pd.read_csv(csv_path)
    load_time_ram = time.time() - start_time
    mem_after = measure_memory()
    mem_ram = mem_after - mem_before

    print(f"In-Memory Load Time: {load_time_ram:.4f}s")
    print(f"In-Memory Usage: {mem_ram:.2f} MB")

    del df_loaded
    gc_collect()

    # Measure Memory-Mapped Creation
    mem_before = measure_memory()
    start_time = time.time()
    descriptor = create_memory_mapped_from_csv(csv_path, cache_dir=os.path.join(temp_dir, "mmap_cache"), overwrite=True)
    create_time_mmap = time.time() - start_time
    mem_after = measure_memory()
    mem_mmap_create = mem_after - mem_before  # Should be low

    print(f"MMap Creation Time: {create_time_mmap:.4f}s")

    # Measure Memory-Mapped Load
    gc_collect()
    mem_before = measure_memory()
    start_time = time.time()
    _, mmap_array = load_memory_mapped_from_csv(csv_path, cache_dir=os.path.join(temp_dir, "mmap_cache"))
    load_time_mmap = time.time() - start_time

    # Access some data to trigger paging
    _ = mmap_array["close"][:1000]
    _ = mmap_array["close"][-1000:]

    mem_after = measure_memory()
    mem_mmap_load = mem_after - mem_before

    print(f"MMap Load Time: {load_time_mmap:.4f}s")
    print(f"MMap Usage: {mem_mmap_load:.2f} MB")

    # Validation
    print("-" * 20)
    print(f"Memory Reduction: {(1 - mem_mmap_load / mem_ram) * 100:.1f}%" if mem_ram > 0 else "N/A")

    return {
        "ram_load_time": load_time_ram,
        "ram_memory": mem_ram,
        "mmap_create_time": create_time_mmap,
        "mmap_load_time": load_time_mmap,
        "mmap_memory": mem_mmap_load,
    }


def benchmark_compression(df, temp_dir):
    """Benchmark data compression."""
    print("\n" + "=" * 50)
    print("BENCHMARK: Data Compression (Blosc)")
    print("=" * 50)

    # Convert to simple numpy array for compression test
    prices = df["close"].values
    original_size = prices.nbytes
    print(f"Original Data Size: {original_size / 1024 / 1024:.2f} MB")

    results = []

    # Test different compression levels
    for level in [1, 5, 9]:
        start_time = time.time()
        compressed = compress_pickle(prices, compression_level=level)
        compress_time = time.time() - start_time

        compressed_size = len(compressed)

        start_time = time.time()
        decompressed = decompress_pickle(compressed)
        decompress_time = time.time() - start_time

        ratio = original_size / compressed_size

        print(f"Level {level}:")
        print(f"  Ratio: {ratio:.2f}x")
        print(f"  Size: {compressed_size / 1024 / 1024:.2f} MB")
        print(f"  Compress Time: {compress_time:.4f}s")
        print(f"  Decompress Time: {decompress_time:.4f}s")

        # Verify integrity
        np.testing.assert_array_equal(prices, decompressed)

        results.append(
            {"level": level, "ratio": ratio, "compress_time": compress_time, "decompress_time": decompress_time}
        )

    return results


def gc_collect():
    import gc

    gc.collect()


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Running benchmarks in temporary directory: {temp_dir}")

        # Generate Data
        df = generate_dummy_data(rows=1_000_000)  # ~1 million rows

        # Run Benchmarks
        benchmark_memory_mapped(df, temp_dir)
        benchmark_compression(df, temp_dir)


if __name__ == "__main__":
    main()
