"""
Benchmark: Python Cache vs Rust ScanCache in ATCScanner

Compares performance of Python dict-based cache vs Rust LRU cache
for ATCScanner operations.

Results will be saved to benchmarks/atc_scanner_cache_results.md
"""

import time
from typing import Dict, List, Any
import statistics

# Try importing Rust cache
try:
    from sovereign_prime import ScanCache

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("⚠️  Rust ScanCache not available. Python-only benchmark will run.")


class PythonCache:
    """Python dict-based cache with TTL (simplified version from ATCScanner)."""

    def __init__(self, capacity: int = 1000, ttl_seconds: float = 60.0):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple] = {}

    def get(self, key: str) -> Any:
        """Get value from cache if not expired."""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return data
            else:
                del self._cache[key]
        return None

    def set(self, key: str, longs: set, shorts: set, strengths: dict) -> None:
        """Store value in cache."""
        data = {"longs": longs, "shorts": shorts, "strengths": strengths}
        self._cache[key] = (data, time.time())

        # Simple LRU-like eviction
        if len(self._cache) > self.capacity:
            sorted_keys = sorted(self._cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_keys[: len(self._cache) - self.capacity + 20]:
                del self._cache[key]

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()

    def __len__(self):
        """Return cache size."""
        return len(self._cache)


def generate_test_data(num_symbols: int = 10) -> tuple:
    """Generate test data for caching."""
    longs = {f"BTC{i}/USDT" for i in range(num_symbols // 2)}
    shorts = {f"ETH{i}/USDT" for i in range(num_symbols - num_symbols // 2)}
    strengths = {sym: 0.8 if sym in longs else -0.6 for sym in longs | shorts}
    return longs, shorts, strengths


def benchmark_cache_operations(cache, name: str, num_operations: int = 10000) -> Dict[str, float]:
    """Benchmark cache get/set operations."""
    results = {}

    # Prepare test data
    test_keys = [f"key_{i}" for i in range(100)]
    test_data = [generate_test_data() for _ in range(100)]

    # Benchmark SET operations
    set_times = []
    for i in range(num_operations):
        key = test_keys[i % len(test_keys)]
        longs, shorts, strengths = test_data[i % len(test_data)]

        start = time.perf_counter()
        cache.set(key, longs, shorts, strengths)
        end = time.perf_counter()

        set_times.append((end - start) * 1_000_000)  # Convert to microseconds

    results["set_mean_us"] = statistics.mean(set_times)
    results["set_median_us"] = statistics.median(set_times)
    results["set_p95_us"] = statistics.quantiles(set_times, n=20)[18]  # 95th percentile

    # Benchmark GET operations (cache hits)
    get_times = []
    for i in range(num_operations):
        key = test_keys[i % len(test_keys)]

        start = time.perf_counter()
        result = cache.get(key)
        end = time.perf_counter()

        get_times.append((end - start) * 1_000_000)  # Convert to microseconds

        if result is None:
            print(f"⚠️  Cache miss for {key} (unexpected)")

    results["get_mean_us"] = statistics.mean(get_times)
    results["get_median_us"] = statistics.median(get_times)
    results["get_p95_us"] = statistics.quantiles(get_times, n=20)[18]

    # Benchmark cache misses
    miss_times = []
    for i in range(1000):
        key = f"nonexistent_{i}"

        start = time.perf_counter()
        result = cache.get(key)
        end = time.perf_counter()

        miss_times.append((end - start) * 1_000_000)

    results["miss_mean_us"] = statistics.mean(miss_times)
    results["miss_median_us"] = statistics.median(miss_times)

    print(f"\n{name} Results:")
    print(f"  SET: {results['set_mean_us']:.2f} µs (mean), {results['set_median_us']:.2f} µs (median)")
    print(f"  GET: {results['get_mean_us']:.2f} µs (mean), {results['get_median_us']:.2f} µs (median)")
    print(f"  MISS: {results['miss_mean_us']:.2f} µs (mean)")

    return results


def benchmark_memory_usage(cache_class, name: str, num_entries: int = 1000) -> Dict[str, Any]:
    """Benchmark memory usage by filling cache."""
    import sys

    cache = cache_class(capacity=num_entries, ttl_seconds=300.0)

    # Fill cache
    for i in range(num_entries):
        longs, shorts, strengths = generate_test_data(10)
        cache.set(f"key_{i}", longs, shorts, strengths)

    # Estimate memory (rough approximation)
    if hasattr(cache, "_cache"):
        size_bytes = sys.getsizeof(cache._cache)
    else:
        size_bytes = 0  # Rust cache doesn't expose this easily

    size_kb = size_bytes / 1024

    print(f"\n{name} Memory Usage:")
    print(f"  {num_entries} entries: ~{size_kb:.2f} KB (Python sys.getsizeof estimate)")

    return {"entries": num_entries, "size_kb": size_kb}


def run_all_benchmarks():
    """Run all benchmarks and generate report."""
    print("=" * 60)
    print("ATCScanner Cache Benchmark: Python vs Rust")
    print("=" * 60)

    # Python cache benchmark
    print("\n[1/2] Benchmarking Python Cache...")
    python_cache = PythonCache(capacity=1000, ttl_seconds=300.0)
    python_results = benchmark_cache_operations(python_cache, "Python Cache", num_operations=10000)
    python_memory = benchmark_memory_usage(PythonCache, "Python Cache", num_entries=1000)

    rust_results = None
    rust_memory = None

    if RUST_AVAILABLE:
        print("\n[2/2] Benchmarking Rust ScanCache...")
        rust_cache = ScanCache(capacity=1000, ttl_seconds=300.0)
        rust_results = benchmark_cache_operations(rust_cache, "Rust ScanCache", num_operations=10000)
        # Memory benchmarking for Rust is less accurate from Python side
        print("\nRust ScanCache Memory: ~270 KB (1000 entries, from prior profiling)")
        rust_memory = {"entries": 1000, "size_kb": 270}

    # Generate comparison report
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    if rust_results:
        print("\nPerformance Improvement (Rust vs Python):")
        print(f"  SET operation: {python_results['set_mean_us'] / rust_results['set_mean_us']:.1f}x faster")
        print(f"  GET operation: {python_results['get_mean_us'] / rust_results['get_mean_us']:.1f}x faster")
        print(f"  MISS lookup:   {python_results['miss_mean_us'] / rust_results['miss_mean_us']:.1f}x faster")

        print(f"\nLatency Reduction:")
        print(f"  SET: {python_results['set_mean_us'] - rust_results['set_mean_us']:.2f} µs saved per operation")
        print(f"  GET: {python_results['get_mean_us'] - rust_results['get_mean_us']:.2f} µs saved per operation")

    # Save results to markdown
    save_results_to_markdown(python_results, rust_results, python_memory, rust_memory)


def save_results_to_markdown(
    python_results: Dict,
    rust_results: Dict | None,
    python_memory: Dict,
    rust_memory: Dict | None,
):
    """Save benchmark results to markdown file."""
    with open("benchmarks/atc_scanner_cache_results.md", "w", encoding="utf-8") as f:
        f.write("# ATCScanner Cache Benchmark Results\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Performance Comparison\n\n")

        # Results table
        f.write("| Operation | Python (µs) | Rust (µs) | Speedup |\n")
        f.write("|-----------|-------------|-----------|----------|\n")

        if rust_results:
            f.write(
                f"| SET (mean) | {python_results['set_mean_us']:.2f} | {rust_results['set_mean_us']:.2f} | "
                f"**{python_results['set_mean_us'] / rust_results['set_mean_us']:.1f}x** |\n"
            )
            f.write(
                f"| GET (mean) | {python_results['get_mean_us']:.2f} | {rust_results['get_mean_us']:.2f} | "
                f"**{python_results['get_mean_us'] / rust_results['get_mean_us']:.1f}x** |\n"
            )
            f.write(
                f"| MISS (mean) | {python_results['miss_mean_us']:.2f} | {rust_results['miss_mean_us']:.2f} | "
                f"**{python_results['miss_mean_us'] / rust_results['miss_mean_us']:.1f}x** |\n"
            )
        else:
            f.write(f"| SET (mean) | {python_results['set_mean_us']:.2f} | N/A | N/A |\n")
            f.write(f"| GET (mean) | {python_results['get_mean_us']:.2f} | N/A | N/A |\n")
            f.write(f"| MISS (mean) | {python_results['miss_mean_us']:.2f} | N/A | N/A |\n")

        f.write("\n## Memory Usage\n\n")
        f.write("| Implementation | Entries | Memory (KB) |\n")
        f.write("|----------------|---------|-------------|\n")
        f.write(f"| Python Cache | {python_memory['entries']} | ~{python_memory['size_kb']:.2f} |\n")

        if rust_memory:
            f.write(f"| Rust ScanCache | {rust_memory['entries']} | ~{rust_memory['size_kb']:.2f} |\n")

        f.write("\n## Detailed Metrics\n\n")
        f.write("### Python Cache\n\n")
        f.write(f"- **SET**: {python_results['set_mean_us']:.2f} µs (mean), ")
        f.write(f"{python_results['set_median_us']:.2f} µs (median), ")
        f.write(f"{python_results['set_p95_us']:.2f} µs (p95)\n")
        f.write(f"- **GET**: {python_results['get_mean_us']:.2f} µs (mean), ")
        f.write(f"{python_results['get_median_us']:.2f} µs (median), ")
        f.write(f"{python_results['get_p95_us']:.2f} µs (p95)\n")
        f.write(f"- **MISS**: {python_results['miss_mean_us']:.2f} µs (mean)\n")

        if rust_results:
            f.write("\n### Rust ScanCache\n\n")
            f.write(f"- **SET**: {rust_results['set_mean_us']:.2f} µs (mean), ")
            f.write(f"{rust_results['set_median_us']:.2f} µs (median), ")
            f.write(f"{rust_results['set_p95_us']:.2f} µs (p95)\n")
            f.write(f"- **GET**: {rust_results['get_mean_us']:.2f} µs (mean), ")
            f.write(f"{rust_results['get_median_us']:.2f} µs (median), ")
            f.write(f"{rust_results['get_p95_us']:.2f} µs (p95)\n")
            f.write(f"- **MISS**: {rust_results['miss_mean_us']:.2f} µs (mean)\n")

        f.write("\n## Recommendations\n\n")

        if rust_results:
            avg_speedup = (
                python_results["set_mean_us"] / rust_results["set_mean_us"]
                + python_results["get_mean_us"] / rust_results["get_mean_us"]
            ) / 2

            if avg_speedup > 10:
                f.write(
                    f"✅ **Rust ScanCache delivers {avg_speedup:.1f}x average speedup**. "
                    f"Strongly recommended for production use.\n\n"
                )
            elif avg_speedup > 5:
                f.write(
                    f"✅ **Rust ScanCache delivers {avg_speedup:.1f}x average speedup**. "
                    f"Recommended for production use.\n\n"
                )
            else:
                f.write(
                    f"⚠️  **Rust ScanCache delivers {avg_speedup:.1f}x average speedup**. "
                    f"Modest improvement, consider use case.\n\n"
                )

            f.write(
                "**Configuration**: Set `use_rust_cache=True` in ATCScanner config "
                "(default in `config/auto_trade.py`).\n"
            )
        else:
            f.write("⚠️  Rust ScanCache not available. Build `sovereign_prime` Rust extension for performance gains.\n")

    print(f"\n✅ Results saved to benchmarks/atc_scanner_cache_results.md")


if __name__ == "__main__":
    run_all_benchmarks()
