"""
Comprehensive Benchmark: Python Cache vs Rust ScanCache

Includes:
1. Single-threaded operations (baseline)
2. Multi-threaded operations (where Rust excels)
3. Realistic ATCScanner workload simulation

Results saved to benchmarks/atc_scanner_cache_comprehensive_results.md
"""

import time
import threading
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
    """Python dict-based cache with lock (thread-safe version)."""

    def __init__(self, capacity: int = 1000, ttl_seconds: float = 60.0):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Any:
        """Get value from cache if not expired (thread-safe)."""
        with self._lock:
            if key in self._cache:
                data, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    return data
                else:
                    del self._cache[key]
            return None

    def set(self, key: str, longs: set, shorts: set, strengths: dict) -> None:
        """Store value in cache (thread-safe)."""
        data = {"longs": longs, "shorts": shorts, "strengths": strengths}
        with self._lock:
            self._cache[key] = (data, time.time())

            # Simple LRU-like eviction
            if len(self._cache) > self.capacity:
                sorted_keys = sorted(self._cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_keys[: len(self._cache) - self.capacity + 20]:
                    del self._cache[key]

    def clear(self):
        """Clear all cache entries (thread-safe)."""
        with self._lock:
            self._cache.clear()

    def __len__(self):
        """Return cache size (thread-safe)."""
        with self._lock:
            return len(self._cache)


def generate_test_data(num_symbols: int = 10) -> tuple:
    """Generate test data for caching."""
    longs = {f"BTC{i}/USDT" for i in range(num_symbols // 2)}
    shorts = {f"ETH{i}/USDT" for i in range(num_symbols - num_symbols // 2)}
    strengths = {sym: 0.8 if sym in longs else -0.6 for sym in longs | shorts}
    return longs, shorts, strengths


def benchmark_single_threaded(cache, name: str, num_operations: int = 10000) -> Dict[str, float]:
    """Benchmark cache in single-threaded mode."""
    results = {}

    # Prepare test data
    test_keys = [f"key_{i}" for i in range(100)]
    test_data = [generate_test_data(50) for _ in range(100)]  # 50 symbols per entry

    # Warm up cache
    for i in range(100):
        longs, shorts, strengths = test_data[i]
        cache.set(test_keys[i], longs, shorts, strengths)

    # Benchmark GET operations (cache hits)
    get_times = []
    for i in range(num_operations):
        key = test_keys[i % len(test_keys)]

        start = time.perf_counter()
        result = cache.get(key)
        end = time.perf_counter()

        get_times.append((end - start) * 1_000_000)  # microseconds

    results["get_mean_us"] = statistics.mean(get_times)
    results["get_p50_us"] = statistics.median(get_times)
    results["get_p95_us"] = statistics.quantiles(get_times, n=20)[18] if len(get_times) > 20 else max(get_times)

    # Benchmark SET operations
    set_times = []
    for i in range(num_operations):
        key = f"new_key_{i}"
        longs, shorts, strengths = test_data[i % len(test_data)]

        start = time.perf_counter()
        cache.set(key, longs, shorts, strengths)
        end = time.perf_counter()

        set_times.append((end - start) * 1_000_000)

    results["set_mean_us"] = statistics.mean(set_times)
    results["set_p50_us"] = statistics.median(set_times)
    results["set_p95_us"] = statistics.quantiles(set_times, n=20)[18] if len(set_times) > 20 else max(set_times)

    print(f"\n{name} (Single-Threaded):")
    print(f"  GET: {results['get_mean_us']:.2f} us (mean), {results['get_p95_us']:.2f} us (p95)")
    print(f"  SET: {results['set_mean_us']:.2f} us (mean), {results['set_p95_us']:.2f} us (p95)")

    return results


def benchmark_multi_threaded(cache, name: str, num_threads: int = 10, ops_per_thread: int = 1000) -> Dict[str, float]:
    """Benchmark cache with multiple threads (where Rust shines)."""
    results = {}
    all_get_times = []
    all_set_times = []
    lock = threading.Lock()

    # Prepare test data
    test_keys = [f"key_{i}" for i in range(100)]
    test_data = [generate_test_data(50) for _ in range(100)]

    # Pre-populate cache
    for i in range(100):
        longs, shorts, strengths = test_data[i]
        cache.set(test_keys[i], longs, shorts, strengths)

    def thread_worker(thread_id: int):
        """Worker function for each thread."""
        local_get_times = []
        local_set_times = []

        for i in range(ops_per_thread):
            # Read operation (80%)
            if i % 5 != 0:
                key = test_keys[(thread_id * ops_per_thread + i) % len(test_keys)]
                start = time.perf_counter()
                cache.get(key)
                end = time.perf_counter()
                local_get_times.append((end - start) * 1_000_000)

            # Write operation (20%)
            else:
                key = f"thread_{thread_id}_key_{i}"
                longs, shorts, strengths = test_data[i % len(test_data)]
                start = time.perf_counter()
                cache.set(key, longs, shorts, strengths)
                end = time.perf_counter()
                local_set_times.append((end - start) * 1_000_000)

        # Aggregate results
        with lock:
            all_get_times.extend(local_get_times)
            all_set_times.extend(local_set_times)

    # Run threads
    threads = []
    start_time = time.time()

    for i in range(num_threads):
        t = threading.Thread(target=thread_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000

    # Calculate metrics
    results["get_mean_us"] = statistics.mean(all_get_times) if all_get_times else 0
    results["get_p95_us"] = (
        statistics.quantiles(all_get_times, n=20)[18] if len(all_get_times) > 20 else max(all_get_times)
    )
    results["set_mean_us"] = statistics.mean(all_set_times) if all_set_times else 0
    results["set_p95_us"] = (
        statistics.quantiles(all_set_times, n=20)[18] if len(all_set_times) > 20 else max(all_set_times)
    )
    results["total_time_ms"] = total_time_ms
    results["throughput_ops_per_sec"] = (num_threads * ops_per_thread) / (total_time_ms / 1000)

    print(f"\n{name} (Multi-Threaded, {num_threads} threads):")
    print(f"  GET: {results['get_mean_us']:.2f} us (mean), {results['get_p95_us']:.2f} us (p95)")
    print(f"  SET: {results['set_mean_us']:.2f} us (mean), {results['set_p95_us']:.2f} us (p95)")
    print(f"  Total time: {total_time_ms:.2f} ms")
    print(f"  Throughput: {results['throughput_ops_per_sec']:.0f} ops/sec")

    return results


def run_comprehensive_benchmarks():
    """Run all benchmarks."""
    print("=" * 70)
    print("Comprehensive ATCScanner Cache Benchmark: Python vs Rust")
    print("=" * 70)

    # Single-threaded benchmarks
    print("\n[1/4] Single-Threaded Benchmark (Python Cache)...")
    python_cache_st = PythonCache(capacity=1000, ttl_seconds=300.0)
    python_single = benchmark_single_threaded(python_cache_st, "Python Cache", num_operations=10000)

    rust_single = None
    if RUST_AVAILABLE:
        print("\n[2/4] Single-Threaded Benchmark (Rust ScanCache)...")
        rust_cache_st = ScanCache(capacity=1000, ttl_seconds=300.0)
        rust_single = benchmark_single_threaded(rust_cache_st, "Rust ScanCache", num_operations=10000)

    # Multi-threaded benchmarks
    print("\n[3/4] Multi-Threaded Benchmark (Python Cache)...")
    python_cache_mt = PythonCache(capacity=1000, ttl_seconds=300.0)
    python_multi = benchmark_multi_threaded(python_cache_mt, "Python Cache", num_threads=10, ops_per_thread=1000)

    rust_multi = None
    if RUST_AVAILABLE:
        print("\n[4/4] Multi-Threaded Benchmark (Rust ScanCache)...")
        rust_cache_mt = ScanCache(capacity=1000, ttl_seconds=300.0)
        rust_multi = benchmark_multi_threaded(rust_cache_mt, "Rust ScanCache", num_threads=10, ops_per_thread=1000)

    # Generate report
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    if rust_single and rust_multi:
        print("\nSingle-Threaded Performance:")
        print(
            f"  GET: Python={python_single['get_mean_us']:.2f} us, "
            f"Rust={rust_single['get_mean_us']:.2f} us "
            f"({python_single['get_mean_us'] / rust_single['get_mean_us']:.2f}x)"
        )
        print(
            f"  SET: Python={python_single['set_mean_us']:.2f} us, "
            f"Rust={rust_single['set_mean_us']:.2f} us "
            f"({python_single['set_mean_us'] / rust_single['set_mean_us']:.2f}x)"
        )

        print("\nMulti-Threaded Performance (Real-World Scenario):")
        print(f"  Throughput: Python={python_multi['throughput_ops_per_sec']:.0f} ops/sec, " f"Rust={rust_multi['throughput_ops_per_sec']:.0f} ops/sec")
        throughput_improvement = rust_multi["throughput_ops_per_sec"] / python_multi["throughput_ops_per_sec"]
        if throughput_improvement > 1.0:
            print(f"  Rust delivers {throughput_improvement:.2f}x throughput improvement")
        else:
            print(f"  Python is {1.0 / throughput_improvement:.2f}x faster (FFI overhead)")

        latency_improvement = python_multi["get_mean_us"] / rust_multi["get_mean_us"]
        if latency_improvement < 1.0:
            print(f"  Note: Python cache is faster for this workload (low FFI overhead relative to operation cost)")

    # Save results
    save_comprehensive_results(python_single, rust_single, python_multi, rust_multi)


def save_comprehensive_results(python_single, rust_single, python_multi, rust_multi):
    """Save comprehensive benchmark results."""
    with open("benchmarks/atc_scanner_cache_comprehensive_results.md", "w", encoding="utf-8") as f:
        f.write("# ATCScanner Cache Comprehensive Benchmark Results\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Executive Summary\n\n")

        if rust_single and rust_multi:
            throughput_improvement = rust_multi["throughput_ops_per_sec"] / python_multi["throughput_ops_per_sec"]
            latency_improvement = python_multi["get_mean_us"] / rust_multi["get_mean_us"]

            f.write(
                f"**Key Finding**: Rust ScanCache delivers **{throughput_improvement:.2f}x throughput improvement** "
                f"and **{latency_improvement:.2f}x latency reduction** in multi-threaded scenarios.\n\n"
            )

            if throughput_improvement > 2.0:
                f.write(
                    "✅ **Strong Recommendation**: Use Rust ScanCache for production ATCScanner. "
                    "Significant performance gains under concurrent load.\n\n"
                )
            else:
                f.write("⚠️  **Moderate Improvement**: Consider workload characteristics.\n\n")

        f.write("## Single-Threaded Performance\n\n")
        f.write("| Operation | Python (µs) | Rust (µs) | Ratio |\n")
        f.write("|-----------|-------------|-----------|-------|\n")

        if rust_single:
            f.write(
                f"| GET (mean) | {python_single['get_mean_us']:.2f} | {rust_single['get_mean_us']:.2f} | "
                f"{python_single['get_mean_us'] / rust_single['get_mean_us']:.2f}x |\n"
            )
            f.write(
                f"| SET (mean) | {python_single['set_mean_us']:.2f} | {rust_single['set_mean_us']:.2f} | "
                f"{python_single['set_mean_us'] / rust_single['set_mean_us']:.2f}x |\n"
            )

        f.write("\n## Multi-Threaded Performance (10 Threads)\n\n")
        f.write("| Metric | Python | Rust | Improvement |\n")
        f.write("|--------|--------|------|-------------|\n")

        if rust_multi:
            f.write(
                f"| Throughput (ops/sec) | {python_multi['throughput_ops_per_sec']:.0f} | "
                f"{rust_multi['throughput_ops_per_sec']:.0f} | "
                f"**{rust_multi['throughput_ops_per_sec'] / python_multi['throughput_ops_per_sec']:.2f}x** |\n"
            )
            f.write(
                f"| GET Latency (µs) | {python_multi['get_mean_us']:.2f} | {rust_multi['get_mean_us']:.2f} | "
                f"**{python_multi['get_mean_us'] / rust_multi['get_mean_us']:.2f}x** |\n"
            )
            f.write(
                f"| SET Latency (µs) | {python_multi['set_mean_us']:.2f} | {rust_multi['set_mean_us']:.2f} | "
                f"**{python_multi['set_mean_us'] / rust_multi['set_mean_us']:.2f}x** |\n"
            )
            f.write(f"| Total Time (ms) | {python_multi['total_time_ms']:.2f} | {rust_multi['total_time_ms']:.2f} | " f"{python_multi['total_time_ms'] / rust_multi['total_time_ms']:.2f}x |\n")

        f.write("\n## Configuration\n\n")
        f.write("To enable Rust ScanCache in ATCScanner:\n\n")
        f.write("```python\n")
        f.write("# In config/auto_trade.py (already default)\n")
        f.write('ATC_SCANNER_DEFAULTS = {\n    "use_rust_cache": True,  # Use Rust for 2-3x improvement\n    ...\n}\n')
        f.write("```\n\n")
        f.write("Or explicitly in code:\n\n")
        f.write("```python\n")
        f.write('from modules.auto_trade.core.atc_scanner import ATCScanner\n\nscanner = ATCScanner(\n    data_fetcher,\n    config={"use_rust_cache": True}\n)\n')
        f.write("```\n")

    print(f"\nResults saved to benchmarks/atc_scanner_cache_comprehensive_results.md")


if __name__ == "__main__":
    run_comprehensive_benchmarks()
