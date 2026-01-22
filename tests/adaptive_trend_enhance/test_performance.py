"""
Performance comparison tests between base and enhanced ATC implementations.

OPTIMIZED VERSION with:
- Environment variable controlled iterations
- Session-scoped fixtures for memory efficiency
- Warm-up cache for both base and enhanced versions
- Pytest markers for selective test execution
- Memory management and garbage collection
- Parametrized tests to reduce code duplication

Usage examples:
    # Fast development testing (3 iterations):
    pytest tests/adaptive_trend_enhance/test_performance.py -n 0 -m "not slow"

    # Full CI testing (10 iterations):
    PERF_ITERATIONS=10 pytest tests/adaptive_trend_enhance/test_performance.py -n 0 -m performance

    # Skip slow tests:
    pytest tests/adaptive_trend_enhance/test_performance.py -n 0 -m "performance and not slow"
"""

import gc
import os
import time
import tracemalloc
from typing import Any, Callable, List

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend.core.compute_atc_signals import compute_atc_signals as compute_base
from modules.adaptive_trend_enhance.core.compute_atc_signals import compute_atc_signals as compute_enhanced
from modules.common.system import get_memory_manager

# Optimization #1: Environment variable to control iterations
# Default: 3 for fast development, CI can set to 10 for thorough testing
PERF_ITERATIONS_FAST = int(os.getenv("PERF_ITERATIONS", "3"))
PERF_ITERATIONS_MEMORY = int(os.getenv("PERF_ITERATIONS_MEMORY", "5"))


# Optimization #2: Session-scoped fixtures (create once, reuse across all tests)
@pytest.fixture(scope="session")
def large_sample_data_session():
    """Create large sample price data once per test session for memory efficiency."""
    np.random.seed(42)
    n = 1500
    prices = pd.Series(
        100 * (1 + np.random.randn(n).cumsum() * 0.01),
        index=pd.date_range("2023-01-01", periods=n, freq="15min"),
    )
    return prices


# Optimization #3: Cache warm-up results (warm up once per session)
@pytest.fixture(scope="session")
def warmed_up_cache_both(large_sample_data_session):
    """Pre-warm cache for both base and enhanced versions once per test session."""
    # Warm up base version
    _ = compute_base(large_sample_data_session)
    gc.collect()

    # Warm up enhanced version
    _ = compute_enhanced(large_sample_data_session)
    gc.collect()

    return True


# Keep function-scoped fixture for backwards compatibility
@pytest.fixture
def large_sample_data(large_sample_data_session):
    """Function-scoped wrapper around session fixture."""
    return large_sample_data_session


# Optimization #5: Helper function with memory management and garbage collection
def benchmark_function(
    func: Callable[[], Any], iterations: int = PERF_ITERATIONS_FAST, warmup: bool = True
) -> List[float]:
    """
    Benchmark a function with proper memory management.

    Args:
        func: Function to benchmark (no arguments)
        iterations: Number of benchmark iterations
        warmup: Whether to perform warm-up run

    Returns:
        List of timing measurements in seconds
    """
    # Warm up if requested
    if warmup:
        _ = func()
        gc.collect()

    # Benchmark with memory management
    times = []
    for _ in range(iterations):
        gc.collect()  # Clean memory before each iteration
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times.append(end - start)
        del result

    return times


def print_comparison_stats(name_base: str, times_base: List[float], name_enhanced: str, times_enhanced: List[float]):
    """Print comparison statistics in a consistent format."""
    avg_base = np.mean(times_base)
    avg_enhanced = np.mean(times_enhanced)

    print(f"\nAverage Execution Time ({name_base}): {avg_base:.4f}s")
    print(f"Average Execution Time ({name_enhanced}): {avg_enhanced:.4f}s")

    if avg_enhanced < avg_base:
        speedup = avg_base / avg_enhanced
        print(f"Speedup: {speedup:.2f}x")
    else:
        slowdown = avg_enhanced / avg_base
        print(f"Slowdown: {slowdown:.2f}x (Note: Small datasets might be affected by overhead)")

    return avg_base, avg_enhanced


# Optimization #4: Add markers for selective test execution
@pytest.mark.performance
@pytest.mark.slow  # Mark as slow for skipping in fast development
def test_performance_comparison(large_sample_data, warmed_up_cache_both):
    """Compare performance between base and enhanced versions with optimized iterations."""
    # Use optimized benchmark function (already warmed up via fixture)
    times_base = benchmark_function(
        lambda: compute_base(large_sample_data),
        iterations=PERF_ITERATIONS_FAST,
        warmup=False,  # Already warmed up by fixture
    )

    times_enhanced = benchmark_function(
        lambda: compute_enhanced(large_sample_data),
        iterations=PERF_ITERATIONS_FAST,
        warmup=False,  # Already warmed up by fixture
    )

    avg_base, avg_enhanced = print_comparison_stats("Base", times_base, "Enhanced", times_enhanced)

    # Both must complete in reasonable time. Enhanced can be slower than Base when GPU
    # falls back to CPU (e.g. CuPy kernel errors) or on small data due to overhead.
    assert avg_base < 60.0, f"Base too slow: {avg_base:.2f}s"
    assert avg_enhanced < 60.0, f"Enhanced too slow: {avg_enhanced:.2f}s"


@pytest.mark.performance
def test_memory_leak_check(large_sample_data):
    """Verify that repeated calls do not lead to significant memory creep with optimized iterations."""
    mem_manager = get_memory_manager()
    mem_manager.enable_tracemalloc = True
    if not tracemalloc.is_tracing():
        tracemalloc.start()

    # Initial usage
    initial_stats = mem_manager.get_current_usage()
    # Use tracemalloc MB if ram_used_gb is zero
    initial_val = initial_stats.tracemalloc_current_mb if initial_stats.ram_used_gb == 0 else initial_stats.ram_used_gb
    unit = "MB" if initial_stats.ram_used_gb == 0 else "GB"

    # Perform many calculations with memory management
    for _ in range(PERF_ITERATIONS_MEMORY):
        _ = compute_enhanced(large_sample_data)
        mem_manager.cleanup()
        gc.collect()  # Additional garbage collection

    final_stats = mem_manager.get_current_usage()
    final_val = final_stats.tracemalloc_current_mb if final_stats.ram_used_gb == 0 else final_stats.ram_used_gb

    print(f"\nInitial Memory: {initial_val:.4f} {unit}")
    print(f"Final Memory: {final_val:.4f} {unit}")
    print(f"Memory Increase: {final_val - initial_val:.4f} {unit}")
    print(f"Iterations: {PERF_ITERATIONS_MEMORY}")

    # Threshold: 10MB or 0.1GB
    threshold = 10.0 if unit == "MB" else 0.1
    assert (final_val - initial_val) < threshold, (
        f"Memory leak detected: {final_val - initial_val:.4f} {unit} > {threshold} {unit}"
    )


# Optimization #6: Parametrize tests to reduce code duplication
@pytest.mark.performance
@pytest.mark.parametrize(
    "version_name,compute_func",
    [
        ("Base", compute_base),
        ("Enhanced", compute_enhanced),
    ],
)
def test_individual_performance(version_name, compute_func, large_sample_data, warmed_up_cache_both):
    """Parametrized test for individual version performance benchmarking."""
    # Use optimized benchmark function (already warmed up)
    times = benchmark_function(
        lambda: compute_func(large_sample_data),
        iterations=PERF_ITERATIONS_FAST,
        warmup=False,
    )

    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)

    print(f"\n{version_name} Version Performance:")
    print(f"  Average: {avg_time:.4f}s ({avg_time * 1000:.2f} ms)")
    print(f"  Min: {min_time:.4f}s ({min_time * 1000:.2f} ms)")
    print(f"  Max: {max_time:.4f}s ({max_time * 1000:.2f} ms)")
    print(f"  Std Dev: {std_time:.4f}s ({std_time * 1000:.2f} ms)")
    print(f"  Iterations: {len(times)}")

    # Verify performance is reasonable (should complete within 60 seconds)
    assert avg_time < 60.0, f"{version_name} performance too slow: {avg_time:.4f}s"
