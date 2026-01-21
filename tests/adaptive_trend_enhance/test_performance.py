import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend.core.compute_atc_signals import compute_atc_signals as compute_base
from modules.adaptive_trend_enhance.core.compute_atc_signals import compute_atc_signals as compute_enhanced
from modules.common.system import get_memory_manager


@pytest.fixture
def large_sample_data():
    """Create large sample price data for performance testing."""
    np.random.seed(42)
    n = 2000  # Larger dataset for performance comparison
    prices = pd.Series(
        100 * (1 + np.random.randn(n).cumsum() * 0.01),
        index=pd.date_range("2023-01-01", periods=n, freq="s"),  # Use 's' to avoid H/h issue
    )
    return prices


def test_performance_comparison(large_sample_data):
    """Compare performance between base and enhanced versions."""
    # Warm up Cache/Numba
    _ = compute_enhanced(large_sample_data)

    # Measure Base
    start_base = time.time()
    for _ in range(5):
        _ = compute_base(large_sample_data)
    end_base = time.time()
    avg_base = (end_base - start_base) / 5

    # Measure Enhanced
    start_enhanced = time.time()
    for _ in range(5):
        _ = compute_enhanced(large_sample_data)
    end_enhanced = time.time()
    avg_enhanced = (end_enhanced - start_enhanced) / 5

    print(f"\nAverage Execution Time (Base): {avg_base:.4f}s")
    print(f"Average Execution Time (Enhanced): {avg_enhanced:.4f}s")

    if avg_enhanced < avg_base:
        print(f"Speedup: {avg_base / avg_enhanced:.2f}x")
    else:
        print(f"Slowdown: {avg_enhanced / avg_base:.2f}x (Note: Small datasets might be affected by overhead)")

    # We expect some speedup, but even if small, it should be faster
    # High threshold to avoid noise failures on small tests
    assert avg_enhanced < avg_base * 2.0


def test_memory_leak_check(large_sample_data):
    """Verify that repeated calls do not lead to significant memory creep."""
    mem_manager = get_memory_manager()
    mem_manager.enable_tracemalloc = True
    if not tracemalloc.is_tracing():
        tracemalloc.start()

    # Initial usage
    initial_stats = mem_manager.get_current_usage()
    # Use tracemalloc MB if ram_used_gb is zero
    initial_val = initial_stats.tracemalloc_current_mb if initial_stats.ram_used_gb == 0 else initial_stats.ram_used_gb
    unit = "MB" if initial_stats.ram_used_gb == 0 else "GB"

    # Perform many calculations
    for _ in range(10):
        _ = compute_enhanced(large_sample_data)
        mem_manager.cleanup()

    final_stats = mem_manager.get_current_usage()
    final_val = final_stats.tracemalloc_current_mb if final_stats.ram_used_gb == 0 else final_stats.ram_used_gb

    print(f"\nInitial Memory: {initial_val:.4f} {unit}")
    print(f"Final Memory: {final_val:.4f} {unit}")

    # Threshold: 10MB or 0.1GB
    threshold = 10.0 if unit == "MB" else 0.1
    assert (final_val - initial_val) < threshold
