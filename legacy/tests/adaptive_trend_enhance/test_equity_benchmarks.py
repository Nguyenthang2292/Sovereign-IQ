"""
Tests for equity calculation caching and parallel processing.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.adaptive_trend_enhance.core.compute_atc_signals.calculate_layer2_equities import calculate_layer2_equities
from modules.adaptive_trend_enhance.core.compute_equity import equity_series
from modules.adaptive_trend_enhance.utils.cache_manager import get_cache_manager, reset_cache_manager


@pytest.fixture(autouse=True)
def clean_cache():
    """Reset the cache manager before each test."""
    reset_cache_manager()
    yield
    reset_cache_manager()


def create_test_data(size=5000):
    """Create test price, ROC and signal data."""
    # Ensure reproducibility
    np.random.seed(42)
    # Use smaller ROC to avoid hitting floor too quickly
    R = pd.Series(np.random.normal(0, 0.001, size))
    sig = pd.Series(np.random.choice([-1, 0, 1], size))
    return sig, R


def test_equity_caching_correctness():
    """Verify that cached results are identical and faster."""
    sig, R = create_test_data()
    L = 0.00002
    De = 0.0003
    start_eq = 1.0

    # First call (uncached)
    start_time = time.time()
    eq1 = equity_series(start_eq, sig, R, L=L, De=De)
    duration1 = time.time() - start_time

    # Second call (cached)
    start_time = time.time()
    eq2 = equity_series(start_eq, sig, R, L=L, De=De)
    duration2 = time.time() - start_time

    # Verify results are identical
    pd.testing.assert_series_equal(eq1, eq2)

    # Verify second call is faster (cache hit)
    # Cache hit should be near-instant compared to Numba (even if Numba is fast)
    # We use a broad threshold to avoid flakiness
    assert duration2 < duration1

    # Check cache stats
    cache = get_cache_manager()
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_equity_cache_key_sensitivity():
    """Verify that cache correctly distinguishes different inputs."""
    sig, R = create_test_data()
    L = 0.0002
    De = 0.003

    # Base call
    eq1 = equity_series(1.0, sig, R, L=L, De=De)

    # Different Lambda (significantly different)
    eq2 = equity_series(1.0, sig, R, L=0.0, De=De)
    # They should be different at least in some values
    assert not eq1.equals(eq2)

    # Different Decay
    eq3 = equity_series(1.0, sig, R, L=L, De=0.1)
    assert not eq1.equals(eq3)

    # Different Signal (even one value)
    # We change a value at index 100
    sig_diff = sig.copy()
    sig_diff.iloc[100] = -1 if sig_diff.iloc[100] >= 0 else 1
    eq4 = equity_series(1.0, sig_diff, R, L=L, De=De)
    assert not eq1.equals(eq4)

    # Check cache stats
    cache = get_cache_manager()
    stats = cache.get_stats()
    # 4 misses (one for each unique call)
    assert stats["misses"] == 4
    assert stats["hits"] == 0


def test_layer2_parallel_benchmark():
    """Benchmark parallel vs sequential Layer 2 equity processing."""
    size = 10000
    sig, R = create_test_data(size=size)
    L = 0.00002
    De = 0.0003

    # Prepare Layer 1 signals (6 types)
    ma_types = ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]
    layer1_signals = {ma: sig for ma in ma_types}  # Use same signal for simplicity
    ma_configs = [(ma, 28, 1.0) for ma in ma_types]

    # Sequential run
    reset_cache_manager()  # Ensure no cache hits
    start_time = time.time()
    res_seq = calculate_layer2_equities(layer1_signals, ma_configs, R, L, De, parallel=False)
    duration_seq = time.time() - start_time

    # Parallel run
    reset_cache_manager()  # Ensure no cache hits
    start_time = time.time()
    res_par = calculate_layer2_equities(layer1_signals, ma_configs, R, L, De, parallel=True)
    duration_par = time.time() - start_time

    # Verify results
    assert len(res_seq) == len(res_par)
    for ma in ma_types:
        pd.testing.assert_series_equal(res_seq[ma], res_par[ma])

    # Print results (not an assertion because small workloads might not show speedup)
    print(f"\nLayer 2 Benchmark (6 MAs, {size} bars):")
    print(f"Sequential: {duration_seq:.4f}s")
    print(f"Parallel: {duration_par:.4f}s")
    if duration_par < duration_seq:
        print(f"Speedup: {duration_seq / duration_par:.2f}x")
    else:
        print("Note: Parallel was not faster (overhead > benefit for small workload)")


def test_cache_eviction():
    """Verify that cache eviction works when full."""
    cache = get_cache_manager()
    # Mock settings for testing eviction
    cache.max_entries_l1 = 5
    cache.max_entries_l2 = 0  # Disable L2 for testing
    cache._l2_cache.clear()  # Clear any existing L2 entries

    sig, R = create_test_data(size=100)

    # Add 6 entries
    for i in range(6):
        equity_series(1.0 + i, sig, R, L=0.02, De=0.03)

    stats = cache.get_stats()
    assert stats["entries_l1"] <= 5
    assert stats["misses"] == 6
