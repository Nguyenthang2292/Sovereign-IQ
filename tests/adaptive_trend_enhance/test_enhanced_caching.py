import os
import shutil
import time

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_enhance.utils.cache_manager import CacheEntry, CacheManager


def test_l1_l2_promotion():
    """Test that L2 hits promote entries to L1."""
    # Small L1 to trigger promotion logic easily
    cm = CacheManager(max_entries_l1=2, max_entries_l2=10, cache_dir=".test_cache")

    data = pd.Series([1, 2, 3])

    # Put in L2 (and L1)
    cm.put("EMA", 10, data, pd.Series([1, 1, 1]))
    key = cm._generate_key("EMA", 10, data)

    # Initially in both
    assert key in cm._l1_cache
    assert key in cm._l2_cache

    # Fill L1 with other stuff to evict first key from L1
    cm.put("EMA", 20, data, pd.Series([2, 2, 2]))
    cm.put("EMA", 30, data, pd.Series([3, 3, 3]))

    assert key not in cm._l1_cache  # Evicted from L1 (FIFO/Timestamp based)
    assert key in cm._l2_cache  # Still in L2

    # Get it again (L2 hit)
    res = cm.get("EMA", 10, data)
    assert res is not None
    assert cm._hits_l2 == 1

    # Should be promoted back to L1
    assert key in cm._l1_cache

    if os.path.exists(".test_cache"):
        shutil.rmtree(".test_cache")


def test_eviction_score():
    """Test hybrid LRU+LFU eviction."""
    cm = CacheManager(max_entries_l1=10, max_entries_l2=2, cache_dir=".test_cache")
    data = pd.Series([1, 2, 3])

    # Add two entries
    cm.put("EMA", 10, data, pd.Series([10]))
    cm.put("EMA", 20, data, pd.Series([20]))

    key10 = cm._generate_key("EMA", 10, data)
    key20 = cm._generate_key("EMA", 20, data)

    # Increase hits for key10
    cm.get("EMA", 10, data)  # hits=2
    cm.get("EMA", 10, data)  # hits=3

    # Add third entry, should evict key20 (lower hits and older than 10 if we use score)
    # Actually key10 has higher score due to more hits.
    cm.put("EMA", 30, data, pd.Series([30]))

    assert key10 in cm._l2_cache
    assert key20 not in cm._l2_cache

    if os.path.exists(".test_cache"):
        shutil.rmtree(".test_cache")


def test_persistence():
    """Test saving and loading from disk."""
    cache_dir = ".test_cache_persist"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    cm = CacheManager(cache_dir=cache_dir)
    data = pd.Series([1, 2, 3])

    # Persist only items with hits > 1
    cm.put("EMA", 10, data, pd.Series([1, 2, 3]))
    cm.get("EMA", 10, data)  # hit count = 2

    cm.save_to_disk("test.pkl")
    assert os.path.exists(os.path.join(cache_dir, "test.pkl"))

    # New manager, same dir
    cm2 = CacheManager(cache_dir=cache_dir)
    # Auto-load doesn't happen for "test.pkl", only default "cache_v1.pkl"
    # So we load manually
    cm2.load_from_disk("test.pkl")

    res = cm2.get("EMA", 10, data)
    assert res is not None
    assert isinstance(res, pd.Series)
    assert res.iloc[1] == 2

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


if __name__ == "__main__":
    pytest.main([__file__])
