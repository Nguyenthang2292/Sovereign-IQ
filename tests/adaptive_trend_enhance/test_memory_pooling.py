import os
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.common.system.memory_pool import cleanup_pools, get_array_pool, get_series_pool


class TestMemoryPool:
    def setup_method(self):
        cleanup_pools()

    def test_array_pool_acquire_release(self):
        pool = get_array_pool()
        shape = (100, 100)
        dtype = np.float64

        # First acquire - should allocate new
        arr1 = pool.acquire(shape, dtype)
        assert arr1.shape == shape
        assert arr1.dtype == dtype

        # Modify it
        arr1[0, 0] = 99.9

        # Release it
        pool.release(arr1)

        # Second acquire - should reuse (and be zeroed if acquire() used)
        arr2 = pool.acquire(shape, dtype)
        assert arr2.shape == shape
        assert arr2.dtype == dtype
        # Check if zeroed (default acquire zeroes)
        assert arr2[0, 0] == 0.0

        # Check instance equality (implementation detail: assuming direct object reuse)
        # Note: Depending on queue implementation, might not be exact same object if queue order varies
        # or concurrent access. But here strictly single threaded.
        assert arr1 is arr2

    def test_array_pool_acquire_dirty(self):
        pool = get_array_pool()
        shape = (50,)

        arr1 = pool.acquire_dirty(shape)
        arr1[0] = 123.45
        pool.release(arr1)

        arr2 = pool.acquire_dirty(shape)
        # Should contain old data (dirty)
        assert arr2[0] == 123.45

    def test_series_pool(self):
        pool = get_series_pool()
        length = 100
        index = pd.RangeIndex(length)

        s1 = pool.acquire(length, index=index, dtype=np.float64)
        assert isinstance(s1, pd.Series)
        assert len(s1) == length

        s1.iloc[0] = 55.5
        pool.release(s1)

        s2 = pool.acquire(length, index=index, dtype=np.float64)
        assert s2.iloc[0] == 0.0  # Should be clean
        # Checking reuse behavior depends on SeriesPool implementation detail.
        # But functional correctness is met.

    def test_pool_stress(self):
        pool = get_array_pool()
        shape = (10,)

        arrays = []
        # Allocate 50 arrays
        for _ in range(50):
            arrays.append(pool.acquire(shape))

        # Release all
        for arr in arrays:
            pool.release(arr)

        # Re-acquire
        reacquired = []
        for _ in range(50):
            reacquired.append(pool.acquire(shape))

        # Should not raise errors
        assert len(reacquired) == 50
