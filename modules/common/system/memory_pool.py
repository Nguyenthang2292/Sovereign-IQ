"""
Memory Pooling System for Optimized Resource Management.

This module provides object pools for NumPy arrays and Pandas Series to reduce
allocation overhead and GC pressure in high-throughput applications.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from modules.common.ui.logging import log_error

logger = logging.getLogger(__name__)


class ArrayPool:
    """
    Thread-safe pool for reusable NumPy arrays.

    Buckets arrays by (shape, dtype) tuple.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ArrayPool, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._pools: Dict[Tuple[Tuple[int, ...], np.dtype], queue.Queue] = {}
        self._pool_lock = threading.Lock()

        # Metrics
        self.hits = 0
        self.misses = 0
        self.allocations = 0

        self._initialized = True

    def acquire(self, shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> np.ndarray:
        """
        Acquire an array from the pool or create a new one.

        Args:
            shape: Shape of the array needed.
            dtype: Data type of the array.

        Returns:
            A NumPy array of the specified shape/dtype. Content is initialized to zeros.
        """
        key = (shape, np.dtype(dtype))

        try:
            with self._pool_lock:
                if key in self._pools:
                    q = self._pools[key]
                    if not q.empty():
                        arr = q.get_nowait()
                        self.hits += 1
                        # Reset memory to zeros? OR let user handle it?
                        # Safer to zero it out or assume it's dirty.
                        # For performance, we usually leave it dirty, but let's zero it for safety
                        # comparable to np.zeros. If user wants empty, they can use acquire_dirty.
                        arr.fill(0)
                        return arr

            # Miss
            self.misses += 1
            self.allocations += 1
            return np.zeros(shape, dtype=dtype)

        except Exception as e:
            log_error(f"Error accessing ArrayPool: {e}")
            return np.zeros(shape, dtype=dtype)

    def acquire_dirty(self, shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> np.ndarray:
        """
        Acquire an array but do NOT clear it (faster, content works like np.empty).
        """
        key = (shape, np.dtype(dtype))
        try:
            with self._pool_lock:
                if key in self._pools:
                    q = self._pools[key]
                    if not q.empty():
                        arr = q.get_nowait()
                        self.hits += 1
                        return arr

            self.misses += 1
            self.allocations += 1
            return np.empty(shape, dtype=dtype)
        except Exception:
            return np.empty(shape, dtype=dtype)

    def release(self, arr: np.ndarray) -> None:
        """
        Return an array to the pool.

        Args:
            arr: NumPy array to release.
        """
        if arr is None:
            return

        # Don't pool views, only own-data arrays
        if not arr.flags.owndata:
            return

        key = (arr.shape, arr.dtype)

        with self._pool_lock:
            if key not in self._pools:
                self._pools[key] = queue.Queue(maxsize=100)  # Cap per bucket

            q = self._pools[key]
            if not q.full():
                q.put(arr)
            # else: drop, let GC handle it


class SeriesPool:
    """
    Thread-safe pool for reusable Pandas Series.

    Pandas Series wrap numpy arrays. We can reuse the Series object itself
    wrapping a pooled array, or pool Series by length.

    Actually, replacing the underlying array of a Series is possible but tricky properties.
    Easier is to create a new Series wrapping a pooled array (cheap)
    OR pool the entire Series if Index is identical.

    Strategy:
    Pool Series by (length, dtype).
    When reacquired, reset index is expensive?
    Most ATC operations use the SAME index (Time index).

    If we can cache by (length, dtype), and user sets index via `series.index = new_index`?
    Setting index is O(1) if it's just a reference assign.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SeriesPool, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Pools keyed by (length, dtype)
        self._pools: Dict[Tuple[int, np.dtype], queue.Queue] = {}
        self._pool_lock = threading.Lock()

        self.hits = 0
        self.misses = 0
        self._initialized = True

    def acquire(self, length: int, dtype: np.dtype = np.float64, index=None, name=None) -> pd.Series:
        """
        Acquire a Series of specific length.
        """
        key = (length, np.dtype(dtype))

        s = None
        with self._pool_lock:
            if key in self._pools:
                q = self._pools[key]
                if not q.empty():
                    s = q.get_nowait()
                    self.hits += 1

        if s is None:
            self.misses += 1
            # Create new
            s = pd.Series(np.zeros(length, dtype=dtype))
        else:
            # Check if underlying array needs zeroing?
            # Yes, standard behavior is clean.
            s[:] = 0

        # Set attributes
        if index is not None:
            # If lengths match, this is fast
            s.index = index
        # If lengths mismatch, pandas might complain or reindex?
        # We pooled by length, so we assume index has same length.

        if name is not None:
            s.name = name

        return s

    def release(self, s: pd.Series) -> None:
        """
        Release a Series back to pool.
        """
        if s is None:
            return

        length = len(s)
        dtype = s.dtype
        key = (length, dtype)

        # Clear references
        s.index = pd.RangeIndex(length)  # Detach from potentially large DatetimeIndex
        s.name = None

        with self._pool_lock:
            if key not in self._pools:
                self._pools[key] = queue.Queue(maxsize=50)  # Cap per bucket

            q = self._pools[key]
            if not q.full():
                q.put(s)


# Global accessors
_array_pool = ArrayPool()
_series_pool = SeriesPool()


def get_array_pool() -> ArrayPool:
    return _array_pool


def get_series_pool() -> SeriesPool:
    return _series_pool


def cleanup_pools():
    """Clear all pools to free memory."""
    # Logic to empty queues
    # ...
    pass
