from __future__ import annotations

"""
Simple In-Memory Cache Module with TTL.
"""

import time
from threading import RLock
from typing import Any, Dict, Optional, Tuple


class Cache:
    """
    A simple thread-safe in-memory cache with Time-To-Live (TTL).

    Security Note:
    - Data Storage: Data is stored in-memory in plain text. Do not store sensitive information (keys, passwords) without encryption.
    - Capacity: This simple cache does not enforce a maximum size. Susceptible to cache filling attacks if keys are user-controlled.
      For unbounded inputs, use a cache with eviction policies (LRU) like configured in ATCScanner.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = RLock()

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache if it exists and hasn't expired.
        """
        with self._lock:
            if key not in self._cache:
                return None

            value, expiry = self._cache[key]
            if time.time() > expiry:
                del self._cache[key]
                return None

            return value

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """
        Set value in cache with a TTL (default 300 seconds / 5 minutes).
        """
        with self._lock:
            expiry = time.time() + ttl
            self._cache[key] = (value, expiry)

    def delete(self, key: str) -> None:
        """
        Remove a key from the cache.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self) -> None:
        """
        Clear all items from the cache.
        """
        with self._lock:
            self._cache.clear()

    def cleanup(self) -> None:
        """
        Remove expired items from the cache.
        """
        with self._lock:
            now = time.time()
            keys_to_remove = [k for k, (_, expiry) in self._cache.items() if now > expiry]
            for k in keys_to_remove:
                del self._cache[k]

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the cache (and not expired)."""
        return self.get(key) is not None
