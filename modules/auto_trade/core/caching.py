"""
Simple In-Memory Cache Module with TTL.
"""

import time
from typing import Any, Dict, Optional, Tuple


class Cache:
    """
    A simple thread-safe in-memory cache with Time-To-Live (TTL).
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache if it exists and hasn't expired.
        """
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
        expiry = time.time() + ttl
        self._cache[key] = (value, expiry)

    def delete(self, key: str) -> None:
        """
        Remove a key from the cache.
        """
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """
        Clear all items from the cache.
        """
        self._cache.clear()

    def cleanup(self) -> None:
        """
        Remove expired items from the cache.
        """
        now = time.time()
        keys_to_remove = [k for k, (_, expiry) in self._cache.items() if now > expiry]
        for k in keys_to_remove:
            del self._cache[k]
