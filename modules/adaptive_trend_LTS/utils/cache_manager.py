"""
Cache Manager for Adaptive Trend Enhanced

Intelligent caching system for Moving Average results.
Prevents redundant calculations and improves performance.

Author: Adaptive Trend Enhanced Team
"""

import hashlib
import os
import pickle
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional

from modules.common.ui.logging import log_error, log_info, log_warn

try:
    import numpy as np
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    log_warn("pandas/numpy not available")


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    key: str
    value: Any
    timestamp: float
    hits: int = 0
    size_bytes: int = 0
    ma_type: Optional[str] = None
    length: Optional[int] = None

    def score(self) -> float:
        """Calculate importance score (Hybrid LRU+LFU)"""
        # Frequency weight
        freq = self.hits
        # Recency weight (seconds since creation relative to current time)
        recency = self.timestamp
        return freq + recency


class CacheManager:
    """
    Enhanced multi-level cache manager for ATC calculations.

    Levels:
    - L1 (Memory): Small, very fast LRU for current symbol components.
    - L2 (Memory): Larger pool for frequent patterns across symbols.
    - Persistent (Disk): Pickled L2 cache for cross-session reuse.
    """

    def __init__(
        self,
        max_entries_l1: int = 128,
        max_entries_l2: int = 1024,
        max_size_mb_l2: float = 1000.0,
        ttl_seconds: Optional[float] = 3600.0,
        cache_dir: str = ".cache/atc",
    ):
        """
        Initialize Enhanced Cache Manager.

        Args:
            max_entries_l1: Max entries in L1 (very fast)
            max_entries_l2: Max entries in L2 (bulk)
            max_size_mb_l2: Max size for L2 in MB
            ttl_seconds: TTL for entries
            cache_dir: Directory for persistent cache
        """
        self.max_entries_l1 = max_entries_l1
        self.max_entries_l2 = max_entries_l2
        self.max_size_bytes_l2 = int(max_size_mb_l2 * 1024 * 1024)
        self.ttl_seconds = ttl_seconds
        self.cache_dir = cache_dir

        self._l1_cache: Dict[str, CacheEntry] = {}
        self._l2_cache: Dict[str, CacheEntry] = {}
        self._l2_size_bytes = 0
        self._hits_l1 = 0
        self._hits_l2 = 0
        self._misses = 0

        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except Exception as e:
                log_warn(f"Could not create cache directory {cache_dir}: {e}")
        else:
            # Auto-load existing cache
            self.load_from_disk()

        log_info(
            f"Enhanced Cache Manager initialized: L1={max_entries_l1}, L2={max_entries_l2}, "
            f"L2_max_size={max_size_mb_l2}MB, dir={cache_dir}"
        )

    def _generate_key(self, ma_type: str, length: int, price_data: Any, extra_params: Optional[Dict] = None) -> str:
        """Generate cache key using MD5 (fast)."""
        # Convert price data to hashable format
        if PANDAS_AVAILABLE and isinstance(price_data, pd.Series):
            data_raw = price_data.values.tobytes()
        elif isinstance(price_data, np.ndarray):
            data_raw = price_data.tobytes()
        else:
            data_raw = str(price_data).encode()

        data_hash = hashlib.md5(data_raw).hexdigest()[:16]

        # Build key components
        key_parts = [f"ma={ma_type}", f"len={length}", f"d={data_hash}"]

        if extra_params:
            for k, v in sorted(extra_params.items()):
                key_parts.append(f"{k}={v}")

        return "|".join(key_parts)

    def _generate_equity_key(self, signal_hash: str, R_hash: str, L: float, De: float, starting_equity: float) -> str:
        """
        Generate cache key for equity calculation.

        Args:
            signal_hash: Hash of signal series
            R_hash: Hash of rate of change series
            L: Lambda parameter
            De: Decay parameter
            starting_equity: Starting equity value

        Returns:
            Cache key string
        """
        key_parts = [
            "equity",
            f"signal={signal_hash}",
            f"R={R_hash}",
            f"L={L:.6f}",
            f"De={De:.6f}",
            f"start={starting_equity:.6f}",
        ]
        return "|".join(key_parts)

    def get_equity(self, signal: Any, R: Any, L: float, De: float, starting_equity: float) -> Optional[Any]:
        """Get cached equity curve (checks L1 and L2)."""
        # Generate hashes
        if PANDAS_AVAILABLE and isinstance(signal, pd.Series):
            s_raw = signal.values.tobytes()
        elif isinstance(signal, np.ndarray):
            s_raw = signal.tobytes()
        else:
            s_raw = str(signal).encode()

        signal_hash = hashlib.md5(s_raw).hexdigest()[:16]

        if PANDAS_AVAILABLE and isinstance(R, pd.Series):
            r_raw = R.values.tobytes()
        elif isinstance(R, np.ndarray):
            r_raw = R.tobytes()
        else:
            r_raw = str(R).encode()

        R_hash = hashlib.md5(r_raw).hexdigest()[:16]

        key = self._generate_equity_key(signal_hash, R_hash, L, De, starting_equity)
        return self._get_entry(key)

    def _get_entry(self, key: str) -> Optional[Any]:
        """Base get logic with multi-level promotion."""
        # Check L1
        entry = self._l1_cache.get(key)
        if entry:
            self._hits_l1 += 1
            entry.hits += 1
            return entry.value

        # Check L2
        entry = self._l2_cache.get(key)
        if entry:
            self._hits_l2 += 1
            entry.hits += 1
            # Promote to L1 (replace oldest if full)
            if len(self._l1_cache) >= self.max_entries_l1:
                oldest_key = min(self._l1_cache.keys(), key=lambda k: self._l1_cache[k].timestamp)
                self._l1_cache.pop(oldest_key)
            self._l1_cache[key] = entry
            return entry.value

        self._misses += 1
        return None

    def put_equity(self, signal: Any, R: Any, L: float, De: float, starting_equity: float, equity: Any):
        """Cache equity curve (puts in L1 and L2)."""
        # Generate hashes
        if PANDAS_AVAILABLE and isinstance(signal, pd.Series):
            s_raw = signal.values.tobytes()
        elif isinstance(signal, np.ndarray):
            s_raw = signal.tobytes()
        else:
            s_raw = str(signal).encode()
        signal_hash = hashlib.md5(s_raw).hexdigest()[:16]

        if PANDAS_AVAILABLE and isinstance(R, pd.Series):
            r_raw = R.values.tobytes()
        elif isinstance(R, np.ndarray):
            r_raw = R.tobytes()
        else:
            r_raw = str(R).encode()
        R_hash = hashlib.md5(r_raw).hexdigest()[:16]

        key = self._generate_equity_key(signal_hash, R_hash, L, De, starting_equity)
        self._put_entry(key, equity)

    def _put_entry(self, key: str, value: Any, ma_type: str = None, length: int = None):
        """Base put logic for multi-level cache."""
        size_bytes = self._estimate_size(value)
        entry = CacheEntry(
            key=key, value=value, timestamp=time.time(), hits=1, size_bytes=size_bytes, ma_type=ma_type, length=length
        )

        # L1 logic (Strict LRU if full)
        if len(self._l1_cache) >= self.max_entries_l1:
            # Pop oldest from L1
            oldest_key = min(self._l1_cache.keys(), key=lambda k: self._l1_cache[k].timestamp)
            self._l1_cache.pop(oldest_key)
        self._l1_cache[key] = entry

        # L2 logic (Hybrid LRU+LFU)
        while len(self._l2_cache) >= self.max_entries_l2 or self._l2_size_bytes + size_bytes > self.max_size_bytes_l2:
            if not self._evict_l2():
                break

        if len(self._l2_cache) < self.max_entries_l2 and self._l2_size_bytes + size_bytes <= self.max_size_bytes_l2:
            self._l2_cache[key] = entry
            self._l2_size_bytes += size_bytes

    def _estimate_size(self, value: Any) -> int:
        """
        Estimate size of cached value in bytes.

        Args:
            value: Value to estimate

        Returns:
            Estimated size in bytes
        """
        try:
            # Try pickle serialization for accurate size
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Fallback estimates
            if PANDAS_AVAILABLE and isinstance(value, pd.Series):
                return value.memory_usage(deep=True)
            elif isinstance(value, np.ndarray):
                return value.nbytes
            else:
                # Very rough estimate
                return 1000

    def get(self, ma_type: str, length: int, price_data: Any, extra_params: Optional[Dict] = None) -> Optional[Any]:
        """Get cached MA result."""
        key = self._generate_key(ma_type, length, price_data, extra_params)
        return self._get_entry(key)

    def put(self, ma_type: str, length: int, price_data: Any, value: Any, extra_params: Optional[Dict] = None):
        """Store MA result in cache."""
        key = self._generate_key(ma_type, length, price_data, extra_params)
        self._put_entry(key, value, ma_type, length)

    def _remove_entry(self, key: str):
        """Remove entry from all cache levels."""
        self._l1_cache.pop(key, None)
        entry = self._l2_cache.pop(key, None)
        if entry:
            self._l2_size_bytes -= entry.size_bytes

    def _evict_l2(self) -> bool:
        """Evict entry from L2 using Hybrid LRU+LFU."""
        if not self._l2_cache:
            return False

        # Find entry with lowest score
        evict_key = min(self._l2_cache.keys(), key=lambda k: self._l2_cache[k].score())
        self._remove_entry(evict_key)
        return True

    def save_to_disk(self, filename: str = "cache_v1.pkl"):
        """Save L2 cache to disk."""
        path = os.path.join(self.cache_dir, filename)
        log_info(f"Saving cache to {path}...")
        try:
            # We only save entries with hits > 1 to avoid bloating
            to_save = {k: v for k, v in self._l2_cache.items() if v.hits > 1}
            with open(path, "wb") as f:
                pickle.dump(to_save, f)
            log_info(f"Saved {len(to_save)} persistent entries")
        except Exception as e:
            log_error(f"Failed to save cache: {e}")

    def load_from_disk(self, filename: str = "cache_v1.pkl"):
        """Load L2 cache from disk."""
        path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(path):
            return
        log_info(f"Loading cache from {path}...")
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)

            for k, v in loaded.items():
                if k not in self._l2_cache:
                    # Reset stats for new session? Or keep?
                    v.timestamp = time.time()
                    self._l2_cache[k] = v
                    self._l2_size_bytes += v.size_bytes
            log_info(f"Loaded {len(loaded)} entries from disk")
        except Exception as e:
            log_error(f"Failed to load cache: {e}")

    def clear(self):
        """Clear all cache levels."""
        self._l1_cache.clear()
        self._l2_cache.clear()
        self._l2_size_bytes = 0
        log_info("Enhanced Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced cache statistics."""
        total_hits = self._hits_l1 + self._hits_l2
        total_entries = len(self._l1_cache) + len(self._l2_cache)
        total_requests = total_hits + self._misses
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            "entries": total_entries,
            "entries_l1": len(self._l1_cache),
            "entries_l2": len(self._l2_cache),
            "size_l2_mb": self._l2_size_bytes / (1024 * 1024),
            "hits": total_hits,
            "hits_l1": self._hits_l1,
            "hits_l2": self._hits_l2,
            "misses": self._misses,
            "hit_rate_percent": hit_rate,
        }

    def log_stats(self):
        """Log enhanced cache statistics."""
        stats = self.get_stats()
        log_info(
            f"Cache Stats: L1={stats['entries_l1']}, L2={stats['entries_l2']}, "
            f"Size={stats['size_l2_mb']:.2f}MB, Hit Rate={stats['hit_rate_percent']:.1f}% "
            f"(L1={stats['hits_l1']}, L2={stats['hits_l2']})"
        )


# Global singleton instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global CacheManager instance (singleton)"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def reset_cache_manager():
    """Reset global CacheManager (useful for testing)"""
    global _cache_manager
    if _cache_manager is not None:
        _cache_manager.clear()
    _cache_manager = None


def cached_ma(ma_type: str, extra_params: Optional[Dict] = None):
    """
    Decorator for caching MA calculations.

    Args:
        ma_type: Moving average type
        extra_params: Additional parameters for cache key

    Example:
        @cached_ma('EMA')
        def calculate_ema(price_data, length):
            # Your EMA calculation
            return result
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(price_data, length, *args, **kwargs):
            cache = get_cache_manager()

            # Try to get from cache
            cached_result = cache.get(ma_type, length, price_data, extra_params)
            if cached_result is not None:
                return cached_result

            # Calculate
            result = func(price_data, length, *args, **kwargs)

            # Store in cache
            cache.put(ma_type, length, price_data, result, extra_params)

            return result

        return wrapper

    return decorator


# Convenience function for manual caching
def get_cached_ma(
    ma_type: str, length: int, price_data: Any, calculator: Callable, extra_params: Optional[Dict] = None
) -> Any:
    """
    Get cached MA or calculate if not cached.

    Args:
        ma_type: Moving average type
        length: MA length
        price_data: Price series
        calculator: Function to calculate MA if not cached (signature: (price_data, length) -> result)
        extra_params: Additional parameters

    Returns:
        MA result (from cache or freshly calculated)

    Example:
        result = get_cached_ma(
            'EMA', 20, price_data,
            lambda data, len: ta.ema(data, len)
        )
    """
    cache = get_cache_manager()

    # Try cache
    cached = cache.get(ma_type, length, price_data, extra_params)
    if cached is not None:
        return cached

    # Calculate
    result = calculator(price_data, length)

    # Store
    cache.put(ma_type, length, price_data, result, extra_params)

    return result
