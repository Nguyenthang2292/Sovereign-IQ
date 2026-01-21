"""
Cache Manager for Adaptive Trend Enhanced

Intelligent caching system for Moving Average results.
Prevents redundant calculations and improves performance.

Author: Adaptive Trend Enhanced Team
"""

import hashlib
import logging
import pickle
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional

try:
    import numpy as np
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("pandas/numpy not available")


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    key: str
    value: Any
    timestamp: float
    hits: int = 0
    size_bytes: int = 0


class CacheManager:
    """
    Intelligent cache manager for MA calculations.

    Features:
    - Hash-based caching (length + price series)
    - LRU eviction policy
    - Size-based eviction
    - Hit rate tracking
    - TTL (time-to-live) support
    - Memory-efficient storage
    """

    def __init__(self, max_entries: int = 1000, max_size_mb: float = 500.0, ttl_seconds: Optional[float] = 3600.0):
        """
        Initialize Cache Manager.

        Args:
            max_entries: Maximum number of cache entries
            max_size_mb: Maximum cache size in MB
            ttl_seconds: Time-to-live in seconds (None for no expiration)
        """
        self.max_entries = max_entries
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.ttl_seconds = ttl_seconds

        self._cache: Dict[str, CacheEntry] = {}
        self._total_size_bytes = 0
        self._hits = 0
        self._misses = 0

        logger.info(
            f"Cache Manager initialized: max_entries={max_entries}, max_size={max_size_mb}MB, ttl={ttl_seconds}s"
        )

    def _generate_key(self, ma_type: str, length: int, price_data: Any, extra_params: Optional[Dict] = None) -> str:
        """
        Generate cache key from parameters.

        Args:
            ma_type: Moving average type (e.g., 'EMA', 'WMA')
            length: MA length
            price_data: Price series (pandas Series or numpy array)
            extra_params: Additional parameters

        Returns:
            Cache key string
        """
        # Convert price data to hashable format
        if PANDAS_AVAILABLE and isinstance(price_data, pd.Series):
            # Use Series values and length
            data_hash = hashlib.sha256(price_data.values.tobytes()).hexdigest()[:16]
        elif isinstance(price_data, np.ndarray):
            data_hash = hashlib.sha256(price_data.tobytes()).hexdigest()[:16]
        else:
            # Fallback: convert to string
            data_hash = hashlib.sha256(str(price_data).encode()).hexdigest()[:16]

        # Build key components
        key_parts = [f"ma_type={ma_type}", f"length={length}", f"data={data_hash}"]

        if extra_params:
            for k, v in sorted(extra_params.items()):
                key_parts.append(f"{k}={v}")

        return "|".join(key_parts)

    def _generate_equity_key(
        self, signal_hash: str, R_hash: str, L: float, De: float, cutout: int, starting_equity: float
    ) -> str:
        """
        Generate cache key for equity calculation.

        Args:
            signal_hash: Hash of signal series
            R_hash: Hash of rate of change series
            L: Lambda parameter
            De: Decay parameter
            cutout: Cutout parameter
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
            f"cutout={cutout}",
            f"start={starting_equity:.6f}",
        ]
        return "|".join(key_parts)

    def get_equity(
        self, signal: Any, R: Any, L: float, De: float, cutout: int, starting_equity: float
    ) -> Optional[Any]:
        """
        Get cached equity curve.

        Args:
            signal: Signal series
            R: Rate of change series
            L: Lambda parameter
            De: Decay parameter
            cutout: Cutout parameter
            starting_equity: Starting equity value

        Returns:
            Cached equity series or None if not found
        """
        # Generate hashes for signal and R
        if PANDAS_AVAILABLE and isinstance(signal, pd.Series):
            signal_hash = hashlib.sha256(signal.values.tobytes()).hexdigest()[:16]
        elif isinstance(signal, np.ndarray):
            signal_hash = hashlib.sha256(signal.tobytes()).hexdigest()[:16]
        else:
            signal_hash = hashlib.sha256(str(signal).encode()).hexdigest()[:16]

        if PANDAS_AVAILABLE and isinstance(R, pd.Series):
            R_hash = hashlib.sha256(R.values.tobytes()).hexdigest()[:16]
        elif isinstance(R, np.ndarray):
            R_hash = hashlib.sha256(R.tobytes()).hexdigest()[:16]
        else:
            R_hash = hashlib.sha256(str(R).encode()).hexdigest()[:16]

        key = self._generate_equity_key(signal_hash, R_hash, L, De, cutout, starting_equity)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        # Check TTL
        if self.ttl_seconds is not None:
            age = time.time() - entry.timestamp
            if age > self.ttl_seconds:
                self._remove_entry(key)
                self._misses += 1
                return None

        # Hit
        entry.hits += 1
        self._hits += 1
        logger.debug(f"Equity cache HIT: L={L}, De={De}, cutout={cutout}")
        return entry.value

    def put_equity(self, signal: Any, R: Any, L: float, De: float, cutout: int, starting_equity: float, equity: Any):
        """
        Cache equity curve.

        Args:
            signal: Signal series
            R: Rate of change series
            L: Lambda parameter
            De: Decay parameter
            cutout: Cutout parameter
            starting_equity: Starting equity value
            equity: Equity series to cache
        """
        # Generate hashes
        if PANDAS_AVAILABLE and isinstance(signal, pd.Series):
            signal_hash = hashlib.sha256(signal.values.tobytes()).hexdigest()[:16]
        elif isinstance(signal, np.ndarray):
            signal_hash = hashlib.sha256(signal.tobytes()).hexdigest()[:16]
        else:
            signal_hash = hashlib.sha256(str(signal).encode()).hexdigest()[:16]

        if PANDAS_AVAILABLE and isinstance(R, pd.Series):
            R_hash = hashlib.sha256(R.values.tobytes()).hexdigest()[:16]
        elif isinstance(R, np.ndarray):
            R_hash = hashlib.sha256(R.tobytes()).hexdigest()[:16]
        else:
            R_hash = hashlib.sha256(str(R).encode()).hexdigest()[:16]

        key = self._generate_equity_key(signal_hash, R_hash, L, De, cutout, starting_equity)

        # Estimate size
        size_bytes = self._estimate_size(equity)

        # Check if we need to evict
        while len(self._cache) >= self.max_entries or self._total_size_bytes + size_bytes > self.max_size_bytes:
            if not self._evict_lru():
                logger.warning("Equity cache full, cannot add new entry")
                return

        # Create entry
        entry = CacheEntry(key=key, value=equity, timestamp=time.time(), hits=0, size_bytes=size_bytes)

        self._cache[key] = entry
        self._total_size_bytes += size_bytes

        logger.debug(f"Equity cache PUT: L={L}, De={De}, cutout={cutout} ({size_bytes} bytes)")

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
        """
        Get cached MA result.

        Args:
            ma_type: Moving average type
            length: MA length
            price_data: Price series
            extra_params: Additional parameters

        Returns:
            Cached value or None if not found
        """
        key = self._generate_key(ma_type, length, price_data, extra_params)

        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        # Check TTL
        if self.ttl_seconds is not None:
            age = time.time() - entry.timestamp
            if age > self.ttl_seconds:
                # Expired
                self._remove_entry(key)
                self._misses += 1
                return None

        # Hit
        entry.hits += 1
        self._hits += 1

        logger.debug(f"Cache HIT: {ma_type} length={length}")
        return entry.value

    def put(self, ma_type: str, length: int, price_data: Any, value: Any, extra_params: Optional[Dict] = None):
        """
        Store MA result in cache.

        Args:
            ma_type: Moving average type
            length: MA length
            price_data: Price series
            value: MA result to cache
            extra_params: Additional parameters
        """
        key = self._generate_key(ma_type, length, price_data, extra_params)

        # Estimate size
        size_bytes = self._estimate_size(value)

        # Check if we need to evict
        while len(self._cache) >= self.max_entries or self._total_size_bytes + size_bytes > self.max_size_bytes:
            if not self._evict_lru():
                # Can't evict anymore
                logger.warning("Cache full, cannot add new entry")
                return

        # Create entry
        entry = CacheEntry(key=key, value=value, timestamp=time.time(), hits=0, size_bytes=size_bytes)

        self._cache[key] = entry
        self._total_size_bytes += size_bytes

        logger.debug(f"Cache PUT: {ma_type} length={length} ({size_bytes} bytes)")

    def _remove_entry(self, key: str):
        """Remove entry from cache"""
        entry = self._cache.pop(key, None)
        if entry:
            self._total_size_bytes -= entry.size_bytes

    def _evict_lru(self) -> bool:
        """
        Evict least recently used entry.

        Returns:
            True if an entry was evicted
        """
        if not self._cache:
            return False

        # Find LRU entry (oldest timestamp, fewest hits)
        lru_key = min(self._cache.keys(), key=lambda k: (self._cache[k].timestamp, self._cache[k].hits))

        self._remove_entry(lru_key)
        logger.debug(f"Evicted LRU entry: {lru_key}")
        return True

    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
        self._total_size_bytes = 0
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            "entries": len(self._cache),
            "size_mb": self._total_size_bytes / (1024 * 1024),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": hit_rate,
            "max_entries": self.max_entries,
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
        }

    def log_stats(self):
        """Log cache statistics"""
        stats = self.get_stats()
        logger.info(
            f"Cache Stats: {stats['entries']} entries, "
            f"{stats['size_mb']:.2f}MB, "
            f"Hit rate: {stats['hit_rate_percent']:.1f}%"
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
