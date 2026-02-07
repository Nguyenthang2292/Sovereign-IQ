"""
Cache Manager for Adaptive Trend LTS-mini-version

Intelligent caching system for Moving Average results.
Prevents redundant calculations and improves performance.
"""

import hashlib
import os
import pickle
import threading
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from modules.common.ui.logging import log_error, log_info, log_warn

try:
    import numpy as np
    import pandas as pd  # type: ignore[import-untyped]

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
        use_compression: bool = False,
        compression_level: int = 5,
        compression_algorithm: str = "blosclz",
        periodic_log_interval_requests: Optional[int] = None,
        periodic_log_interval_seconds: Optional[float] = None,
    ):
        """
        Initialize Enhanced Cache Manager.

        Args:
            max_entries_l1: Max entries in L1 (very fast)
            max_entries_l2: Max entries in L2 (bulk)
            max_size_mb_l2: Max size for L2 in MB
            ttl_seconds: TTL for entries
            cache_dir: Directory for persistent cache
            use_compression: Enable blosc compression for disk cache
            compression_level: Compression level (0-9)
            compression_algorithm: Compression algorithm name
            periodic_log_interval_requests: Log stats after every N requests (None to disable)
            periodic_log_interval_seconds: Log stats every N seconds (None to use default 60s)
        """
        if periodic_log_interval_requests is not None and periodic_log_interval_requests < 0:
            raise ValueError("periodic_log_interval_requests must be non-negative")
        if periodic_log_interval_seconds is not None and periodic_log_interval_seconds < 0:
            raise ValueError("periodic_log_interval_seconds must be non-negative")

        self.max_entries_l1 = max_entries_l1
        self.max_entries_l2 = max_entries_l2
        self.max_size_bytes_l2 = int(max_size_mb_l2 * 1024 * 1024)
        self.ttl_seconds = ttl_seconds
        self.cache_dir = cache_dir
        self.use_compression = use_compression
        self.compression_level = compression_level
        self.compression_algorithm = compression_algorithm
        self.periodic_log_interval_requests = periodic_log_interval_requests

        self._l1_cache: Dict[str, CacheEntry] = {}
        self._l2_cache: Dict[str, CacheEntry] = {}
        self._l2_size_bytes = 0
        self._hits_l1 = 0
        self._hits_l2 = 0
        self._misses = 0
        self._initial_entries = 0

        # Enhanced monitoring metrics
        self._hit_timestamps: List[float] = []  # Track hit times for rate calculation
        self._miss_timestamps: List[float] = []  # Track miss times for rate calculation
        self._eviction_count = 0  # Track number of evictions
        self._promotion_count = 0  # Track L2 -> L1 promotions
        self._last_metrics_log_time = time.time()  # Last time metrics were logged
        self._metrics_log_interval = 60.0  # Log metrics every 60 seconds

        if periodic_log_interval_seconds is not None:
            self._metrics_log_interval = periodic_log_interval_seconds

        # Thread-safety: Use RLock to allow recursive locking (same thread can acquire multiple times)
        # This prevents deadlocks when cache operations call other cache operations
        self._cache_lock = threading.RLock()

        # Check if compression is available
        if use_compression:
            from modules.adaptive_trend_LTS_mini.utils.data_compression import (
                is_compression_available,
            )

            if not is_compression_available():
                log_warn(
                    "blosc compression requested but not available. "
                    "Falling back to uncompressed mode. "
                    "Install with: pip install blosc"
                )
                self.use_compression = False

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
            data_raw = np.asarray(price_data).tobytes()
            # FIX: Include index in hash to avoid collisions when same values have different indices
            index_raw = str(price_data.index.tolist()).encode()
        elif isinstance(price_data, np.ndarray):
            data_raw = price_data.tobytes()
            index_raw = b""
        else:
            data_raw = str(price_data).encode()
            index_raw = b""

        data_hash = hashlib.md5(data_raw).hexdigest()[:16]
        index_hash = hashlib.md5(index_raw).hexdigest()[:8] if index_raw else "noindex"

        # Build key components
        key_parts = [f"ma={ma_type}", f"len={length}", f"d={data_hash}", f"idx={index_hash}"]

        if extra_params:
            for k, v in sorted(extra_params.items()):
                key_parts.append(f"{k}={v}")

        return "|".join(key_parts)

    def _generate_equity_key(
        self,
        signal_hash: str,
        rate_of_change_hash: str,
        lambda_val: float,
        decay_val: float,
        starting_equity: float,
    ) -> str:
        """
        Generate cache key for equity calculation.

        Args:
            signal_hash: Hash of signal series
            rate_of_change_hash: Hash of rate of change series
            lambda_val: Lambda parameter (growth rate)
            decay_val: Decay parameter (depreciation rate)
            starting_equity: Starting equity value

        Returns:
            Cache key string
        """
        key_parts = [
            "equity",
            f"signal={signal_hash}",
            f"roc={rate_of_change_hash}",
            f"lambda={lambda_val:.6f}",
            f"decay={decay_val:.6f}",
            f"start={starting_equity:.6f}",
        ]
        return "|".join(key_parts)

    def get_equity(
        self,
        signal: Any,
        rate_of_change: Any,
        lambda_val: float,
        decay_val: float,
        starting_equity: float,
    ) -> Optional[Any]:
        """Get cached equity curve (checks L1 and L2)."""
        # Generate hashes
        if PANDAS_AVAILABLE and isinstance(signal, pd.Series):
            s_raw = np.asarray(signal).tobytes()
        elif isinstance(signal, np.ndarray):
            s_raw = signal.tobytes()
        else:
            s_raw = str(signal).encode()

        signal_hash = hashlib.md5(s_raw).hexdigest()[:16]

        if PANDAS_AVAILABLE and isinstance(rate_of_change, pd.Series):
            r_raw = np.asarray(rate_of_change).tobytes()
        elif isinstance(rate_of_change, np.ndarray):
            r_raw = rate_of_change.tobytes()
        else:
            r_raw = str(rate_of_change).encode()

        rate_of_change_hash = hashlib.md5(r_raw).hexdigest()[:16]

        key = self._generate_equity_key(signal_hash, rate_of_change_hash, lambda_val, decay_val, starting_equity)
        return self._get_entry(key)

    def _get_entry(self, key: str) -> Optional[Any]:
        """Base get logic with multi-level promotion. Thread-safe."""
        with self._cache_lock:
            current_time = time.time()

            # Check L1
            entry = self._l1_cache.get(key)
            if entry:
                self._hits_l1 += 1
                entry.hits += 1
                self._hit_timestamps.append(current_time)
                self._maybe_log_metrics()
                return entry.value

            # Check L2
            entry = self._l2_cache.get(key)
            if entry:
                self._hits_l2 += 1
                entry.hits += 1
                self._hit_timestamps.append(current_time)
                self._promotion_count += 1

                # Promote to L1 (replace oldest if full)
                if len(self._l1_cache) >= self.max_entries_l1:
                    oldest_key = min(self._l1_cache.keys(), key=lambda k: self._l1_cache[k].timestamp)
                    self._l1_cache.pop(oldest_key)
                self._l1_cache[key] = entry
                self._maybe_log_metrics()
                return entry.value

            if self._misses < 5:
                from modules.common.ui.logging import log_debug

                log_debug(f"  L2 miss - key not in L2: {key[:50]}")
                log_debug(f"  L2 has {len(self._l2_cache)} keys")

            self._misses += 1
            self._miss_timestamps.append(current_time)
            self._maybe_log_metrics()
            return None

    def put_equity(
        self,
        signal: Any,
        rate_of_change: Any,
        lambda_val: float,
        decay_val: float,
        starting_equity: float,
        equity: Any,
    ):
        """Cache equity curve (puts in L1 and L2)."""
        # Generate hashes
        if PANDAS_AVAILABLE and isinstance(signal, pd.Series):
            s_raw = np.asarray(signal).tobytes()
        elif isinstance(signal, np.ndarray):
            s_raw = signal.tobytes()
        else:
            s_raw = str(signal).encode()
        signal_hash = hashlib.md5(s_raw).hexdigest()[:16]

        if PANDAS_AVAILABLE and isinstance(rate_of_change, pd.Series):
            r_raw = np.asarray(rate_of_change).tobytes()
        elif isinstance(rate_of_change, np.ndarray):
            r_raw = rate_of_change.tobytes()
        else:
            r_raw = str(rate_of_change).encode()
        rate_of_change_hash = hashlib.md5(r_raw).hexdigest()[:16]

        key = self._generate_equity_key(signal_hash, rate_of_change_hash, lambda_val, decay_val, starting_equity)
        self._put_entry(key, equity)

    def _put_entry(self, key: str, value: Any, ma_type: Optional[str] = None, length: Optional[int] = None):
        """Base put logic for multi-level cache. Thread-safe."""
        with self._cache_lock:
            size_bytes = self._estimate_size(value)
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                hits=1,
                size_bytes=size_bytes,
                ma_type=ma_type,
                length=length,
            )

            # L1 logic (Strict LRU if full)
            if len(self._l1_cache) >= self.max_entries_l1:
                # Pop oldest from L1
                oldest_key = min(self._l1_cache.keys(), key=lambda k: self._l1_cache[k].timestamp)
                self._l1_cache.pop(oldest_key)
            self._l1_cache[key] = entry

            # L2 logic (Hybrid LRU+LFU)
            while (
                len(self._l2_cache) >= self.max_entries_l2 or self._l2_size_bytes + size_bytes > self.max_size_bytes_l2
            ):
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
        result = self._get_entry(key)
        if result is None and self._misses < 30:
            from modules.common.ui.logging import log_debug

            log_debug(f"Cache MISS #{self._misses}: ma_type={ma_type}, length={length}, key={key[:50]}...")
        return result

    def put(self, ma_type: str, length: int, price_data: Any, value: Any, extra_params: Optional[Dict] = None):
        """Store MA result in cache."""
        key = self._generate_key(ma_type, length, price_data, extra_params)
        self._put_entry(key, value, ma_type, length)

    def _remove_entry(self, key: str):
        """Remove entry from all cache levels. Must be called within lock."""
        self._l1_cache.pop(key, None)
        entry = self._l2_cache.pop(key, None)
        if entry:
            self._l2_size_bytes -= entry.size_bytes

    def _evict_l2(self) -> bool:
        """Evict entry from L2 using Hybrid LRU+LFU. Must be called within lock."""
        if not self._l2_cache:
            return False

        # Find entry with lowest score
        evict_key = min(self._l2_cache.keys(), key=lambda k: self._l2_cache[k].score())
        self._remove_entry(evict_key)
        self._eviction_count += 1
        return True

    def save_to_disk(self, filename: str = "cache_v1.pkl"):
        """Save L2 cache to disk. Thread-safe."""
        path = os.path.join(self.cache_dir, filename)
        log_info(f"Saving cache to {path}...")
        try:
            # We only save entries with hits > 1 to avoid bloating
            with self._cache_lock:
                to_save = {k: v for k, v in self._l2_cache.items() if v.hits > 1}

            if self.use_compression:
                # Save compressed
                from modules.adaptive_trend_LTS_mini.utils.data_compression import (
                    compress_to_file,
                )

                compressed_filename = filename.replace(".pkl", ".pkl.blosc")
                compressed_path = os.path.join(self.cache_dir, compressed_filename)

                compress_to_file(
                    to_save,
                    compressed_path,
                    compression_level=self.compression_level,
                    algorithm=self.compression_algorithm,
                )

                log_info(f"Saved {len(to_save)} persistent entries (compressed)")
            else:
                # Save uncompressed (legacy)
                with open(path, "wb") as f:
                    pickle.dump(to_save, f)
                log_info(f"Saved {len(to_save)} persistent entries")
        except Exception as e:
            log_error(f"Failed to save cache: {e}")

    def load_from_disk(self, filename: str = "cache_v1.pkl"):
        """Load L2 cache from disk."""
        path = os.path.join(self.cache_dir, filename)
        compressed_path = path.replace(".pkl", ".pkl.blosc")

        if not os.path.exists(path) and not os.path.exists(compressed_path):
            return

        try:
            if self.use_compression and os.path.exists(compressed_path):
                # Try loading compressed first
                from modules.adaptive_trend_LTS_mini.utils.data_compression import (
                    decompress_from_file,
                )

                log_info(f"Loading cache from {compressed_path}...")
                loaded = decompress_from_file(compressed_path)
            elif os.path.exists(path):
                # Fall back to uncompressed
                log_info(f"Loading cache from {path}...")
                with open(path, "rb") as f:
                    loaded = pickle.load(f)
            else:
                # Compression enabled but compressed file doesn't exist
                log_info(f"Compressed cache not found, checking for uncompressed: {path}")
                if os.path.exists(path):
                    log_info(f"Loading cache from {path}...")
                    with open(path, "rb") as f:
                        loaded = pickle.load(f)
                else:
                    return

            for k, v in loaded.items():
                if k not in self._l2_cache:
                    # Reset stats for new session? Or keep?
                    v.timestamp = time.time()
                    self._l2_cache[k] = v
                    self._l2_size_bytes += v.size_bytes
            log_info(f"Loaded {len(loaded)} entries from disk")
        except Exception as e:
            log_error(f"Failed to load cache: {e}")

        self._initial_entries = len(self._l2_cache)

    def warm_cache(self, symbols_data: Dict[str, Any], configs: Optional[List[Dict[str, Any]]] = None):
        """
        Warm cache by pre-calculating signals for symbols and configs.

        Args:
            symbols_data: Dictionary mapping symbol names to price Series
            configs: List of configuration dictionaries for compute_atc_signals
        """
        if not configs:
            configs = [{}]  # Default config

        total_tasks = len(symbols_data) * len(configs)
        log_info(
            f"Warming cache with {len(symbols_data)} symbols and {len(configs)} configs ({total_tasks} total tasks)..."
        )

        start_time = time.time()
        # Save current stats
        initial_stats = self.get_stats()

        from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.compute_atc_signals import compute_atc_signals

        count = 0
        for symbol, prices in symbols_data.items():
            for config in configs:
                try:
                    # Run with use_cache=True (default) to populate cache
                    # We don't store the returned results
                    compute_atc_signals(prices, **config)
                    count += 1
                    if count % 10 == 0:
                        log_info(f"Warmed {count}/{total_tasks} tasks...")
                except Exception as e:
                    log_warn(f"Failed to warm cache for {symbol} with config {config}: {e}")

        # Update stats
        duration = time.time() - start_time
        final_stats = self.get_stats()
        entries_added = final_stats["entries"] - initial_stats["entries"]

        log_info(f"Cache warming complete in {duration:.2f}s. Added {entries_added} new entries.")
        self.save_to_disk()

    def log_cache_effectiveness(self):
        """Log detailed cache effectiveness after a workflow."""
        stats = self.get_stats()
        log_info("\n" + "=" * 40)
        log_info("=== Cache Effectiveness Report ===")
        log_info(f"Entries at Start: {self._initial_entries}")
        log_info(f"Current Entries: {stats['entries']} (L1: {stats['entries_l1']}, L2: {stats['entries_l2']})")
        log_info(f"Total Requests: {stats['hits'] + stats['misses']}")
        log_info(f"Total Hits: {stats['hits']} (L1: {stats['hits_l1']}, L2: {stats['hits_l2']})")
        log_info(f"Total Misses: {stats['misses']}")
        log_info(f"Overall Hit Rate: {stats['hit_rate_percent']:.2f}%")
        log_info("=" * 40 + "\n")

    def clear(self):
        """Clear all cache levels and reset metrics."""
        with self._cache_lock:
            self._l1_cache.clear()
            self._l2_cache.clear()
            self._l2_size_bytes = 0
            # Keep cumulative metrics but clear recent tracking
            self._hit_timestamps.clear()
            self._miss_timestamps.clear()
            log_info("Enhanced Cache cleared")

    def _maybe_log_metrics(self):
        """Automatically log metrics at intervals if enabled. Must be called within lock."""
        should_log = False
        current_time = time.time()

        # Time-based check
        if self._metrics_log_interval > 0 and current_time - self._last_metrics_log_time >= self._metrics_log_interval:
            should_log = True

        # Request-based check
        if not should_log and self.periodic_log_interval_requests:
            total_requests = self._hits_l1 + self._hits_l2 + self._misses
            if total_requests > 0 and total_requests % self.periodic_log_interval_requests == 0:
                should_log = True

        if should_log:
            self._last_metrics_log_time = current_time
            try:
                # Log outside of this method to avoid recursive locks
                stats = self._get_stats_internal()
                from modules.common.ui.logging import log_info

                log_info(
                    f"Cache Metrics: Hit Rate={stats['hit_rate_percent']:.1f}% "
                    f"(L1={stats['hit_rate_l1_percent']:.1f}%, L2={stats['hit_rate_l2_percent']:.1f}%), "
                    f"Requests={stats['total_requests']}, "
                    f"Entries={stats['entries']} (L1={stats['entries_l1']}, L2={stats['entries_l2']}), "
                    f"Evictions={stats['evictions']}, Promotions={stats['promotions']}"
                )
            except Exception:
                # Catch logging exceptions so cache path is unaffected
                pass

    def _get_stats_internal(self) -> Dict[str, Any]:
        """Internal method to get stats without acquiring lock (called from within locked methods)."""
        total_hits = self._hits_l1 + self._hits_l2
        total_entries = len(self._l1_cache) + len(self._l2_cache)
        total_requests = total_hits + self._misses
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0
        hit_rate_l1 = (self._hits_l1 / total_requests * 100) if total_requests > 0 else 0.0
        hit_rate_l2 = (self._hits_l2 / total_requests * 100) if total_requests > 0 else 0.0
        miss_rate = (self._misses / total_requests * 100) if total_requests > 0 else 0.0

        # Calculate recent hit/miss rates (last minute)
        current_time = time.time()
        recent_window = 60.0  # 1 minute
        recent_hits = sum(1 for t in self._hit_timestamps if current_time - t <= recent_window)
        recent_misses = sum(1 for t in self._miss_timestamps if current_time - t <= recent_window)
        recent_total = recent_hits + recent_misses
        recent_hit_rate = (recent_hits / recent_total * 100) if recent_total > 0 else 0.0

        return {
            "entries": total_entries,
            "entries_l1": len(self._l1_cache),
            "entries_l2": len(self._l2_cache),
            "size_l2_mb": self._l2_size_bytes / (1024 * 1024),
            "hits": total_hits,
            "hits_l1": self._hits_l1,
            "hits_l2": self._hits_l2,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate,
            "hit_rate_l1_percent": hit_rate_l1,
            "hit_rate_l2_percent": hit_rate_l2,
            "miss_rate_percent": miss_rate,
            "recent_hit_rate_percent": recent_hit_rate,
            "recent_hits": recent_hits,
            "recent_misses": recent_misses,
            "evictions": self._eviction_count,
            "promotions": self._promotion_count,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced cache statistics with detailed metrics."""
        with self._cache_lock:
            return self._get_stats_internal()

    def log_stats(self):
        """Log enhanced cache statistics with detailed metrics."""
        stats = self.get_stats()
        log_info(
            f"Cache Stats: L1={stats['entries_l1']}, L2={stats['entries_l2']}, "
            f"Size={stats['size_l2_mb']:.2f}MB, Hit Rate={stats['hit_rate_percent']:.1f}% "
            f"(L1={stats['hit_rate_l1_percent']:.1f}%, L2={stats['hit_rate_l2_percent']:.1f}%), "
            f"Recent Hit Rate={stats['recent_hit_rate_percent']:.1f}%, "
            f"Evictions={stats['evictions']}, Promotions={stats['promotions']}"
        )

    def set_metrics_log_interval(self, interval_seconds: float):
        """
        Set the interval for automatic metrics logging.

        Args:
            interval_seconds: Interval in seconds (0 to disable automatic logging)
        """
        with self._cache_lock:
            self._metrics_log_interval = interval_seconds
            log_info(f"Cache metrics log interval set to {interval_seconds}s")

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """
        Get detailed cache metrics including performance analytics.

        Returns:
            Dictionary with comprehensive cache metrics and analytics
        """
        stats = self.get_stats()

        # Add performance insights
        insights = []
        if stats["total_requests"] > 0:
            if stats["hit_rate_percent"] < 50:
                insights.append("LOW_HIT_RATE: Consider warming cache or increasing cache size")
            if stats["hit_rate_l1_percent"] < 20 and stats["hit_rate_l2_percent"] > 30:
                insights.append("L2_HEAVY: Consider increasing L1 size for better performance")
            if stats["evictions"] > stats["entries"]:
                insights.append("HIGH_EVICTION: Cache thrashing detected, consider increasing cache size")
            if stats["recent_hit_rate_percent"] < stats["hit_rate_percent"] - 10:
                insights.append("DEGRADING_PERFORMANCE: Recent hit rate is declining")

        stats["insights"] = insights
        return stats


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

    # FIX: Don't cache None results to avoid freezing error results
    if result is not None:
        cache.put(ma_type, length, price_data, result, extra_params)

    return result
