
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import pandas as pd

from config.position_sizing import (

from config.position_sizing import (

"""
Cache Manager Mixin for Hybrid Signal Calculator.

This module provides cache management functionality using LRU (Least Recently Used)
strategy for signal, data, and indicator caches.
"""



    DATA_CACHE_MAX_SIZE,
    INDICATOR_CACHE_MAX_SIZE,
    SIGNAL_CACHE_MAX_SIZE,
)


class CacheManagerMixin:
    """
    Mixin class providing cache management functionality.

    Manages three types of caches:
    - Signal cache: Cached hybrid signal results
    - Data cache: Cached OHLCV data
    - Indicator cache: Cached individual indicator results
    """

    def _init_cache(self):
        """Initialize cache structures. Should be called in __init__."""
        # Cache for signal calculations (to avoid recalculating for nearby periods)
        # Key: (symbol, period_index, signal_type), Value: (signal, confidence)
        # Using OrderedDict for LRU cache implementation
        self._signal_cache: OrderedDict[Tuple[str, int, str], Tuple[int, float]] = OrderedDict()
        self._cache_max_size = SIGNAL_CACHE_MAX_SIZE

        # Cache for data fetching (to avoid redundant fetches)
        # Key: (symbol, limit, timeframe, exchange_id), Value: Tuple[DataFrame, Optional[str]] (df, exchange_id)
        # Including exchange_id in the cache key ensures correct cache hits per exchange.
        self._data_cache: OrderedDict[Tuple[str, int, str, Optional[str]], Tuple[pd.DataFrame, Optional[str]]] = (
            OrderedDict()
        )
        self._data_cache_max_size = DATA_CACHE_MAX_SIZE

        # Cache for intermediate indicator results
        # Key: (symbol, period_index, indicator_name), Value: Dict with signal and confidence
        # Using OrderedDict for LRU cache implementation
        self._indicator_cache: OrderedDict[Tuple[str, int, str], Dict] = OrderedDict()
        self._indicator_cache_max_size = INDICATOR_CACHE_MAX_SIZE

        # Cache statistics for monitoring
        self._cache_hits = 0
        self._cache_misses = 0

    def _cache_result(self, cache_key: Tuple[str, int, str], result: Tuple[int, float]):
        """Cache a result using LRU strategy, removing oldest entries if cache is full."""
        if cache_key in self._signal_cache:
            # Update existing entry
            self._signal_cache[cache_key] = result
        else:
            # Add new entry
            if len(self._signal_cache) >= self._cache_max_size:
                # Remove oldest entry (LRU - first item)
                self._signal_cache.popitem(last=False)
            self._signal_cache[cache_key] = result
        # Move to end to mark as most recently used
        self._signal_cache.move_to_end(cache_key)

    def _cache_data_result(
        self, cache_key: Tuple[str, int, str], result: pd.DataFrame, exchange_id: Optional[str] = None
    ):
        """
        Cache OHLCV data result using LRU strategy.

        If the key is new and cache is at max size, removes the oldest entry
        (LRU eviction). Then assigns the result and moves it to the end to mark
        as most recently used.

        Args:
            cache_key: Tuple of (symbol, limit, timeframe)
            result: DataFrame containing OHLCV data
            exchange_id: Optional exchange identifier where the data was fetched from
        """
        cache_value = (result, exchange_id)
        if cache_key in self._data_cache:
            # Update existing entry
            self._data_cache[cache_key] = cache_value
        else:
            # Add new entry
            if len(self._data_cache) >= self._data_cache_max_size:
                # Remove oldest entry (LRU - first item)
                self._data_cache.popitem(last=False)
            self._data_cache[cache_key] = cache_value
        # Move to end to mark as most recently used
        self._data_cache.move_to_end(cache_key)

    def _cache_indicator_result(self, cache_key: Tuple[str, int, str], result: Dict):
        """Cache an indicator result using LRU strategy, removing oldest entries if cache is full."""
        if cache_key in self._indicator_cache:
            # Update existing entry
            self._indicator_cache[cache_key] = result
        else:
            # Add new entry
            if len(self._indicator_cache) >= self._indicator_cache_max_size:
                # Remove oldest entry (LRU - first item)
                self._indicator_cache.popitem(last=False)
            self._indicator_cache[cache_key] = result
        # Move to end to mark as most recently used
        self._indicator_cache.move_to_end(cache_key)

    def clear_cache(self):
        """Clear all caches."""
        self._signal_cache.clear()
        self._data_cache.clear()
        self._indicator_cache.clear()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics for monitoring."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        return {
            "signal_cache_size": len(self._signal_cache),
            "signal_cache_max_size": self._cache_max_size,
            "data_cache_size": len(self._data_cache),
            "data_cache_max_size": self._data_cache_max_size,
            "indicator_cache_size": len(self._indicator_cache),
            "indicator_cache_max_size": self._indicator_cache_max_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": hit_rate,
        }
