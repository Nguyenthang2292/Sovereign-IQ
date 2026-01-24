"""
Utility functions for Adaptive Trend Classification Enhanced (ATC Enhanced).

ENHANCED VERSION with:
- Cache management for MA results (NEW)
- Original utility functions preserved

This package provides utility functions used throughout the ATC Enhanced system:
- rate_of_change: Calculate percentage price change
- diflen: Calculate length offsets for Moving Averages based on robustness
- exp_growth: Calculate exponential growth factor over time
- ATCConfig: Configuration dataclass for ATC analysis
- create_atc_config_from_dict: Helper to create ATCConfig from dictionary
- CacheManager: Intelligent caching for MA results (NEW)
- get_cached_ma: Convenience function for cached MA retrieval (NEW)
"""

from modules.adaptive_trend_enhance.utils.config import ATCConfig, create_atc_config_from_dict
from modules.adaptive_trend_enhance.utils.diflen import diflen
from modules.adaptive_trend_enhance.utils.exp_growth import exp_growth
from modules.adaptive_trend_enhance.utils.rate_of_change import rate_of_change

# NEW: Cache Management
from modules.adaptive_trend_enhance.utils.cache_manager import (
    CacheManager,
    CacheEntry,
    get_cache_manager,
    reset_cache_manager,
    cached_ma,
    get_cached_ma,
)


__all__ = [
    # Original utilities
    "rate_of_change",
    "diflen",
    "exp_growth",
    "ATCConfig",
    "create_atc_config_from_dict",
    # Cache Management (NEW)
    "CacheManager",
    "CacheEntry",
    "get_cache_manager",
    "reset_cache_manager",
    "cached_ma",
    "get_cached_ma",
]
