"""
Utility functions for Adaptive Trend Classification Enhanced (ATC Enhanced).

ENHANCED VERSION with:
- Cache management for MA results (NEW)
- Memory-mapped data utilities for large datasets (NEW)
- Original utility functions preserved

This package provides utility functions used throughout the ATC Enhanced system:
- rate_of_change: Calculate percentage price change
- diflen: Calculate length offsets for Moving Averages based on robustness
- exp_growth: Calculate exponential growth factor over time
- ATCConfig: Configuration dataclass for ATC analysis
- create_atc_config_from_dict: Helper to create ATCConfig from dictionary
- CacheManager: Intelligent caching for MA results (NEW)
- get_cached_ma: Convenience function for cached MA retrieval (NEW)
- MemoryMappedDataManager: Memory-mapped file manager for large datasets (NEW)
- create_memory_mapped_from_csv: Create memory-mapped files from CSV (NEW)
"""

from modules.adaptive_trend_LTS.utils.config import ATCConfig, create_atc_config_from_dict
from modules.adaptive_trend_LTS.utils.diflen import diflen
from modules.adaptive_trend_LTS.utils.exp_growth import exp_growth
from modules.adaptive_trend_LTS.utils.rate_of_change import rate_of_change

# NEW: Cache Management
from modules.adaptive_trend_LTS.utils.cache_manager import (
    CacheManager,
    CacheEntry,
    get_cache_manager,
    reset_cache_manager,
    cached_ma,
    get_cached_ma,
)

# NEW: Memory-mapped data management
from modules.adaptive_trend_LTS.utils.memory_mapped_data import (
    MemmapDescriptor,
    MemoryMappedDataManager,
    create_memory_mapped_from_csv,
    load_memory_mapped_from_csv,
    get_manager as get_memory_mapped_manager,
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
    # Memory-mapped data management (NEW)
    "MemmapDescriptor",
    "MemoryMappedDataManager",
    "create_memory_mapped_from_csv",
    "load_memory_mapped_from_csv",
    "get_memory_mapped_manager",
]
