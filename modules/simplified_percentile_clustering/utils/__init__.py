"""
Utility functions for Simplified Percentile Clustering.

Provides helper functions for common operations, validation, and optimized calculations.
"""

from modules.simplified_percentile_clustering.utils.validation import (
    validate_clustering_config,
    validate_feature_config,
    validate_strategy_config,
    validate_input_data,
)
from modules.simplified_percentile_clustering.utils.helpers import (
    safe_isna,
    safe_isfinite,
    vectorized_min_distance,
    vectorized_min_and_second_min,
    normalize_cluster_name,
    vectorized_cluster_duration,
    vectorized_extreme_duration,
    vectorized_transition_detection,
    vectorized_crossing_detection,
)

__all__ = [
    "validate_clustering_config",
    "validate_feature_config",
    "validate_strategy_config",
    "validate_input_data",
    "safe_isna",
    "safe_isfinite",
    "vectorized_min_distance",
    "vectorized_min_and_second_min",
    "normalize_cluster_name",
    "vectorized_cluster_duration",
    "vectorized_extreme_duration",
    "vectorized_transition_detection",
    "vectorized_crossing_detection",
]

