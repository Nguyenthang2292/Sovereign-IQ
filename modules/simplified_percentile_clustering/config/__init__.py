"""
Configuration classes for Simplified Percentile Clustering strategies.

This module contains all configuration dataclasses for clustering and strategies.
Each strategy configuration is in its own file for better organization.
"""

from modules.simplified_percentile_clustering.config.aggregation_config import (
    SPCAggregationConfig,
)
from modules.simplified_percentile_clustering.config.cluster_transition_config import (
    ClusterTransitionConfig,
)
from modules.simplified_percentile_clustering.config.mean_reversion_config import (
    MeanReversionConfig,
)
from modules.simplified_percentile_clustering.config.regime_following_config import (
    RegimeFollowingConfig,
)

__all__ = [
    "ClusterTransitionConfig",
    "RegimeFollowingConfig",
    "MeanReversionConfig",
    "SPCAggregationConfig",
]
