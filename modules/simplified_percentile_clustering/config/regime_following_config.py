
from dataclasses import dataclass
from typing import Optional

from __future__ import annotations
from modules.simplified_percentile_clustering.core.clustering import ClusteringConfig
from modules.simplified_percentile_clustering.utils.validation import (
from modules.simplified_percentile_clustering.utils.validation import (

"""
Configuration for Regime Following Strategy.

This strategy follows the current market regime (cluster) and generates
signals when the market is strongly in a particular regime.
"""



    validate_clustering_config,
)


@dataclass
class RegimeFollowingConfig:
    """Configuration for regime following strategy."""

    # Clustering configuration
    clustering_config: Optional[ClusteringConfig] = None

    # Signal generation parameters
    min_regime_strength: float = 0.7  # Minimum regime strength (1 - rel_pos)
    min_cluster_duration: int = 2  # Minimum bars in same cluster before signal
    require_momentum: bool = True  # Require price momentum confirmation
    momentum_period: int = 5  # Period for momentum calculation

    # Cluster preferences
    bullish_clusters: Optional[list[int]] = None  # Clusters considered bullish (e.g., [1, 2])
    bearish_clusters: Optional[list[int]] = None  # Clusters considered bearish (e.g., [0])

    # Real_clust thresholds
    bullish_real_clust_threshold: float = 0.5  # Minimum real_clust for bullish
    bearish_real_clust_threshold: float = 0.5  # Maximum real_clust for bearish

    def __post_init__(self):
        """Set default cluster preferences if not provided and validate config."""
        if self.bullish_clusters is None:
            # Set defaults based on k if clustering_config is provided
            if self.clustering_config is not None:
                k = self.clustering_config.k
                if k == 2:
                    self.bullish_clusters = [1]
                else:  # k == 3
                    self.bullish_clusters = [1, 2]
            else:
                # Default to k=3 clusters (most common case)
                self.bullish_clusters = [1, 2]
        if self.bearish_clusters is None:
            # Set defaults based on k if clustering_config is provided
            if self.clustering_config is not None:
                k = self.clustering_config.k
                # For both k=2 and k=3, cluster 0 is bearish
                self.bearish_clusters = [0]
            else:
                # Default to k=3 clusters (most common case)
                self.bearish_clusters = [0]
        # Validate configuration
        if self.clustering_config is not None:
            validate_clustering_config(self.clustering_config)
            # Validate cluster preferences are within valid range
            k = self.clustering_config.k
            max_cluster = k - 1
            for cluster_list, name in [
                (self.bullish_clusters, "bullish_clusters"),
                (self.bearish_clusters, "bearish_clusters"),
            ]:
                for cluster in cluster_list:
                    if cluster < 0 or cluster > max_cluster:
                        raise ValueError(
                            f"{name}: invalid cluster index {cluster} for k={k} (valid range: 0-{max_cluster})"
                        )
        if not (0.0 <= self.min_regime_strength <= 1.0):
            raise ValueError(f"min_regime_strength must be in [0.0, 1.0], got {self.min_regime_strength}")
        if self.min_cluster_duration < 1:
            raise ValueError(f"min_cluster_duration must be at least 1, got {self.min_cluster_duration}")
        if self.momentum_period < 1:
            raise ValueError(f"momentum_period must be at least 1, got {self.momentum_period}")
        if not (0.0 <= self.bullish_real_clust_threshold <= 1.0):
            raise ValueError(
                f"bullish_real_clust_threshold must be in [0.0, 1.0], got {self.bullish_real_clust_threshold}"
            )
        if not (0.0 <= self.bearish_real_clust_threshold <= 1.0):
            raise ValueError(
                f"bearish_real_clust_threshold must be in [0.0, 1.0], got {self.bearish_real_clust_threshold}"
            )


__all__ = ["RegimeFollowingConfig"]
