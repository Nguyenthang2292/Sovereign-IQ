"""
Configuration for Cluster Transition Strategy.

This strategy generates trading signals based on cluster transitions.
When the market transitions from one cluster to another, it may indicate
a regime change and potential trading opportunity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modules.simplified_percentile_clustering.core.clustering import ClusteringConfig
from modules.simplified_percentile_clustering.utils.validation import validate_clustering_config


@dataclass
class ClusterTransitionConfig:
    """Configuration for cluster transition strategy."""

    # Clustering configuration
    clustering_config: Optional[ClusteringConfig] = None

    # Signal generation parameters
    require_price_confirmation: bool = True  # Require price to move in same direction
    min_rel_pos_change: float = 0.1  # Minimum relative position change for signal
    use_real_clust_cross: bool = True  # Use real_clust crossing cluster boundaries
    min_signal_strength: float = 0.3  # Minimum signal strength (0.0 to 1.0)

    # Cluster transition rules
    bullish_transitions: Optional[list[tuple[int, int]]] = None  # e.g., [(0, 1), (0, 2), (1, 2)]
    bearish_transitions: Optional[list[tuple[int, int]]] = None  # e.g., [(2, 1), (2, 0), (1, 0)]

    def __post_init__(self):
        """Set default transition rules if not provided and validate config."""
        if self.bullish_transitions is None:
            # Default: transitions to higher clusters are bullish
            # Set defaults based on k if clustering_config is provided
            if self.clustering_config is not None:
                k = self.clustering_config.k
                if k == 2:
                    self.bullish_transitions = [(0, 1)]
                else:  # k == 3
                    self.bullish_transitions = [(0, 1), (0, 2), (1, 2)]
            else:
                # Default to k=3 transitions (most common case)
                self.bullish_transitions = [(0, 1), (0, 2), (1, 2)]
        if self.bearish_transitions is None:
            # Default: transitions to lower clusters are bearish
            # Set defaults based on k if clustering_config is provided
            if self.clustering_config is not None:
                k = self.clustering_config.k
                if k == 2:
                    self.bearish_transitions = [(1, 0)]
                else:  # k == 3
                    self.bearish_transitions = [(2, 1), (2, 0), (1, 0)]
            else:
                # Default to k=3 transitions (most common case)
                self.bearish_transitions = [(2, 1), (2, 0), (1, 0)]
        # Validate configuration
        if not (0.0 <= self.min_signal_strength <= 1.0):
            raise ValueError(f"min_signal_strength must be in [0.0, 1.0], got {self.min_signal_strength}")
        if not (0.0 <= self.min_rel_pos_change <= 1.0):
            raise ValueError(f"min_rel_pos_change must be in [0.0, 1.0], got {self.min_rel_pos_change}")
        if self.clustering_config is not None:
            validate_clustering_config(self.clustering_config)
            # Validate transitions are within valid cluster range
            k = self.clustering_config.k
            max_cluster = k - 1
            for transition_list, name in [
                (self.bullish_transitions, "bullish_transitions"),
                (self.bearish_transitions, "bearish_transitions"),
            ]:
                for from_cluster, to_cluster in transition_list:
                    if from_cluster < 0 or from_cluster > max_cluster:
                        raise ValueError(
                            f"{name}: invalid cluster index {from_cluster} for k={k} (valid range: 0-{max_cluster})"
                        )
                    if to_cluster < 0 or to_cluster > max_cluster:
                        raise ValueError(
                            f"{name}: invalid cluster index {to_cluster} for k={k} (valid range: 0-{max_cluster})"
                        )


__all__ = ["ClusterTransitionConfig"]
