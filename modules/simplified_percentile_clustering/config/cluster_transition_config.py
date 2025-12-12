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
from modules.simplified_percentile_clustering.utils.validation import (
    validate_strategy_config,
)


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
    bullish_transitions: list[tuple[int, int]] = None  # e.g., [(0, 1), (0, 2), (1, 2)]
    bearish_transitions: list[tuple[int, int]] = None  # e.g., [(2, 1), (2, 0), (1, 0)]

    def __post_init__(self):
        """Set default transition rules if not provided and validate config."""
        if self.bullish_transitions is None:
            # Default: transitions to higher clusters are bullish
            self.bullish_transitions = [(0, 1), (0, 2), (1, 2)]
        if self.bearish_transitions is None:
            # Default: transitions to lower clusters are bearish
            self.bearish_transitions = [(2, 1), (2, 0), (1, 0)]
        # Validate configuration
        validate_strategy_config(self)


__all__ = ["ClusterTransitionConfig"]

