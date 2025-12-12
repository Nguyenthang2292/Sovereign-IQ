"""
Configuration for Mean Reversion Strategy.

This strategy generates signals when the market is at cluster extremes
and expects mean reversion back to the center cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modules.simplified_percentile_clustering.core.clustering import ClusteringConfig
from modules.simplified_percentile_clustering.utils.validation import (
    validate_strategy_config,
)


@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion strategy."""

    # Clustering configuration
    clustering_config: Optional[ClusteringConfig] = None

    # Signal generation parameters
    extreme_threshold: float = 0.2  # Real_clust threshold for extreme (0.0-1.0)
    min_extreme_duration: int = 3  # Minimum bars in extreme before signal
    require_reversal_signal: bool = True  # Require price reversal confirmation
    reversal_lookback: int = 3  # Bars to look back for reversal

    # Reversion targets
    bullish_reversion_target: float = 0.5  # Target real_clust for bullish reversion
    bearish_reversion_target: float = 0.5  # Target real_clust for bearish reversion

    # Signal strength parameters
    min_signal_strength: float = 0.4  # Minimum signal strength

    def __post_init__(self):
        """Set default reversion targets based on k and validate config."""
        if self.clustering_config:
            k = self.clustering_config.k
            if k == 3:
                self.bullish_reversion_target = 1.0  # Target middle cluster
                self.bearish_reversion_target = 1.0
            else:
                self.bullish_reversion_target = 0.5  # Target middle
                self.bearish_reversion_target = 0.5
        # Validate configuration
        validate_strategy_config(self)


__all__ = ["MeanReversionConfig"]

