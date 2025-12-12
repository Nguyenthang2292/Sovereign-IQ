"""
Trading strategies based on Simplified Percentile Clustering.

This module provides various trading strategies that utilize cluster
assignments and transitions to generate trading signals.

Note: Strategy configurations are now in the config module.
Import configs from modules.simplified_percentile_clustering.config
"""

from modules.simplified_percentile_clustering.strategies.cluster_transition import (
    generate_signals_cluster_transition,
)
from modules.simplified_percentile_clustering.strategies.regime_following import (
    generate_signals_regime_following,
)
from modules.simplified_percentile_clustering.strategies.mean_reversion import (
    generate_signals_mean_reversion,
)

__all__ = [
    "generate_signals_cluster_transition",
    "generate_signals_regime_following",
    "generate_signals_mean_reversion",
]

