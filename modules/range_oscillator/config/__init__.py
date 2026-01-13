"""
Configuration classes for Range Oscillator.

This module provides configuration classes for the Range Oscillator combined strategy,
including consensus settings, dynamic selection, strategy-specific parameters, and
the main combined strategy configuration.
"""

from .consensus_config import ConsensusConfig
from .dynamic_selection_config import DynamicSelectionConfig
from .strategy_combine_config import CombinedStrategyConfig
from .strategy_specific_config import StrategySpecificConfig

__all__ = [
    "DynamicSelectionConfig",
    "ConsensusConfig",
    "StrategySpecificConfig",
    "CombinedStrategyConfig",
]
