"""
Utilities for Range Oscillator module.

This module provides utility functions for the Range Oscillator.
Configuration classes have been moved to modules.range_oscillator.config.
"""

from modules.range_oscillator.config import (
    CombinedStrategyConfig,
    ConsensusConfig,
    DynamicSelectionConfig,
    StrategySpecificConfig,
)

from .oscillator_data import get_oscillator_data

__all__ = [
    "DynamicSelectionConfig",
    "ConsensusConfig",
    "StrategySpecificConfig",
    "CombinedStrategyConfig",
    "get_oscillator_data",
]
