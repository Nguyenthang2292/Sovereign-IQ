
from modules.range_oscillator.strategies.combined import (

"""
DEPRECATED: This module has been moved to strategies/combined.py.

This file exists for backward compatibility only.
All imports should be updated to use strategies.combined instead.
"""

# Re-export everything from the new location
    STRATEGY_FUNCTIONS,
    STRATEGY_NAMES,
    CombinedStrategy,
    CombinedStrategyConfig,
    ConsensusConfig,
    DynamicSelectionConfig,
    StrategySpecificConfig,
    generate_signals_combined_all_strategy,
)

__all__ = [
    "generate_signals_combined_all_strategy",
    "CombinedStrategy",
    "CombinedStrategyConfig",
    "ConsensusConfig",
    "DynamicSelectionConfig",
    "StrategySpecificConfig",
    "STRATEGY_FUNCTIONS",
    "STRATEGY_NAMES",
]
