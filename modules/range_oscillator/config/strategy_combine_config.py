
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .consensus_config import ConsensusConfig
from .dynamic_selection_config import DynamicSelectionConfig
from .strategy_specific_config import StrategySpecificConfig
from .strategy_specific_config import StrategySpecificConfig

"""Main configuration for Combined Strategy."""




@dataclass
class CombinedStrategyConfig:
    """Main configuration for Range Oscillator Combined Strategy.

    This configuration class combines all settings for the combined strategy,
    including which strategies to use, consensus settings, dynamic selection,
    and strategy-specific parameters.

    Attributes:
        enabled_strategies: List of strategy IDs to enable (e.g., [2, 3, 4]).
        consensus: Consensus configuration for combining signals.
        dynamic: Dynamic selection configuration.
        params: Strategy-specific parameter configuration.
        min_signal_strength: Minimum signal strength to include (0-1).
        strategy_weights: Optional custom weights for each strategy.
        return_confidence_score: Whether to return confidence scores.
        return_strategy_stats: Whether to return strategy statistics.
        enable_debug: Whether to enable debug logging.
    """

    enabled_strategies: List[int] = field(default_factory=list)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    dynamic: DynamicSelectionConfig = field(default_factory=DynamicSelectionConfig)
    params: StrategySpecificConfig = field(default_factory=StrategySpecificConfig)

    # General
    min_signal_strength: float = 0.0
    strategy_weights: Optional[Dict[int, float]] = None

    # Outputs
    return_confidence_score: bool = False
    return_strategy_stats: bool = False
    enable_debug: bool = False

    def __post_init__(self):
        """Validate configuration values."""
        from config import VALID_STRATEGY_IDS

        if self.enabled_strategies:
            invalid = set(self.enabled_strategies) - VALID_STRATEGY_IDS
            if invalid:
                raise ValueError(f"Invalid strategy IDs: {invalid}. Valid IDs: {sorted(VALID_STRATEGY_IDS)}")
        if self.min_signal_strength < 0:
            raise ValueError(f"min_signal_strength must be >= 0, got {self.min_signal_strength}")
        if self.strategy_weights:
            for sid, weight in self.strategy_weights.items():
                if sid not in VALID_STRATEGY_IDS:
                    raise ValueError(f"Invalid strategy ID in weights: {sid}")
                if weight < 0:
                    raise ValueError(f"Weight for strategy {sid} must be >= 0, got {weight}")
