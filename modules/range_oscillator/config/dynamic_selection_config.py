"""Configuration for dynamic strategy selection."""

from dataclasses import dataclass


@dataclass
class DynamicSelectionConfig:
    """Configuration for dynamic strategy selection based on market conditions.
    
    Attributes:
        enabled: Whether dynamic selection is enabled.
        lookback: Number of bars to look back for market condition analysis.
        volatility_threshold: Threshold for determining high volatility (0-1).
        trend_threshold: Threshold for determining trending market (0-1).
    """
    enabled: bool = False
    lookback: int = 20
    volatility_threshold: float = 0.6
    trend_threshold: float = 0.5
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {self.lookback}")
        if not (0.0 <= self.volatility_threshold <= 1.0):
            raise ValueError(f"volatility_threshold must be between 0 and 1, got {self.volatility_threshold}")
        if not (0.0 <= self.trend_threshold <= 1.0):
            raise ValueError(f"trend_threshold must be between 0 and 1, got {self.trend_threshold}")

