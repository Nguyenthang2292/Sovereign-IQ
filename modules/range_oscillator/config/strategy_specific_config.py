"""Configuration for strategy-specific parameters."""

from dataclasses import dataclass


@dataclass
class StrategySpecificConfig:
    """Configuration for strategy-specific parameters.
    
    This class contains parameters specific to each individual strategy
    used in the combined strategy approach.
    """
    # Strategy 2: Sustained
    use_sustained: bool = True
    min_bars_sustained: int = 3
    
    # Strategy 3: Crossover
    use_crossover: bool = True
    confirmation_bars: int = 2
    
    # Strategy 4: Momentum
    use_momentum: bool = True
    momentum_period: int = 3
    momentum_threshold: float = 5.0
    
    # Strategy 6: Breakout
    use_breakout: bool = False
    breakout_upper_threshold: float = 100.0
    breakout_lower_threshold: float = -100.0
    breakout_confirmation_bars: int = 2
    breakout_use_dynamic_exhaustion: bool = False
    breakout_exhaustion_atr_multiplier: float = 1.0
    breakout_base_exhaustion_threshold: float = 150.0
    breakout_exhaustion_atr_period: int = 50
    
    # Strategy 7: Divergence
    use_divergence: bool = False
    divergence_lookback_period: int = 30
    divergence_min_swing_bars: int = 5
    
    # Strategy 8: Trend Following
    use_trend_following: bool = False
    trend_filter_period: int = 10
    trend_oscillator_threshold: float = 20.0
    
    # Strategy 9: Mean Reversion
    use_mean_reversion: bool = False
    mean_reversion_extreme_threshold: float = 80.0
    mean_reversion_zero_cross_threshold: float = 10.0
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.use_sustained and self.min_bars_sustained <= 0:
            raise ValueError(f"min_bars_sustained must be > 0, got {self.min_bars_sustained}")
        if self.use_crossover and self.confirmation_bars <= 0:
            raise ValueError(f"confirmation_bars must be > 0, got {self.confirmation_bars}")
        if self.use_momentum:
            if self.momentum_period <= 0:
                raise ValueError(f"momentum_period must be > 0, got {self.momentum_period}")
            if self.momentum_threshold < 0:
                raise ValueError(f"momentum_threshold must be >= 0, got {self.momentum_threshold}")
        if self.use_trend_following and self.trend_filter_period <= 0:
            raise ValueError(f"trend_filter_period must be > 0, got {self.trend_filter_period}")

