"""
Configuration for SPC Vote Aggregation.

This module contains the configuration class for aggregating votes from multiple
SPC strategies into a single vote, similar to Range Oscillator's combined strategy approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Literal


@dataclass
class SPCAggregationConfig:
    """Configuration for SPC vote aggregation."""
    
    # Consensus mode: "threshold", "weighted", or "simple"
    # "simple" mode is used as fallback when weighted/threshold produce no signal
    mode: Literal["threshold", "weighted", "simple"] = "weighted"
    
    # Threshold mode: minimum fraction of strategies that must agree
    threshold: float = 0.5  # 0.0-1.0
    
    # Weighted mode: minimum total weight required
    weighted_min_total: float = 0.5  # Minimum total weight for signal
    
    # Weighted mode: minimum difference between LONG and SHORT weights
    weighted_min_diff: float = 0.1  # Minimum difference to avoid ties
    
    # Adaptive weights: enable performance-based weight adjustment
    enable_adaptive_weights: bool = False
    
    # Adaptive weights: lookback window for performance calculation
    adaptive_performance_window: int = 10  # Number of recent signals to consider
    
    # Signal strength filtering: minimum strength required
    min_signal_strength: float = 0.0  # 0.0 = disabled
    
    # Strategy weights: custom weights (overrides accuracy-based weights if provided)
    strategy_weights: Optional[Dict[str, float]] = None
    
    # Simple mode: enable fallback to simple accuracy-based voting
    enable_simple_fallback: bool = True  # Use simple mode when weighted/threshold fail
    
    # Simple mode: minimum total accuracy required for signal
    simple_min_accuracy_total: float = 0.65  # Sum of accuracies from active strategies (reduced from 1.5 to accept single strategy)

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.mode not in ["threshold", "weighted", "simple"]:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'threshold', 'weighted', or 'simple'.")
            
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {self.threshold}")
            
        if self.weighted_min_total < 0:
            raise ValueError(f"weighted_min_total must be non-negative, got {self.weighted_min_total}")
            
        if self.enable_adaptive_weights and self.adaptive_performance_window <= 0:
            raise ValueError(f"adaptive_performance_window must be > 0, got {self.adaptive_performance_window}")
            
        if not (0.0 <= self.min_signal_strength <= 1.0):
            raise ValueError(f"min_signal_strength must be in [0.0, 1.0], got {self.min_signal_strength}")
            
        if self.simple_min_accuracy_total < 0:
            raise ValueError(f"simple_min_accuracy_total must be non-negative, got {self.simple_min_accuracy_total}")


__all__ = ["SPCAggregationConfig"]

