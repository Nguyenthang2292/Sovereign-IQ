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
    
    # Consensus mode: "threshold" or "weighted"
    mode: Literal["threshold", "weighted"] = "weighted"
    
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


__all__ = ["SPCAggregationConfig"]

