"""Configuration for consensus signal generation."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ConsensusConfig:
    """Configuration for consensus signal generation from multiple strategies.

    Attributes:
        mode: Consensus mode - "threshold" or "weighted".
        threshold: Threshold for threshold mode (0-1).
        adaptive_weights: Whether to use adaptive weights in weighted mode.
        performance_window: Window size for performance tracking.
        weighted_min_diff: Minimum difference between Long and Short weights.
        weighted_min_total: Minimum total weight of winning side.
    """

    mode: Literal["threshold", "weighted"] = "threshold"
    threshold: float = 0.5  # For threshold mode

    # Adaptive weights (Weighted mode)
    adaptive_weights: bool = False
    performance_window: int = 10

    # Weighted voting rules
    weighted_min_diff: float = 0.1  # Difference between Long and Short weights must be > 0.1
    weighted_min_total: float = 0.5  # Total weight of the winning side must be > 0.5

    def __post_init__(self):
        """Validate configuration values."""
        if self.mode not in ("threshold", "weighted"):
            raise ValueError(f"mode must be 'threshold' or 'weighted', got {self.mode}")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be between 0 and 1, got {self.threshold}")
        if self.performance_window < 1:
            raise ValueError(f"performance_window must be >= 1, got {self.performance_window}")
        if self.weighted_min_diff < 0:
            raise ValueError(f"weighted_min_diff must be >= 0, got {self.weighted_min_diff}")
        if self.weighted_min_total < 0:
            raise ValueError(f"weighted_min_total must be >= 0, got {self.weighted_min_total}")
