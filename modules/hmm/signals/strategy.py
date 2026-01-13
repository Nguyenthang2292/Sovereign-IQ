"""
HMM Strategy Interface Module

Defines the standard interface for all HMM strategies to enable
scalable strategy management and combination.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from modules.hmm.signals.resolution import HOLD, LONG, SHORT, Signal


@dataclass
class HMMStrategyResult:
    """
    Standard result structure for HMM strategy analysis.

    Attributes:
        signal: Trading signal (LONG=1, HOLD=0, SHORT=-1)
        probability: Signal confidence/probability (0.0 to 1.0)
        state: Internal state value (strategy-specific)
        metadata: Additional strategy-specific data
    """

    signal: Signal
    probability: float
    state: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

        # Ensure probability is in valid range
        self.probability = max(0.0, min(1.0, self.probability))


class HMMStrategy(ABC):
    """
    Abstract base class for all HMM strategies.

    All HMM strategies must implement this interface to be
    compatible with the strategy registry and combiner.
    """

    def __init__(self, name: str, weight: float = 1.0, enabled: bool = True, **kwargs):
        """
        Initialize HMM strategy.

        Args:
            name: Unique strategy identifier
            weight: Strategy weight for voting/scoring (default: 1.0)
            enabled: Whether strategy is enabled (default: True)
            **kwargs: Strategy-specific parameters
        """
        self.name = name
        self.weight = weight
        self.enabled = enabled
        self.params = kwargs

    @abstractmethod
    def analyze(self, df: pd.DataFrame, **kwargs) -> HMMStrategyResult:
        """
        Analyze market data and generate trading signal.

        Args:
            df: DataFrame containing OHLCV data
            **kwargs: Additional parameters (may override self.params)

        Returns:
            HMMStrategyResult with signal, probability, state, and metadata
        """
        pass

    def __repr__(self) -> str:
        """String representation of strategy."""
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.weight}, enabled={self.enabled})"


__all__ = ["HMMStrategy", "HMMStrategyResult", "Signal", "LONG", "HOLD", "SHORT"]
