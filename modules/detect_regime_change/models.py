"""
detect_regime_change/models.py
==============================
Data models for regime change detection results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ChangePoint:
    """A single detected change point."""
    index: int                    # Index in the data series
    timestamp: Optional[str]      # ISO timestamp (if available)


@dataclass
class RegimeSegment:
    """A single regime segment between two change points."""
    start_index: int
    end_index: int
    duration_seconds: float       # Regime duration in seconds
    duration_hours: float         # Regime duration in hours
    mean_return: Optional[float]  # Average return in this segment
    volatility: Optional[float]   # Volatility in this segment


@dataclass
class RegimeDurationResult:
    """
    Regime duration analysis result for one symbol.
    This is the main output consumed by the auto_trade module.
    """
    symbol: str
    timeframe: str                          # Analysis timeframe (e.g., "15m")

    # === PELT Results ===
    pelt_change_points: List[ChangePoint] = field(default_factory=list)
    pelt_segments: List[RegimeSegment] = field(default_factory=list)
    pelt_avg_duration_hours: Optional[float] = None
    pelt_median_duration_hours: Optional[float] = None

    # === HMM Results ===
    hmm_next_state_duration_hours: Optional[float] = None
    hmm_state: Optional[int] = None          # -1=BEARISH, 0=NEUTRAL, 1=BULLISH
    hmm_state_probability: Optional[float] = None

    # === Combined Result ===
    recommended_duration_hours: Optional[float] = None  # Final recommended value

    # === Metadata ===
    data_points_analyzed: int = 0
    analysis_timestamp: Optional[str] = None
    computation_time_ms: Optional[float] = None
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if result has a valid recommendation."""
        return self.recommended_duration_hours is not None and self.error is None
