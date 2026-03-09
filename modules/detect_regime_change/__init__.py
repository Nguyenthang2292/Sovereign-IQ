"""
detect_regime_change module
===========================
Regime change detection engine using PELT + HMM.

This module provides regime duration analysis for trading symbols.
It is a pure computation module with no knowledge of trading logic.

Main exports:
    - RegimeDurationAnalyzer: Main analysis engine
    - RegimeDurationResult: Result dataclass with regime metrics
    - ChangePoint, RegimeSegment: Supporting data models
"""

from modules.detect_regime_change.models import (
    ChangePoint,
    RegimeDurationResult,
    RegimeSegment,
)
from modules.detect_regime_change.regime_duration_analyzer import (
    RegimeDurationAnalyzer,
)

__all__ = [
    "RegimeDurationAnalyzer",
    "RegimeDurationResult",
    "ChangePoint",
    "RegimeSegment",
]
