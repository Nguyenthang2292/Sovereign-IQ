"""Adaptive Trend Classification (ATC) - Main computation module.

This sub-module provides the complete ATC signal computation pipeline:

Public API:
- compute_atc_signals: Main orchestration function for ATC computation
- calculate_layer2_equities: Layer 2 equity calculations

Internal modules:
- validation: Input validation utilities
- average_signal: Final Average_Signal calculation
"""

from .calculate_layer2_equities import calculate_layer2_equities
from .compute_atc_signals import compute_atc_signals

__all__ = [
    "compute_atc_signals",
    "calculate_layer2_equities",
]
