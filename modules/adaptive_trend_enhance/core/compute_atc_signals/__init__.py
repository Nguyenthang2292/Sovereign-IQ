"""
Adaptive Trend Classification (ATC) - Main computation module.

This package exposes:
- compute_atc_signals: orchestrates the full ATC pipeline
- calculate_layer2_equities: Layer 2 equity calculations
"""

from .calculate_layer2_equities import calculate_layer2_equities
from .compute_atc_signals import compute_atc_signals

__all__ = [
    "compute_atc_signals",
    "calculate_layer2_equities",
]
