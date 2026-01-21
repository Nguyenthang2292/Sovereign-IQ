"""Layer 1 Processing functions for Adaptive Trend Classification (ATC).

This sub-module provides functions for processing signals from multiple Moving
Averages in Layer 1 of the ATC system:
- weighted_signal: Calculate weighted average signal from multiple signals and weights
- cut_signal: Discretize continuous signal into {-1, 0, 1}
- trend_sign: Determine trend direction (+1 for bullish, -1 for bearish, 0 for neutral)
- _layer1_signal_for_ma: Calculate Layer 1 signal for a specific MA type
"""

from __future__ import annotations

from .cut_signal import cut_signal
from .layer1_signal import _layer1_signal_for_ma
from .trend_sign import trend_sign
from .weighted_signal import weighted_signal

__all__ = [
    "weighted_signal",
    "cut_signal",
    "trend_sign",
    "_layer1_signal_for_ma",
]
