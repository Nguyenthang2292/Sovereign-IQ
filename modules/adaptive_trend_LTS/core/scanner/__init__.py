"""ATC Symbol Scanner.

This sub-module provides functions for scanning multiple symbols using
Adaptive Trend Classification (ATC) to find LONG/SHORT signals.

The scanner fetches data for multiple symbols, calculates ATC signals,
and filters results based on signal strength and trend direction.
"""

from __future__ import annotations

from .scan_all_symbols import scan_all_symbols

# Re-export for testing patches
from modules.adaptive_trend_enhance.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend_enhance.core.process_layer1 import trend_sign

__all__ = [
    "scan_all_symbols",
    "compute_atc_signals",
    "trend_sign",
]
