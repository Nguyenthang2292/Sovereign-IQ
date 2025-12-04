"""
Core Range Oscillator calculations.

This module provides the fundamental oscillator calculations.
"""

from modules.range_oscillator.core.oscillator import (
    calculate_weighted_ma,
    calculate_atr_range,
    calculate_trend_direction,
    calculate_range_oscillator,
)
from modules.range_oscillator.core.utils import get_oscillator_data

__all__ = [
    "calculate_weighted_ma",
    "calculate_atr_range",
    "calculate_trend_direction",
    "calculate_range_oscillator",
    "get_oscillator_data",
]

