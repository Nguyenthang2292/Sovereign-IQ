"""
Core Range Oscillator calculations.

This module provides the fundamental oscillator calculations.
Note: calculate_weighted_ma, calculate_atr_range, and calculate_trend_direction
have been moved to modules.common.indicators for reusability.
"""

from modules.range_oscillator.core.oscillator import calculate_range_oscillator

__all__ = [
    "calculate_range_oscillator",
]
