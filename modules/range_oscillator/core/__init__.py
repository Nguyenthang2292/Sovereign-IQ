"""
Core Range Oscillator calculations.

This module provides the fundamental oscillator calculations.
Note: calculate_weighted_ma, calculate_atr_range, and calculate_trend_direction
have been moved to modules.common.indicators for reusability.
"""

from modules.range_oscillator.core.heatmap import calculate_heat_colors, calculate_trend_direction
from modules.range_oscillator.core.oscillator import (
    calculate_range_oscillator,
    calculate_range_oscillator_with_heatmap,
)

__all__ = [
    "calculate_range_oscillator",
    "calculate_range_oscillator_with_heatmap",
    "calculate_heat_colors",
    "calculate_trend_direction",
]
