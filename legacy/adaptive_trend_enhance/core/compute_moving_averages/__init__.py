"""
Moving Average calculations for Adaptive Trend Classification Enhanced (ATC Enhanced).

This is a package version of the former `compute_moving_averages.py`.
Public API is preserved via re-exports and backward-compatible aliases.
"""

from .calculate_kama_atc import calculate_kama_atc
from .ma_calculation_enhanced import ma_calculation_enhanced
from .set_of_moving_averages_enhanced import set_of_moving_averages_enhanced

__all__ = [
    "calculate_kama_atc",
    "ma_calculation_enhanced",
    "set_of_moving_averages_enhanced",
    # Backward compatibility
    "ma_calculation",
    "set_of_moving_averages",
]

# Backward compatibility aliases
ma_calculation = ma_calculation_enhanced
set_of_moving_averages = set_of_moving_averages_enhanced
