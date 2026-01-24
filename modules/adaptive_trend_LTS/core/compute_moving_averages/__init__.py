"""
Moving Average calculations for Adaptive Trend Classification Enhanced V2 (Rust Backend).

This package provides MA calculations with Rust backend for optimal performance.
Public API is preserved via re-exports and backward-compatible aliases.
"""

from .calculate_kama_atc import calculate_kama_atc
from .ma_calculation_enhanced import ma_calculation_enhanced
from .ma_calculation_rust import ma_calculation_rust
from .set_of_moving_averages_enhanced import set_of_moving_averages_enhanced
from .set_of_moving_averages_rust import set_of_moving_averages_rust

__all__ = [
    "calculate_kama_atc",
    # Enhanced (CPU/GPU) versions
    "ma_calculation_enhanced",
    "set_of_moving_averages_enhanced",
    # Rust backend versions
    "ma_calculation_rust",
    "set_of_moving_averages_rust",
    # Backward compatibility (default to Rust for v2)
    "ma_calculation",
    "set_of_moving_averages",
]

# Backward compatibility aliases - V2 defaults to Rust backend
ma_calculation = ma_calculation_rust
set_of_moving_averages = set_of_moving_averages_rust
