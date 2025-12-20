"""
Targets Core Module.

Cung cấp các base classes và implementations cho các loại target khác nhau.
"""

from modules.targets.core.base import TargetResult, TargetCalculator
from modules.targets.core.atr import (
    ATRTargetResult,
    ATRTargetCalculator,
    calculate_atr_targets,
    format_atr_target_display,
)

__all__ = [
    # Base classes
    "TargetResult",
    "TargetCalculator",
    # ATR implementation
    "ATRTargetResult",
    "ATRTargetCalculator",
    "calculate_atr_targets",
    "format_atr_target_display",
]

