from modules.targets.core.atr import (
    ATRTargetCalculator,
    ATRTargetResult,
    calculate_atr_targets,
    format_atr_target_display,
)
from modules.targets.core.base import TargetCalculator, TargetResult

"""
Targets Core Module.

Cung cấp các base classes và implementations cho các loại target khác nhau.
"""

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
