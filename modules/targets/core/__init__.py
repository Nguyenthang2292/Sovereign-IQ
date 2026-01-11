
from modules.targets.core.atr import (

"""
Targets Core Module.

Cung cấp các base classes và implementations cho các loại target khác nhau.
"""

    ATRTargetCalculator,
    ATRTargetResult,
    calculate_atr_targets,
    format_atr_target_display,
)
from modules.targets.core.base import TargetCalculator, TargetResult

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
