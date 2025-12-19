"""
ATR Targets Module.

Module tính toán các target prices dựa trên ATR (Average True Range) multiples.
"""

from modules.atr_targets.core.target_calculator import (
    calculate_atr_targets,
    ATRTargetResult,
    format_atr_target_display,
)

__all__ = [
    "calculate_atr_targets",
    "ATRTargetResult",
    "format_atr_target_display",
]

