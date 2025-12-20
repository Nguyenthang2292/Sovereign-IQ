"""
Targets Module.

Module tính toán các target prices với nhiều phương pháp khác nhau:
- ATR (Average True Range) multiples
- (Có thể mở rộng thêm: Fibonacci, Support/Resistance, v.v.)
"""

from modules.targets.core import (
    # Base classes
    TargetResult,
    TargetCalculator,
    # ATR implementation
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

