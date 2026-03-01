"""
Core sub-package for Smart Money Concept module.
Pure stateless business logic functions - no global state, no Plotly imports.
"""

from modules.smart_money_concept.core.trend import detect_trend, compute_atr, BULLISH, NEUTRAL, BEARISH
from modules.smart_money_concept.core.swing import detect_swings, classify_swing_types, SwingResult
from modules.smart_money_concept.core.bos import identify_bos, identify_bos_choch, BOSResult, BosChochResult

__all__ = [
    "detect_trend",
    "compute_atr",
    "BULLISH",
    "NEUTRAL",
    "BEARISH",
    "detect_swings",
    "classify_swing_types",
    "SwingResult",
    "identify_bos",
    "identify_bos_choch",
    "BOSResult",
    "BosChochResult",
]
