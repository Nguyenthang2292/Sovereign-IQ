"""
Core sub-package for Smart Money Concept module.
Pure stateless business logic functions - no global state, no Plotly imports.
"""

from .trend import detect_trend, compute_atr, BULLISH, NEUTRAL, BEARISH
from .swing import detect_swings, classify_swing_types, SwingResult
from .bos import identify_bos, identify_bos_choch, BOSResult, BosChochResult
from .equal_hl import identify_equal_hl, EqualHLResult
from .order_block import identify_order_blocks_from_structure, OrderBlock

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
    "identify_equal_hl",
    "EqualHLResult",
    "identify_order_blocks_from_structure",
    "OrderBlock",
]
