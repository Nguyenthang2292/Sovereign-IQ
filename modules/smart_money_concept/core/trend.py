"""
Trend detection module for Smart Money Concept.
Pure stateless functions - no global state, no Plotly imports.
"""

from typing import List, Optional

import numpy as np

from modules.smart_money_concept.models import Pivot

BULLISH = 1
NEUTRAL = 0
BEARISH = -1


def detect_trend(swing_highs: List[Pivot], swing_lows: List[Pivot], last_structure_break: Optional[int] = None) -> int:
    """
    Determine market trend based on swing highs and swing lows.

    Args:
        swing_highs: List of swing high Pivot objects
        swing_lows: List of swing low Pivot objects
        last_structure_break: Optional last structure break (BULLISH or BEARISH).
            If provided, used as the trend. Only falls back to HH/HL pattern
            if last_structure_break is not provided.

    Returns:
        int: BULLISH (1), BEARISH (-1), or NEUTRAL (0)
    """
    if last_structure_break is not None:
        return last_structure_break

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return NEUTRAL

    last_high = swing_highs[-1]
    prev_high = swing_highs[-2]
    last_low = swing_lows[-1]
    prev_low = swing_lows[-2]

    if (
        last_high.level is not None
        and prev_high.level is not None
        and last_low.level is not None
        and prev_low.level is not None
    ):
        if last_high.level > prev_high.level and last_low.level > prev_low.level:
            return BULLISH
        elif last_high.level < prev_high.level and last_low.level < prev_low.level:
            return BEARISH

    return NEUTRAL


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 200) -> float | None:
    """
    Compute Average True Range (ATR) indicator.

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        period: ATR period (default 200)

    Returns:
        float: ATR value or None if insufficient data
    """
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(highs)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        true_ranges.append(max(tr1, tr2, tr3))

    if len(true_ranges) < period:
        return None

    return float(np.mean(true_ranges[-period:]))
