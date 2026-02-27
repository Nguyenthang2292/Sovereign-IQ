"""
Change of Character (CHOCH) detection module for Smart Money Concept.
Pure stateless functions - no global state, no Plotly imports.
"""

from dataclasses import dataclass
from typing import List

from modules.smart_money_concept.core.bos import BOSResult
from modules.smart_money_concept.models import Pivot


@dataclass
class ChochResult:
    """Container for CHOCH detection results."""

    bullish: List
    bearish: List


def identify_choch(bos: BOSResult, swing_highs: List[Pivot], swing_lows: List[Pivot]) -> ChochResult:
    """
    Identify Change of Character (CHOCH) pivot points from BOS data.

    For bullish ChoCh:
        For each adjacent pair in high_bos, check if any swing_low has bar_time
        between the two BOS times. If so, add the earlier BOS time to bullish list.

    For bearish ChoCh:
        For each adjacent pair in low_bos, check if any swing_high has bar_time
        between the two BOS times. If so, add the earlier BOS time to bearish list.

    Args:
        bos: BOSResult from identify_bos()
        swing_highs: List of swing high Pivot objects
        swing_lows: List of swing low Pivot objects

    Returns:
        ChochResult containing bullish and bearish CHOCH timestamps
    """
    bullish_choch = []
    bearish_choch = []

    if not bos.high_bos.empty and len(bos.high_bos) >= 2:
        high_bos_sorted = bos.high_bos.sort_values(by="Pivot_bullishBos_Time")
        for i in range(1, len(high_bos_sorted)):
            t_prev = high_bos_sorted.iloc[i - 1]["Pivot_bullishBos_Time"]
            t_curr = high_bos_sorted.iloc[i]["Pivot_bullishBos_Time"]
            for swing in swing_lows:
                if swing.bar_time is not None and t_prev < swing.bar_time < t_curr:
                    bullish_choch.append(t_prev)
                    break

    if not bos.low_bos.empty and len(bos.low_bos) >= 2:
        low_bos_sorted = bos.low_bos.sort_values(by="Pivot_bearishBos_Time")
        for i in range(1, len(low_bos_sorted)):
            t_prev = low_bos_sorted.iloc[i - 1]["Pivot_bearishBos_Time"]
            t_curr = low_bos_sorted.iloc[i]["Pivot_bearishBos_Time"]
            for swing in swing_highs:
                if swing.bar_time is not None and t_prev < swing.bar_time < t_curr:
                    bearish_choch.append(t_prev)
                    break

    return ChochResult(bullish=bullish_choch, bearish=bearish_choch)
