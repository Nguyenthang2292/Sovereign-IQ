"""
Break of Structure (BOS) detection module for Smart Money Concept.
Pure stateless functions - no global state, no Plotly imports.
"""

from dataclasses import dataclass
from typing import List

import pandas as pd

from modules.smart_money_concept.models import Pivot


@dataclass
class BOSResult:
    """Container for BOS detection results."""

    high_bos: pd.DataFrame
    low_bos: pd.DataFrame


def identify_bos(df: pd.DataFrame, swing_highs: List[Pivot], swing_lows: List[Pivot]) -> BOSResult:
    """
    Identify Break of Structure (BOS) pivot points from swing data.

    The resulting BOS DataFrame contains:
    - Pivot_level: the level of the previous swing pivot
    - Pivot_bullishBos_Time / Pivot_bearishBos_Time: bar_time of that previous swing
    - Crossing_Time: bar_time of the breakout candle

    Args:
        df: DataFrame with OHLC data and datetime index
        swing_highs: List of swing high Pivot objects
        swing_lows: List of swing low Pivot objects

    Returns:
        BOSResult containing bullish and bearish BOS DataFrames
    """
    high_bos_records = []
    low_bos_records = []

    if swing_highs and len(swing_highs) >= 2:
        for idx in range(len(swing_highs) - 1):
            current = swing_highs[idx]
            next_swing = swing_highs[idx + 1]

            if current.bar_time is None or next_swing.bar_time is None:
                continue

            df_range = df[(df.index > current.bar_time) & (df.index <= next_swing.bar_time)]

            breakout = df_range[
                (df_range["High"] > current.level)
                | (df_range["Open"] > current.level)
                | (df_range["Close"] > current.level)
                | (df_range["Low"] > current.level)
            ]

            if not breakout.empty:
                breakout_candle_time = breakout.index.min()
                high_bos_records.append(
                    {
                        "Pivot_level": current.level,
                        "Pivot_bullishBos_Time": current.bar_time,
                        "Crossing_Time": breakout_candle_time,
                    }
                )

    if swing_lows and len(swing_lows) >= 2:
        for idx in range(len(swing_lows) - 1):
            current = swing_lows[idx]
            next_swing = swing_lows[idx + 1]

            if current.bar_time is None or next_swing.bar_time is None:
                continue

            df_range = df[(df.index > current.bar_time) & (df.index <= next_swing.bar_time)]

            breakout = df_range[
                (df_range["Low"] < current.level)
                | (df_range["Open"] < current.level)
                | (df_range["Close"] < current.level)
                | (df_range["High"] < current.level)
            ]

            if not breakout.empty:
                breakout_candle_time = breakout.index.min()
                low_bos_records.append(
                    {
                        "Pivot_level": current.level,
                        "Pivot_bearishBos_Time": current.bar_time,
                        "Crossing_Time": breakout_candle_time,
                    }
                )

    high_bos = pd.DataFrame(high_bos_records)
    low_bos = pd.DataFrame(low_bos_records)

    if not high_bos.empty:
        high_bos.columns = ["Pivot_level", "Pivot_bullishBos_Time", "Crossing_Time"]
    if not low_bos.empty:
        low_bos.columns = ["Pivot_level", "Pivot_bearishBos_Time", "Crossing_Time"]

    return BOSResult(high_bos=high_bos, low_bos=low_bos)
