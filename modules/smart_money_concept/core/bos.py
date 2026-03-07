"""
Break of Structure (BOS) detection module for Smart Money Concept.
Pure stateless functions - no global state, no Plotly imports.
"""

from dataclasses import dataclass
from typing import List, cast

import pandas as pd

from .constants import BEARISH, BULLISH, NEUTRAL
from ..models import Pivot


@dataclass
class BosChochResult:
    """Container for BOS and CHoCH detection results."""

    bullish_bos: pd.DataFrame
    bearish_bos: pd.DataFrame
    bullish_choch: pd.DataFrame
    bearish_choch: pd.DataFrame


@dataclass
class BOSResult:
    """Container for BOS detection results."""

    high_bos: pd.DataFrame
    low_bos: pd.DataFrame


def _empty_structure_df(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def identify_bos_choch(
    df: pd.DataFrame, swing_highs: List[Pivot], swing_lows: List[Pivot], initial_trend: int = NEUTRAL
) -> BosChochResult:
    """
    Identify Break of Structure (BOS) and Change of Character (CHoCH) using close crossover only.

    BOS: Price breaks structure in the same direction as current trend
    CHoCH: Price breaks structure in the opposite direction of current trend (trend changes)

    Args:
        df: DataFrame with OHLC data and datetime index
        swing_highs: List of swing high Pivot objects
        swing_lows: List of swing low Pivot objects
        initial_trend: Initial trend state (BULLISH, BEARISH, or NEUTRAL)

    Returns:
        BosChochResult containing bullish/bearish BOS and CHoCH DataFrames
    """
    high_bos_records = []
    low_bos_records = []
    high_choch_records = []
    low_choch_records = []

    current_trend = initial_trend

    df = df.copy()
    df["Prev_Close"] = df["Close"].shift(1)
    index_tz = df.index.tz

    def normalize_index_time(value) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if index_tz is None:
            return ts.tz_localize(None) if ts.tzinfo is not None else ts
        if ts.tzinfo is None:
            return ts.tz_localize(index_tz)
        return ts.tz_convert(index_tz)

    def find_first_bullish_break(pivot_level: float, start_time: pd.Timestamp, end_time: pd.Timestamp):
        df_range = df[(df.index > start_time) & (df.index <= end_time)]
        for bar_time in df_range.index:
            close = cast(float, df_range.at[bar_time, "Close"])
            prev_close = cast(float, df_range.at[bar_time, "Prev_Close"])
            if pd.isna(prev_close):
                continue
            if close > pivot_level and prev_close <= pivot_level:
                return bar_time
        return None

    def find_first_bearish_break(pivot_level: float, start_time: pd.Timestamp, end_time: pd.Timestamp):
        df_range = df[(df.index > start_time) & (df.index <= end_time)]
        for bar_time in df_range.index:
            close = cast(float, df_range.at[bar_time, "Close"])
            prev_close = cast(float, df_range.at[bar_time, "Prev_Close"])
            if pd.isna(prev_close):
                continue
            if close < pivot_level and prev_close >= pivot_level:
                return bar_time
        return None

    if swing_highs and len(swing_highs) >= 2:
        crossed_highs = set()
        for idx in range(len(swing_highs) - 1):
            current = swing_highs[idx]
            next_swing = swing_highs[idx + 1]

            if current.bar_time is None or next_swing.bar_time is None or current.bar_time in crossed_highs:
                continue

            current_time = normalize_index_time(current.bar_time)
            next_time = normalize_index_time(next_swing.bar_time)
            breakout_candle_time = find_first_bullish_break(current.level, current_time, next_time)

            if breakout_candle_time is None:
                continue

            crossed_highs.add(current.bar_time)

            if current_trend == BEARISH:
                high_choch_records.append(
                    {
                        "Pivot_level": current.level,
                        "Pivot_bullishChoch_Time": current.bar_time,
                        "Crossing_Time": breakout_candle_time,
                        "event_type": "CHoCH",
                    }
                )
            else:
                high_bos_records.append(
                    {
                        "Pivot_level": current.level,
                        "Pivot_bullishBos_Time": current.bar_time,
                        "Crossing_Time": breakout_candle_time,
                        "event_type": "BOS",
                    }
                )
            current_trend = BULLISH

    if swing_lows and len(swing_lows) >= 2:
        crossed_lows = set()
        for idx in range(len(swing_lows) - 1):
            current = swing_lows[idx]
            next_swing = swing_lows[idx + 1]

            if current.bar_time is None or next_swing.bar_time is None or current.bar_time in crossed_lows:
                continue

            current_time = normalize_index_time(current.bar_time)
            next_time = normalize_index_time(next_swing.bar_time)
            breakout_candle_time = find_first_bearish_break(current.level, current_time, next_time)

            if breakout_candle_time is None:
                continue

            crossed_lows.add(current.bar_time)

            if current_trend == BULLISH:
                low_choch_records.append(
                    {
                        "Pivot_level": current.level,
                        "Pivot_bearishChoch_Time": current.bar_time,
                        "Crossing_Time": breakout_candle_time,
                        "event_type": "CHoCH",
                    }
                )
            else:
                low_bos_records.append(
                    {
                        "Pivot_level": current.level,
                        "Pivot_bearishBos_Time": current.bar_time,
                        "Crossing_Time": breakout_candle_time,
                        "event_type": "BOS",
                    }
                )
            current_trend = BEARISH

    high_bos = _empty_structure_df(["Pivot_level", "Pivot_bullishBos_Time", "Crossing_Time", "event_type"])
    low_bos = _empty_structure_df(["Pivot_level", "Pivot_bearishBos_Time", "Crossing_Time", "event_type"])
    high_choch = _empty_structure_df(["Pivot_level", "Pivot_bullishChoch_Time", "Crossing_Time", "event_type"])
    low_choch = _empty_structure_df(["Pivot_level", "Pivot_bearishChoch_Time", "Crossing_Time", "event_type"])

    if high_bos_records:
        high_bos = pd.DataFrame(high_bos_records)
    if low_bos_records:
        low_bos = pd.DataFrame(low_bos_records)
    if high_choch_records:
        high_choch = pd.DataFrame(high_choch_records)
    if low_choch_records:
        low_choch = pd.DataFrame(low_choch_records)

    return BosChochResult(
        bullish_bos=high_bos,
        bearish_bos=low_bos,
        bullish_choch=high_choch,
        bearish_choch=low_choch,
    )


def identify_bos(df: pd.DataFrame, swing_highs: List[Pivot], swing_lows: List[Pivot]) -> "BOSResult":
    """
    Legacy function for backwards compatibility.
    Returns BOS only (no CHoCH).
    """
    result = identify_bos_choch(df, swing_highs, swing_lows)
    return BOSResult(high_bos=result.bullish_bos, low_bos=result.bearish_bos)
