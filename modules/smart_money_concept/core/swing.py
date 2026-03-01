"""
Swing detection module for Smart Money Concept.
Pure stateless functions - no global state, no Plotly imports.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from modules.smart_money_concept.models import Pivot


@dataclass
class SwingResult:
    """Container for swing detection results."""

    internal_highs: List[Pivot]
    internal_lows: List[Pivot]
    swing_highs: List[Pivot]
    swing_lows: List[Pivot]


def detect_swings(df: pd.DataFrame, internal_order: int = 5, external_order: int = 50) -> SwingResult:
    """
    Detect swing highs and lows from OHLC DataFrame.

    Args:
        df: DataFrame with 'High' and 'Low' columns and datetime index
        internal_order: Order for internal swing detection (default 5)
        external_order: Order for external swing detection (default 30)

    Returns:
        SwingResult containing internal and swing pivots
    """
    df = df.copy()

    internal_highs, internal_lows = _detect_swing_pivots(df, internal_order, is_internal=True)
    swing_highs, swing_lows = _detect_swing_pivots(df, external_order, is_internal=False)

    return SwingResult(
        internal_highs=internal_highs, internal_lows=internal_lows, swing_highs=swing_highs, swing_lows=swing_lows
    )


def _detect_swing_pivots(df: pd.DataFrame, order: int, is_internal: bool) -> Tuple[List[Pivot], List[Pivot]]:
    """
    Detect swing pivots with given order.
    """
    swing_high_idx = argrelextrema(df["High"].values, np.greater_equal, order=order)[0]
    swing_low_idx = argrelextrema(df["Low"].values, np.less_equal, order=order)[0]

    df["Swing_High"] = np.nan
    df["Swing_Low"] = np.nan

    if swing_high_idx.size > 0:
        df.loc[df.index[swing_high_idx], "Swing_High"] = df["High"].iloc[swing_high_idx]
    if swing_low_idx.size > 0:
        df.loc[df.index[swing_low_idx], "Swing_Low"] = df["Low"].iloc[swing_low_idx]

    swing_H = df.dropna(subset=["Swing_High"])
    swing_L = df.dropna(subset=["Swing_Low"])

    swing_highs = [Pivot(level=row, bar_time=idx) for idx, row in zip(swing_H.index, swing_H["Swing_High"])]
    swing_lows = [Pivot(level=row, bar_time=idx) for idx, row in zip(swing_L.index, swing_L["Swing_Low"])]

    if is_internal and swing_highs and swing_lows:
        if (
            swing_highs[-1].bar_time is not None
            and swing_lows[-1].bar_time is not None
            and swing_highs[-1].bar_time > swing_lows[-1].bar_time
        ):
            swing_highs = swing_highs[:-1]
        else:
            swing_lows = swing_lows[:-1]

    return swing_highs, swing_lows


def classify_swing_types(
    swing_highs: List[Pivot], swing_lows: List[Pivot]
) -> Tuple[List[Tuple[Pivot, str]], List[Tuple[Pivot, str]]]:
    """
    Classify swing highs and lows into types (HH, HL, LH, LL).

    For swing_highs: if swing_high[n].level > swing_high[n-1].level,
        then swing_high[n] is "HH" and swing_high[n-1] is "HL".
    For swing_lows: if swing_low[n].level > swing_low[n-1].level,
        then swing_low[n] is "LH" and swing_low[n-1] is "LL".

    Args:
        swing_highs: List of swing high Pivot objects
        swing_lows: List of swing low Pivot objects

    Returns:
        Tuple of (classified_highs, classified_lows) where each is a list of (Pivot, classification)
    """
    classified_highs = [(ph, "") for ph in swing_highs]
    classified_lows = [(pl, "") for pl in swing_lows]

    for i in range(len(swing_highs) - 1, 0, -1):
        if (
            swing_highs[i].level is not None
            and swing_highs[i - 1].level is not None
            and float(swing_highs[i].level) > float(swing_highs[i - 1].level)
        ):
            classified_highs[i] = (swing_highs[i], "HH")
            classified_highs[i - 1] = (swing_highs[i - 1], "HL")

    for i in range(len(swing_lows) - 1, 0, -1):
        if (
            swing_lows[i].level is not None
            and swing_lows[i - 1].level is not None
            and float(swing_lows[i].level) > float(swing_lows[i - 1].level)
        ):
            classified_lows[i] = (swing_lows[i], "LH")
            classified_lows[i - 1] = (swing_lows[i - 1], "LL")

    return classified_highs, classified_lows
