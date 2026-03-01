from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from ..models.order_block import OrderBlock
from .bos import BosChochResult, identify_bos_choch
from .equal_hl import EqualHLResult, identify_equal_hl
from .order_block import identify_order_blocks
from .swing import SwingResult, detect_swings
from .trend import BULLISH, BEARISH, detect_trend


@dataclass
class SMCState:
    ohlcv: pd.DataFrame
    swings: SwingResult
    trend: int
    internal_structure: BosChochResult
    swing_structure: BosChochResult
    equal_hl: EqualHLResult
    ob_internal: List[OrderBlock]
    ob_swing: List[OrderBlock]


def _last_break_direction(structure: BosChochResult) -> Optional[int]:
    events: list[tuple[pd.Timestamp, int]] = []

    if not structure.bullish_bos.empty:
        for _, row in structure.bullish_bos.iterrows():
            events.append((pd.Timestamp(row["Crossing_Time"]), BULLISH))

    if not structure.bullish_choch.empty:
        for _, row in structure.bullish_choch.iterrows():
            events.append((pd.Timestamp(row["Crossing_Time"]), BULLISH))

    if not structure.bearish_bos.empty:
        for _, row in structure.bearish_bos.iterrows():
            events.append((pd.Timestamp(row["Crossing_Time"]), BEARISH))

    if not structure.bearish_choch.empty:
        for _, row in structure.bearish_choch.iterrows():
            events.append((pd.Timestamp(row["Crossing_Time"]), BEARISH))

    if not events:
        return None

    events.sort(key=lambda item: item[0])
    return events[-1][1]


class SMCAnalyzer:
    def __init__(self, internal_order: int = 5, external_order: int = 50):
        self.internal_order = internal_order
        self.external_order = external_order

    def run(self, df: pd.DataFrame) -> SMCState:
        df_filtered = df.copy()
        if "Date" in df_filtered.columns and not isinstance(df_filtered.index, pd.DatetimeIndex):
            df_filtered.set_index("Date", inplace=True)
        if not isinstance(df_filtered.index, pd.DatetimeIndex):
            df_filtered.index = pd.to_datetime(df_filtered.index)

        df_filtered = df_filtered[["Open", "High", "Low", "Close"]].copy()

        highs = df_filtered["High"].to_numpy()
        lows = df_filtered["Low"].to_numpy()
        closes = df_filtered["Close"].to_numpy()

        swings = detect_swings(df_filtered, self.internal_order, self.external_order)

        initial_trend = detect_trend(swings.swing_highs, swings.swing_lows)
        internal_structure = identify_bos_choch(
            df_filtered,
            swings.internal_highs,
            swings.internal_lows,
            initial_trend=initial_trend,
        )
        swing_structure = identify_bos_choch(
            df_filtered,
            swings.swing_highs,
            swings.swing_lows,
            initial_trend=initial_trend,
        )

        last_break = _last_break_direction(swing_structure)
        trend = detect_trend(swings.swing_highs, swings.swing_lows, last_structure_break=last_break)

        equal_hl = identify_equal_hl(
            df_filtered,
            highs,
            lows,
            closes,
        )

        ob_internal = identify_order_blocks(df_filtered, swings.internal_highs, swings.internal_lows, trend)
        ob_swing = identify_order_blocks(df_filtered, swings.swing_highs, swings.swing_lows, trend)

        return SMCState(
            ohlcv=df_filtered,
            swings=swings,
            trend=trend,
            internal_structure=internal_structure,
            swing_structure=swing_structure,
            equal_hl=equal_hl,
            ob_internal=ob_internal,
            ob_swing=ob_swing,
        )

    def export(self, df: pd.DataFrame) -> tuple:
        """
        Returns a tuple of 15 elements for downstream rendering/export.
        """
        state = self.run(df)
        opens = state.ohlcv["Open"].to_list()
        highs = state.ohlcv["High"].to_list()
        lows = state.ohlcv["Low"].to_list()
        closes = state.ohlcv["Close"].to_list()
        times = state.ohlcv.index.tolist()

        return (
            opens,
            highs,
            lows,
            closes,
            times,
            state.trend,
            state.swings.internal_highs,
            state.internal_structure.bullish_choch,
            state.internal_structure.bearish_choch,
            state.ob_internal,
            state.swings.swing_highs,
            state.swings.swing_lows,
            state.swing_structure.bullish_choch,
            state.swing_structure.bearish_choch,
            state.ob_swing,
        )
