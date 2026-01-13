"""Candlestick pattern indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.common.utils import validate_ohlcv_input

from .base import IndicatorResult, collect_metadata


class CandlestickPatterns:
    """Classic candlestick formations."""

    CATEGORY = "candlestick"

    @staticmethod
    def apply(df: pd.DataFrame) -> IndicatorResult:
        # Validate input
        validate_ohlcv_input(df, required_columns=["open", "high", "low", "close"])

        result = df.copy()
        before = result.columns.tolist()

        o = result["open"]
        h = result["high"]
        low = result["low"]
        c = result["close"]
        body = abs(c - o)
        upper_shadow = h - np.maximum(c, o)
        lower_shadow = np.minimum(c, o) - low
        range_hl = np.where((h - low) == 0, 1e-4, h - low)
        range_series = pd.Series(range_hl, index=result.index)

        body_ratio = body / range_series
        upper_ratio = upper_shadow / range_series
        lower_ratio = lower_shadow / range_series
        bullish = c > o
        bearish = o > c
        # prev_open = o.shift(1)
        # prev_close = c.shift(1)
        prev_high = h.shift(1)
        prev_low = low.shift(1)
        prev_body = body.shift(1)
        # prev_body_ratio = body_ratio.shift(1)

        result["DOJI"] = (body_ratio < 0.1).astype(int)
        result["HAMMER"] = ((lower_shadow > 2 * body) & (upper_shadow < 0.3 * body) & (body_ratio < 0.3)).astype(int)
        result["INVERTED_HAMMER"] = (
            (upper_shadow > 2 * body) & (lower_shadow < 0.3 * body) & (body_ratio < 0.3)
        ).astype(int)
        result["SHOOTING_STAR"] = ((upper_shadow > 2 * body) & (lower_shadow < 0.3 * body) & (c < o)).astype(int)
        result["MARUBOZU_BULL"] = (
            bullish & (body_ratio > 0.9) & (upper_shadow < 0.05 * body) & (lower_shadow < 0.05 * body)
        ).astype(int)
        result["MARUBOZU_BEAR"] = (
            bearish & (body_ratio > 0.9) & (upper_shadow < 0.05 * body) & (lower_shadow < 0.05 * body)
        ).astype(int)
        result["SPINNING_TOP"] = ((body_ratio < 0.3) & (upper_shadow > body) & (lower_shadow > body)).astype(int)
        result["DRAGONFLY_DOJI"] = ((body_ratio < 0.1) & (upper_ratio < 0.1) & (lower_ratio > 0.6)).astype(int)
        result["GRAVESTONE_DOJI"] = ((body_ratio < 0.1) & (lower_ratio < 0.1) & (upper_ratio > 0.6)).astype(int)

        prev_bearish = o.shift(1) > c.shift(1)
        curr_bullish = bullish
        result["BULLISH_ENGULFING"] = (prev_bearish & curr_bullish & (c > o.shift(1)) & (o < c.shift(1))).astype(int)

        prev_bullish = c.shift(1) > o.shift(1)
        curr_bearish = bearish
        result["BEARISH_ENGULFING"] = (prev_bullish & curr_bearish & (o > c.shift(1)) & (c < o.shift(1))).astype(int)
        result["BULLISH_HARAMI"] = (
            prev_bearish & curr_bullish & (o >= c.shift(1)) & (c <= o.shift(1)) & (body <= 0.6 * prev_body)
        ).astype(int)
        result["BEARISH_HARAMI"] = (
            prev_bullish & curr_bearish & (o <= c.shift(1)) & (c >= o.shift(1)) & (body <= 0.6 * prev_body)
        ).astype(int)
        result["HARAMI_CROSS_BULL"] = (result["BULLISH_HARAMI"].astype(bool) & (body_ratio < 0.1)).astype(int)
        result["HARAMI_CROSS_BEAR"] = (result["BEARISH_HARAMI"].astype(bool) & (body_ratio < 0.1)).astype(int)

        body_prev = body.shift(1)
        range_prev = range_series.shift(1)
        first_bearish = o.shift(2) > c.shift(2)
        second_small = (body_prev / range_prev) < 0.3
        third_bullish = c > o
        result["MORNING_STAR"] = (
            first_bearish & second_small & third_bullish & (c > (o.shift(2) + c.shift(2)) / 2)
        ).astype(int)

        first_bullish_es = c.shift(2) > o.shift(2)
        second_small_es = (body_prev / range_prev) < 0.3
        third_bearish_es = o > c
        result["EVENING_STAR"] = (
            first_bullish_es & second_small_es & third_bearish_es & (c < (o.shift(2) + c.shift(2)) / 2)
        ).astype(int)

        prev_bearish = o.shift(1) > c.shift(1)
        curr_bullish = c > o
        gap_down = o < c.shift(1)
        midpoint = (o.shift(1) + c.shift(1)) / 2
        above_midpoint = c > midpoint
        below_prev_open = c < o.shift(1)

        result["PIERCING"] = (prev_bearish & curr_bullish & gap_down & above_midpoint & below_prev_open).astype(int)

        prev_bullish_dc = c.shift(1) > o.shift(1)
        curr_bearish_dc = o > c
        gap_up_dc = o > c.shift(1)
        midpoint_dc = (o.shift(1) + c.shift(1)) / 2
        below_midpoint_dc = c < midpoint_dc
        above_prev_open_dc = c > o.shift(1)

        result["DARK_CLOUD"] = (
            prev_bullish_dc & curr_bearish_dc & gap_up_dc & below_midpoint_dc & above_prev_open_dc
        ).astype(int)
        result["THREE_WHITE_SOLDIERS"] = (
            bullish
            & bullish.shift(1)
            & bullish.shift(2)
            & (c > c.shift(1))
            & (c.shift(1) > c.shift(2))
            & (o >= o.shift(1))
            & (o <= c.shift(1))
            & (o.shift(1) >= o.shift(2))
            & (o.shift(1) <= c.shift(2))
        ).astype(int)
        result["THREE_BLACK_CROWS"] = (
            bearish
            & bearish.shift(1)
            & bearish.shift(2)
            & (c < c.shift(1))
            & (c.shift(1) < c.shift(2))
            & (o <= o.shift(1))
            & (o >= c.shift(1))
            & (o.shift(1) <= o.shift(2))
            & (o.shift(1) >= c.shift(2))
        ).astype(int)
        result["THREE_INSIDE_UP"] = (
            bearish.shift(2)
            & bullish.shift(1)
            & bullish
            & (o.shift(1) >= c.shift(2))
            & (c.shift(1) <= o.shift(2))
            & (c > c.shift(1))
        ).astype(int)
        result["THREE_INSIDE_DOWN"] = (
            bullish.shift(2)
            & bearish.shift(1)
            & bearish
            & (o.shift(1) <= c.shift(2))
            & (c.shift(1) >= o.shift(2))
            & (c < c.shift(1))
        ).astype(int)
        tweezer_tolerance = 0.1 * range_series.combine(range_series.shift(1), np.minimum)
        result["TWEEZER_TOP"] = (bullish.shift(1) & bearish & (np.abs(h - h.shift(1)) <= tweezer_tolerance)).astype(int)
        result["TWEEZER_BOTTOM"] = (
            bearish.shift(1) & bullish & (np.abs(low - low.shift(1)) <= tweezer_tolerance)
        ).astype(int)
        result["RISING_WINDOW"] = (low > prev_high).astype(int)
        result["FALLING_WINDOW"] = (h < prev_low).astype(int)
        gap_up_prev = low.shift(1) > h.shift(2)
        gap_down_prev = h.shift(1) < low.shift(2)
        result["TASUKI_GAP_BULL"] = (
            gap_up_prev
            & bullish.shift(2)
            & bullish.shift(1)
            & bearish
            & (o >= o.shift(1))
            & (o <= c.shift(1))
            & (c > o.shift(2))
            & (c < o.shift(1))
        ).astype(int)
        result["TASUKI_GAP_BEAR"] = (
            gap_down_prev
            & bearish.shift(2)
            & bearish.shift(1)
            & bullish
            & (o <= o.shift(1))
            & (o >= c.shift(1))
            & (c < c.shift(2))
            & (c > c.shift(1))
        ).astype(int)
        strong_bullish = bullish & (body_ratio > 0.6)
        strong_bearish = bearish & (body_ratio > 0.6)
        small_body = body_ratio < 0.3
        mat_hold_mid_guard = (
            (low.shift(2) > o.shift(4))
            & (low.shift(1) > o.shift(4))
            & (h.shift(2) < c.shift(3))
            & (h.shift(1) < c.shift(3))
        )
        result["MAT_HOLD_BULL"] = (
            strong_bullish.shift(4)
            & bullish.shift(3)
            & gap_up_prev.shift(1)
            & small_body.shift(2)
            & small_body.shift(1)
            & mat_hold_mid_guard
            & bullish
            & (c > h.shift(1))
        ).astype(int)
        result["MAT_HOLD_BEAR"] = (
            strong_bearish.shift(4)
            & bearish.shift(3)
            & gap_down_prev.shift(1)
            & small_body.shift(2)
            & small_body.shift(1)
            & mat_hold_mid_guard
            & bearish
            & (c < low.shift(1))
        ).astype(int)
        result["ADVANCE_BLOCK"] = (
            bullish
            & bullish.shift(1)
            & bullish.shift(2)
            & (body < body.shift(1))
            & (body.shift(1) < body.shift(2))
            & (upper_shadow > body)
            & (upper_shadow.shift(1) > body.shift(1))
            & (upper_shadow.shift(2) > body.shift(2))
        ).astype(int)
        result["STALLED_PATTERN"] = (
            bullish
            & bullish.shift(1)
            & bullish.shift(2)
            & (body_ratio.shift(2) > 0.6)
            & (body_ratio.shift(1) > 0.6)
            & (body_ratio < 0.4)
            & (upper_shadow > body)
        ).astype(int)
        result["BELT_HOLD_BULL"] = (bullish & ((o - low) / range_series < 0.05) & (upper_ratio < 0.3)).astype(int)
        result["BELT_HOLD_BEAR"] = (bearish & ((h - o) / range_series < 0.05) & (lower_ratio < 0.3)).astype(int)
        result["KICKER_BULL"] = (prev_bearish & bullish & (o >= prev_high) & (body_ratio > 0.5)).astype(int)
        result["KICKER_BEAR"] = (prev_bullish & bearish & (o <= prev_low) & (body_ratio > 0.5)).astype(int)
        result["HANGING_MAN"] = (
            (lower_shadow > 2 * body) & (upper_shadow < body) & (body_ratio < 0.4) & bullish.shift(1)
        ).astype(int)

        metadata = collect_metadata(
            before,
            result.columns,
            CandlestickPatterns.CATEGORY,
        )
        return result, metadata


__all__ = ["CandlestickPatterns"]
