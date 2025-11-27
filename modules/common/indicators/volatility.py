"""Volatility indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from .base import IndicatorMetadata, IndicatorResult, collect_metadata


class VolatilityIndicators:
    """ATR-based volatility metrics."""

    CATEGORY = "volatility"

    @staticmethod
    def apply(df: pd.DataFrame) -> IndicatorResult:
        result = df.copy()
        before = result.columns.tolist()

        atr_14 = ta.atr(result["high"], result["low"], result["close"], length=14)
        result["ATR_14"] = (
            atr_14.ffill().fillna(result["close"] * 0.01)
            if atr_14 is not None
            else pd.Series(result["close"] * 0.01, index=result.index)
        )

        atr_50 = ta.atr(result["high"], result["low"], result["close"], length=50)
        result["ATR_50"] = (
            atr_50.ffill().fillna(result["close"] * 0.01)
            if atr_50 is not None
            else pd.Series(result["close"] * 0.01, index=result.index)
        )

        result["ATR_RATIO_14_50"] = result["ATR_14"] / result["ATR_50"]
        result["ATR_RATIO_14_50"] = (
            result["ATR_RATIO_14_50"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        )

        metadata = collect_metadata(
            before,
            result.columns,
            VolatilityIndicators.CATEGORY,
        )
        return result, metadata


__all__ = ["VolatilityIndicators"]
