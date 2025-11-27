"""Trend indicator block."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from .base import IndicatorMetadata, IndicatorResult, collect_metadata


class TrendIndicators:
    """Trend moving averages."""

    CATEGORY = "trend"

    @staticmethod
    def apply(df: pd.DataFrame) -> IndicatorResult:
        result = df.copy()
        before = result.columns.tolist()
        result["SMA_20"] = ta.sma(result["close"], length=20)
        result["SMA_50"] = ta.sma(result["close"], length=50)
        result["SMA_200"] = ta.sma(result["close"], length=200)
        metadata = collect_metadata(before, result.columns, TrendIndicators.CATEGORY)
        return result, metadata


__all__ = ["TrendIndicators"]
