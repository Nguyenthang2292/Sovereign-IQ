"""Trend indicator block."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from .base import IndicatorMetadata, IndicatorResult, collect_metadata


class TrendIndicators:
    """Trend indicators: SMA, EMA, ADX, etc."""

    CATEGORY = "trend"

    @staticmethod
    def apply(df: pd.DataFrame) -> IndicatorResult:
        result = df.copy()
        before = result.columns.tolist()

        # Moving averages
        result["SMA_20"] = ta.sma(result["close"], length=20)
        result["SMA_50"] = ta.sma(result["close"], length=50)
        result["SMA_200"] = ta.sma(result["close"], length=200)

        # Average Directional Index (ADX)
        adx_series = calculate_adx_series(result, period=14)
        if adx_series is not None:
            result["ADX_14"] = adx_series

        metadata = collect_metadata(before, result.columns, TrendIndicators.CATEGORY)
        return result, metadata


def calculate_adx_series(ohlcv: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
    """
    Calculate ADX time-series for all periods.
    """
    if ohlcv is None or len(ohlcv) < period * 2:
        return None

    required = {"high", "low", "close"}
    if not required.issubset(ohlcv.columns):
        return None

    data = ohlcv[list(required)].astype(float)
    high = data["high"]
    low = data["low"]
    close = data["close"]

    up_move = high.diff()
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=data.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=data.index,
    )

    tr_components = pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)

    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = (
        plus_dm.ewm(alpha=1 / period, adjust=False).mean()
        * 100
        / atr.replace(0, np.nan)
    )
    minus_di = (
        minus_dm.ewm(alpha=1 / period, adjust=False).mean()
        * 100
        / atr.replace(0, np.nan)
    )

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    adx = adx.dropna()

    if adx.empty:
        return None

    return adx


def calculate_adx(ohlcv: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Calculate single ADX value (latest).
    """
    adx_series = calculate_adx_series(ohlcv, period=period)
    if adx_series is None or adx_series.empty:
        return None
    last_value = adx_series.iloc[-1]
    if pd.isna(last_value) or np.isinf(last_value):
        return None
    return float(last_value)


__all__ = ["TrendIndicators", "calculate_adx", "calculate_adx_series"]
