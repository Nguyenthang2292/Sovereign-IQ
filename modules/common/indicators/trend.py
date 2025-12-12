"""Trend indicator block.

This module provides trend-following indicators including:
- Moving Averages (SMA, EMA)
- Average Directional Index (ADX)
- Commodity Channel Index (CCI)
- Directional Movement Index difference (DMI difference)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from .base import IndicatorMetadata, IndicatorResult, collect_metadata


class TrendIndicators:
    """
    Trend indicators: SMA, EMA, ADX, etc.
    
    This class provides a block-based approach for calculating multiple
    trend indicators at once, suitable for use with IndicatorEngine.
    """

    CATEGORY = "trend"

    @staticmethod
    def apply(df: pd.DataFrame) -> IndicatorResult:
        """
        Apply trend indicators to a DataFrame.
        
        Calculates:
        - SMA_20, SMA_50, SMA_200: Simple Moving Averages
        - ADX_14: Average Directional Index
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (result DataFrame, metadata dict)
        """
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


# ============================================================================
# ADX (Average Directional Index) Functions
# ============================================================================


def calculate_adx_series(ohlcv: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
    """
    Calculate ADX time-series for all periods.
    
    ADX (Average Directional Index) measures trend strength regardless of direction.
    Higher values indicate stronger trends.
    
    Args:
        ohlcv: DataFrame with high, low, close columns
        period: Period for ADX calculation (default: 14)
        
    Returns:
        Series with ADX values, or None if insufficient data
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
    
    Convenience function to get the most recent ADX value.
    
    Args:
        ohlcv: DataFrame with high, low, close columns
        period: Period for ADX calculation (default: 14)
        
    Returns:
        Latest ADX value as float, or None if unavailable
    """
    adx_series = calculate_adx_series(ohlcv, period=period)
    if adx_series is None or adx_series.empty:
        return None
    last_value = adx_series.iloc[-1]
    if pd.isna(last_value) or np.isinf(last_value):
        return None
    return float(last_value)


# ============================================================================
# CCI (Commodity Channel Index) Function
# ============================================================================


def calculate_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20
) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI).
    
    CCI is a trend indicator that measures the deviation of price from its
    statistical mean. High values indicate prices are well above their average,
    while low values indicate prices are well below their average.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        length: Period for CCI calculation (default: 20)
        
    Returns:
        Series with CCI values
    """
    cci = ta.cci(high=high, low=low, close=close, length=length)
    if cci is None:
        cci = pd.Series(0.0, index=close.index)
    return cci.fillna(0.0)


# ============================================================================
# DMI (Directional Movement Index) Functions
# ============================================================================


def calculate_dmi_difference(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 9
) -> pd.Series:
    """
    Calculate simplified DMI difference (plus - minus).
    
    This calculates the difference between +DI and -DI from the Directional
    Movement Index system. Positive values indicate bullish momentum,
    negative values indicate bearish momentum.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        length: Period for DMI calculation (default: 9)
        
    Returns:
        Series with DMI difference values (plus - minus)
    """
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=low.index,
    )

    # True Range
    tr_components = pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)

    # RMA (Running Moving Average) - using EWM with alpha = 1/length
    trur = tr.ewm(alpha=1.0 / length, adjust=False).mean()
    plus = 100 * plus_dm.ewm(alpha=1.0 / length, adjust=False).mean() / trur.replace(0, np.nan)
    minus = (
        100
        * minus_dm.ewm(alpha=1.0 / length, adjust=False).mean()
        / trur.replace(0, np.nan)
    )

    diff = plus - minus
    return diff.fillna(0.0)


__all__ = [
    "TrendIndicators",
    "calculate_adx",
    "calculate_adx_series",
    "calculate_cci",
    "calculate_dmi_difference",
]
