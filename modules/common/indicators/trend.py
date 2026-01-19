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

from modules.common.utils import validate_ohlcv_input

from .base import IndicatorResult, collect_metadata


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
        # Validate input - SMA only needs close, but ADX needs high/low/close
        # So we validate for close (required for SMA)
        validate_ohlcv_input(df, required_columns=["close"])

        result = df.copy()
        before = result.columns.tolist()

        # Moving averages
        result["SMA_20"] = ta.sma(result["close"], length=20)
        result["SMA_50"] = ta.sma(result["close"], length=50)
        result["SMA_200"] = ta.sma(result["close"], length=200)

        # Average Directional Index (ADX) - only calculate if high/low are available
        if "high" in result.columns and "low" in result.columns:
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
    plus_di = plus_dm.ewm(alpha=1 / period, adjust=False).mean() * 100 / atr.replace(0, np.nan)
    minus_di = minus_dm.ewm(alpha=1 / period, adjust=False).mean() * 100 / atr.replace(0, np.nan)

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    # Thay vì dropna(), sử dụng bfill() và fillna() để giữ index alignment với DataFrame gốc
    # Fill NaN bằng giá trị đầu tiên hợp lệ hoặc 0
    adx = adx.bfill().fillna(0.0)

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


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
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
# Weighted Moving Average Function
# ============================================================================


def calculate_weighted_ma(
    close: pd.Series,
    length: int = 50,
) -> pd.Series:
    """Calculate weighted moving average based on price deltas.

    This function calculates a weighted moving average where larger price
    movements receive higher weights. This emphasizes recent volatility and
    creates a more responsive equilibrium line compared to simple MA.

    Port of Pine Script logic:
        sumWeightedClose = 0.0
        sumWeights = 0.0
        for i = 0 to length - 1 by 1
            delta = math.abs(close[i] - close[i + 1])
            w = delta / close[i + 1]
            sumWeightedClose := sumWeightedClose + close[i] * w
            sumWeights := sumWeights + w
        ma = sumWeights != 0 ? sumWeightedClose / sumWeights : na

    Args:
        close: Close price series.
        length: Number of bars to use for calculation (default: 50).

    Returns:
        Series containing weighted moving average values.
        First `length` values are NaN.
    """
    if not isinstance(close, pd.Series):
        raise TypeError(f"close must be a pandas Series, got {type(close)}")
    if len(close) == 0:
        raise ValueError("close series cannot be empty")
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")

    if len(close) < length + 1:
        return pd.Series(np.nan, index=close.index, dtype="float64")

    # Vectorized implementation - much faster than nested loops
    prev_close = close.shift(1)
    delta = (close - prev_close).abs()

    # Handle division by zero
    weights = np.where((prev_close != 0) & (~pd.isna(prev_close)), delta / prev_close, 0.0)
    weights = pd.Series(weights, index=close.index)

    # Weighted close prices
    weighted_close = close * weights

    # Rolling sums
    sum_weighted = weighted_close.rolling(window=length).sum()
    sum_weights = weights.rolling(window=length).sum()

    # Calculate MA with fallback for zero weights
    ma = sum_weighted.copy()
    zero_weights_mask = (sum_weights == 0) | sum_weights.isna()
    ma[~zero_weights_mask] = sum_weighted[~zero_weights_mask] / sum_weights[~zero_weights_mask]
    ma[zero_weights_mask] = close.rolling(window=length).mean()[zero_weights_mask]

    # Ensure initial period is NaN to match intended logic and tests
    ma.iloc[:length] = np.nan

    return ma


# ============================================================================
# Trend Direction Function
# ============================================================================


def calculate_trend_direction(
    close: pd.Series,
    ma: pd.Series,
) -> pd.Series:
    """Calculate trend direction based on close vs weighted MA.

    Determines whether the current price is above or below the weighted MA,
    indicating bullish or bearish bias. This is used to select appropriate
    heatmap colors (bullish colors vs bearish colors).

    Port of Pine Script logic:
        var int trendDir = 0
        trendDir := close > ma ? 1 : close < ma ? -1 : nz(trendDir[1])

    Args:
        close: Close price series.
        ma: Moving average series (typically from calculate_weighted_ma).

    Returns:
        Series with trend direction: 1 (bullish), -1 (bearish), 0 (neutral).
    """
    if not isinstance(close, pd.Series) or not isinstance(ma, pd.Series):
        raise TypeError("close and ma must be pandas Series")
    if len(close) == 0 or len(ma) == 0:
        raise ValueError("close and ma series cannot be empty")
    if not close.index.equals(ma.index):
        raise ValueError("close and ma must have the same index")

    # Vectorized comparison
    trend_dir = pd.Series(0, index=close.index, dtype="int8")

    # Determine initial values based on close vs ma
    mask_bullish = (close > ma) & (~pd.isna(close)) & (~pd.isna(ma))
    mask_bearish = (close < ma) & (~pd.isna(close)) & (~pd.isna(ma))
    mask_equal = (close == ma) & (~pd.isna(close)) & (~pd.isna(ma))

    trend_dir[mask_bullish] = 1
    trend_dir[mask_bearish] = -1

    # Forward fill for equal/NA values (use previous value)
    trend_dir = trend_dir.replace(0, np.nan).ffill().fillna(0).astype("int8")

    return trend_dir


# ============================================================================
# DMI (Directional Movement Index) Functions
# ============================================================================


def calculate_dmi_difference(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 9) -> pd.Series:
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
    minus = 100 * minus_dm.ewm(alpha=1.0 / length, adjust=False).mean() / trur.replace(0, np.nan)

    diff = plus - minus
    return diff.fillna(0.0)


def calculate_ma_series(close: pd.Series, period: int, ma_type: str = "SMA") -> pd.Series:
    """
    Tính toán Moving Average với period tùy chỉnh.

    Args:
        close: Close price series
        period: Period cho MA
        ma_type: Loại MA ('SMA' hoặc 'EMA', default: 'SMA')

    Returns:
        Series với MA values
    """
    if ma_type.upper() == "EMA":
        return ta.ema(close, length=period)
    else:  # SMA
        return ta.sma(close, length=period)


__all__ = [
    "TrendIndicators",
    "calculate_adx",
    "calculate_adx_series",
    "calculate_cci",
    "calculate_dmi_difference",
    "calculate_weighted_ma",
    "calculate_trend_direction",
    "calculate_ma_series",
]
