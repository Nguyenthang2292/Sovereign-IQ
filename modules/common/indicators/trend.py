
from typing import Optional

import numpy as np
import pandas as pd

from .base import IndicatorResult, collect_metadata
from __future__ import annotations
from modules.common.utils import validate_ohlcv_input
import pandas_ta as ta
from modules.common.utils import validate_ohlcv_input
import pandas_ta as ta

"""Trend indicator block.

This module provides trend-following indicators including:
- Moving Averages (SMA, EMA)
- Average Directional Index (ADX)
- Commodity Channel Index (CCI)
- Directional Movement Index difference (DMI difference)
"""







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
    # Input validation
    if not isinstance(close, pd.Series):
        raise TypeError(f"close must be a pandas Series, got {type(close)}")
    if len(close) == 0:
        raise ValueError("close series cannot be empty")
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")

    # If data is shorter than required length, return all NaN.
    # This matches the expected behavior: insufficient data means no valid MA values.
    if len(close) < length + 1:
        return pd.Series(np.nan, index=close.index, dtype="float64")

    ma_values = []
    for i in range(len(close)):
        if i < length:
            ma_values.append(np.nan)
            continue

        sum_weighted_close = 0.0
        sum_weights = 0.0

        for j in range(length):
            idx = i - j
            prev_idx = idx - 1
            if prev_idx < 0:
                break

            # Check for NaN values
            if pd.isna(close.iloc[idx]) or pd.isna(close.iloc[prev_idx]):
                continue

            delta = abs(close.iloc[idx] - close.iloc[prev_idx])
            prev_close = close.iloc[prev_idx]
            if prev_close == 0 or pd.isna(prev_close):
                w = 0.0
            else:
                w = delta / prev_close

            sum_weighted_close += close.iloc[idx] * w
            sum_weights += w

        # IMPROVEMENT (2025-01-16): Handle case when sum_weights = 0 (constant prices).
        # Fallback to simple average when all prices are the same to avoid NaN.
        if sum_weights != 0 and not pd.isna(sum_weights):
            ma_value = sum_weighted_close / sum_weights
        else:
            # If all prices are constant (sum_weights = 0), use simple average
            ma_value = close.iloc[i - length + 1 : i + 1].mean()
        ma_values.append(ma_value)

    return pd.Series(ma_values, index=close.index, dtype="float64")


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
        Series with trend direction:
        - 1: Bullish (close > MA)
        - -1: Bearish (close < MA)
        - 0: Neutral (uses previous value if close == MA)
    """
    # Input validation
    if not isinstance(close, pd.Series) or not isinstance(ma, pd.Series):
        raise TypeError("close and ma must be pandas Series")
    if len(close) == 0 or len(ma) == 0:
        raise ValueError("close and ma series cannot be empty")
    if not close.index.equals(ma.index):
        raise ValueError("close and ma must have the same index")

    trend_dir = pd.Series(0, index=close.index, dtype="int8")

    for i in range(len(close)):
        # Bounds checking
        if i >= len(ma):
            # Use previous value if available
            if i > 0:
                trend_dir.iloc[i] = trend_dir.iloc[i - 1]
            continue

        if pd.isna(close.iloc[i]) or pd.isna(ma.iloc[i]):
            # Use previous value if available
            if i > 0:
                trend_dir.iloc[i] = trend_dir.iloc[i - 1]
            continue

        close_value = close.iloc[i]
        ma_value = ma.iloc[i]

        if close_value > ma_value:
            trend_dir.iloc[i] = 1
        elif close_value < ma_value:
            trend_dir.iloc[i] = -1
        else:
            # Use previous value
            if i > 0:
                trend_dir.iloc[i] = trend_dir.iloc[i - 1]

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
