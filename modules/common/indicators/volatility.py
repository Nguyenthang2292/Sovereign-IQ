
import numpy as np
import pandas as pd

from .base import IndicatorResult, collect_metadata
from __future__ import annotations
from modules.common.utils import validate_ohlcv_input
from modules.common.utils.data import validate_price_series
import pandas_ta as ta
import pandas_ta as ta

"""Volatility indicator block."""






def calculate_returns_volatility(df: pd.DataFrame) -> float:
    """
    Calculate volatility from price returns.

    Computes the standard deviation of returns (pct_change) as a measure of volatility.

    Args:
        df: DataFrame with OHLCV data (must have 'close' column)

    Returns:
        Volatility value (standard deviation of returns), or 0.0 if insufficient data
    """
    if "close" not in df.columns or len(df) < 2:
        return 0.0

    returns = df["close"].pct_change().dropna()
    if len(returns) == 0:
        return 0.0

    return float(returns.std())


def calculate_atr_series(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14, fallback_multiplier: float = 0.01
) -> pd.Series:
    """
    Calculate ATR (Average True Range) series with fallback handling.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        length: ATR period length (default: 14)
        fallback_multiplier: Multiplier for fallback value (default: 0.01 = 1%)

    Returns:
        Series with ATR values, with fallback applied if needed
    """
    # Input validation
    validate_price_series(high, low, close)
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")

    atr = ta.atr(high, low, close, length=length)
    if atr is not None:
        return atr.ffill().fillna(close * fallback_multiplier)
    else:
        return close * fallback_multiplier


def _calculate_atr_with_fallback(df: pd.DataFrame, length: int, fallback_multiplier: float = 0.01) -> pd.Series:
    """
    Calculate ATR for a given period with fallback handling.

    Args:
        df: DataFrame with OHLCV data
        length: ATR period length
        fallback_multiplier: Multiplier for fallback value (default: 0.01 = 1%)

    Returns:
        Series with ATR values, with fallback applied if needed
    """
    return calculate_atr_series(df["high"], df["low"], df["close"], length, fallback_multiplier)


class VolatilityIndicators:
    """ATR-based volatility metrics."""

    CATEGORY = "volatility"

    @staticmethod
    def apply(df: pd.DataFrame) -> IndicatorResult:
        # Validate input
        validate_ohlcv_input(df, required_columns=["high", "low", "close"])

        result = df.copy()
        before = result.columns.tolist()

        # Calculate ATR for different periods using helper function
        result["ATR_14"] = _calculate_atr_with_fallback(result, length=14)
        result["ATR_50"] = _calculate_atr_with_fallback(result, length=50)

        # Improved division by zero handling: avoid warning by using where() instead of direct division
        atr_50_safe = result["ATR_50"].replace(0, np.nan)
        result["ATR_RATIO_14_50"] = np.where(
            pd.notna(atr_50_safe),
            result["ATR_14"] / atr_50_safe,
            1.0,  # Default value when ATR_50 is 0 or NaN
        )
        result["ATR_RATIO_14_50"] = (
            pd.Series(result["ATR_RATIO_14_50"], index=result.index).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        )

        metadata = collect_metadata(
            before,
            result.columns,
            VolatilityIndicators.CATEGORY,
        )
        return result, metadata


def calculate_atr_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    mult: float = 2.0,
    atr_length_primary: int = 2000,
    atr_length_fallback: int = 200,
) -> pd.Series:
    """Calculate ATR-based range bands.

    Calculates the Average True Range (ATR) and multiplies it by a factor
    to create dynamic range bands. These bands adapt to market volatility,
    expanding during volatile periods and contracting during quiet periods.

    Port of Pine Script logic:
        atrRaw = nz(ta.atr(2000), ta.atr(200))
        rangeATR = atrRaw * mult

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        mult: Multiplier for ATR (default: 2.0).
        atr_length_primary: Primary ATR length (default: 2000).
        atr_length_fallback: Fallback ATR length if primary fails (default: 200).

    Returns:
        Series containing ATR-based range values.
    """
    # Input validation
    validate_price_series(high, low, close)
    if mult <= 0:
        raise ValueError(f"mult must be > 0, got {mult}")
    if atr_length_primary < 1 or atr_length_fallback < 1:
        raise ValueError("ATR lengths must be >= 1")

    # Try primary ATR length first
    atr_raw = ta.atr(high, low, close, length=atr_length_primary)
    if atr_raw is None or (isinstance(atr_raw, pd.Series) and atr_raw.isna().all()):
        # Fallback to shorter ATR
        atr_raw = ta.atr(high, low, close, length=atr_length_fallback)

    # IMPROVEMENT (2025-01-16): Additional fallback for short data series.
    # If both primary and fallback ATR lengths are too large for the data,
    # try progressively smaller lengths (14, then 10, then 5) to ensure
    # we get valid ATR values even with limited data.
    if atr_raw is None or (isinstance(atr_raw, pd.Series) and atr_raw.isna().all()):
        data_length = len(close)
        # Try progressively smaller ATR lengths
        for fallback_len in [14, 10, 5]:
            if fallback_len <= data_length:
                atr_raw = ta.atr(high, low, close, length=fallback_len)
                if atr_raw is not None and isinstance(atr_raw, pd.Series) and atr_raw.notna().any():
                    break

    if atr_raw is None:
        # If all attempts fail, return NaN series
        return pd.Series(np.nan, index=close.index, dtype="float64")

    # Fill NaN values forward, then backward
    atr_raw = atr_raw.ffill().bfill()
    if atr_raw.isna().all():
        # If still all NaN, use a default value based on close price
        default_atr = close.abs() * 0.01
        atr_raw = pd.Series(default_atr, index=close.index)

    range_atr = atr_raw * mult
    return range_atr


__all__ = [
    "VolatilityIndicators",
    "calculate_returns_volatility",
    "calculate_atr_range",
    "calculate_atr_series",
]
