"""Volatility indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from .base import IndicatorMetadata, IndicatorResult, collect_metadata


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


def _calculate_atr_with_fallback(
    df: pd.DataFrame, length: int, fallback_multiplier: float = 0.01
) -> pd.Series:
    """
    Calculate ATR for a given period with fallback handling.
    
    Args:
        df: DataFrame with OHLCV data
        length: ATR period length
        fallback_multiplier: Multiplier for fallback value (default: 0.01 = 1%)
        
    Returns:
        Series with ATR values, with fallback applied if needed
    """
    atr = ta.atr(df["high"], df["low"], df["close"], length=length)
    if atr is not None:
        return atr.ffill().fillna(df["close"] * fallback_multiplier)
    else:
        return pd.Series(df["close"] * fallback_multiplier, index=df.index)


class VolatilityIndicators:
    """ATR-based volatility metrics."""

    CATEGORY = "volatility"

    @staticmethod
    def apply(df: pd.DataFrame) -> IndicatorResult:
        result = df.copy()
        before = result.columns.tolist()

        # Calculate ATR for different periods using helper function
        result["ATR_14"] = _calculate_atr_with_fallback(result, length=14)
        result["ATR_50"] = _calculate_atr_with_fallback(result, length=50)

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


__all__ = ["VolatilityIndicators", "calculate_returns_volatility"]
