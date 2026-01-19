"""Price-derived indicator block.

This module provides normalized/derived features from raw OHLCV data:
- Returns (1-period, 5-period)
- Log-normalized volume
- High-Low range (normalized)
- Close-Open difference (normalized)

These features are scale-invariant and generalize across different assets and timeframes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.common.utils import validate_ohlcv_input

from .base import IndicatorResult, collect_metadata


class PriceDerivedIndicators:
    """
    Price-derived indicators: normalized features from OHLCV data.
    
    Calculates scale-invariant features that generalize across assets:
    - returns_1: 1-period return (pct_change)
    - returns_5: 5-period return (pct_change)
    - log_volume: Log-normalized volume
    - high_low_range: (high - low) / close (normalized range)
    - close_open_diff: (close - open) / open (normalized price change)
    """

    CATEGORY = "price_derived"

    @staticmethod
    def apply(df: pd.DataFrame) -> IndicatorResult:
        """
        Apply price-derived indicators to a DataFrame.

        Args:
            df: DataFrame with OHLCV data (must have open, high, low, close, volume)

        Returns:
            Tuple of (result DataFrame with indicators, metadata dict)
        """
        # Validate input - need all OHLCV columns
        validate_ohlcv_input(df, required_columns=["open", "high", "low", "close", "volume"])

        result = df.copy()
        before = result.columns.tolist()

        # 1-period return: (close - close.shift(1)) / close.shift(1)
        result["returns_1"] = result["close"].pct_change(periods=1)

        # 5-period return: (close - close.shift(5)) / close.shift(5)
        result["returns_5"] = result["close"].pct_change(periods=5)

        # Log-normalized volume: log(volume + 1) to handle zero volumes
        # Adding 1 prevents log(0) = -inf
        result["log_volume"] = np.log1p(result["volume"])

        # High-Low range normalized by close: (high - low) / close
        # This gives the price range as a percentage of current price
        result["high_low_range"] = (result["high"] - result["low"]) / result["close"]

        # Close-Open difference normalized by open: (close - open) / open
        # This gives the price change within the candle as a percentage
        result["close_open_diff"] = (result["close"] - result["open"]) / result["open"]

        # Fill NaN values (first rows for returns, etc.)
        result["returns_1"] = result["returns_1"].fillna(0.0)
        result["returns_5"] = result["returns_5"].fillna(0.0)
        result["high_low_range"] = result["high_low_range"].fillna(0.0)
        result["close_open_diff"] = result["close_open_diff"].fillna(0.0)

        metadata = collect_metadata(before, result.columns, PriceDerivedIndicators.CATEGORY)
        return result, metadata


__all__ = ["PriceDerivedIndicators"]
