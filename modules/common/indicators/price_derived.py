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

try:
    from modules.xgboost_LTS.rust_extensions import xgboost_rust

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


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

        if RUST_AVAILABLE:
            try:
                # Use Rust for batch calculation
                rust_results = xgboost_rust.add_price_derived_features_rust(
                    result["open"].values.astype(np.float64),
                    result["high"].values.astype(np.float64),
                    result["low"].values.astype(np.float64),
                    result["close"].values.astype(np.float64),
                    result["volume"].values.astype(np.float64),
                )
                for name, values in rust_results.items():
                    result[name] = values

                # Metadata collection handled at the end
            except Exception:
                # Fallback to Pandas
                PriceDerivedIndicators._apply_pandas(result)
        else:
            PriceDerivedIndicators._apply_pandas(result)

        # Fill NaN values (first rows for returns, etc.)
        for col in ["returns_1", "returns_5", "high_low_range", "close_open_diff"]:
            if col in result.columns:
                result[col] = result[col].fillna(0.0)

        metadata = collect_metadata(before, result.columns, PriceDerivedIndicators.CATEGORY)
        return result, metadata

    @staticmethod
    def _apply_pandas(result: pd.DataFrame):
        """Standard Pandas implementation for price-derived features."""
        # 1-period return: (close - close.shift(1)) / close.shift(1)
        result["returns_1"] = result["close"].pct_change(periods=1)

        # 5-period return: (close - close.shift(5)) / close.shift(5)
        result["returns_5"] = result["close"].pct_change(periods=5)

        # Log-normalized volume: log(volume + 1) to handle zero volumes
        result["log_volume"] = np.log1p(result["volume"])

        # High-Low range normalized by close: (high - low) / close
        result["high_low_range"] = (result["high"] - result["low"]) / result["close"]

        # Close-Open difference normalized by open: (close - open) / open
        result["close_open_diff"] = (result["close"] - result["open"]) / result["open"]


__all__ = ["PriceDerivedIndicators"]
