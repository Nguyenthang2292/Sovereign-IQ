"""Volume indicator block."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from .base import IndicatorMetadata, IndicatorResult, collect_metadata
from modules.common.utils import validate_ohlcv_input


def calculate_obv_series(
    close: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    Calculate OBV (On-Balance Volume) series.
    
    Args:
        close: Close price series
        volume: Volume series
        
    Returns:
        Series with OBV values, with fallback applied if needed
    """
    # Input validation
    if not isinstance(close, pd.Series):
        raise TypeError(f"close must be a pandas.Series, got {type(close).__name__}")
    
    if not isinstance(volume, pd.Series):
        raise TypeError(f"volume must be a pandas.Series, got {type(volume).__name__}")
    
    if close.empty:
        raise ValueError("close series cannot be empty")
    
    if volume.empty:
        raise ValueError("volume series cannot be empty")
    
    if len(close) != len(volume):
        raise ValueError(
            f"close and volume must have the same length: "
            f"close has {len(close)} elements, volume has {len(volume)} elements"
        )
    
    if not close.index.equals(volume.index):
        raise ValueError(
            f"close and volume must have matching index values. "
            f"close.index: {close.index.tolist()[:5]}..., "
            f"volume.index: {volume.index.tolist()[:5]}..."
        )
    
    obv = ta.obv(close, volume)
    if obv is not None:
        # Ensure alignment to close.index with ffill/fillna fallback
        obv_aligned = obv.reindex(close.index).ffill().fillna(0.0)
        return obv_aligned
    else:
        return pd.Series(0.0, index=close.index)


class VolumeIndicators:
    """Volume-driven signals."""

    CATEGORY = "volume"

    @staticmethod
    def apply(df: pd.DataFrame) -> IndicatorResult:
        # Validate input
        validate_ohlcv_input(df, required_columns=["close", "volume"])
        
        result = df.copy()
        before = result.columns.tolist()
        result["OBV"] = calculate_obv_series(result["close"], result["volume"])
        metadata = collect_metadata(
            before,
            result.columns,
            VolumeIndicators.CATEGORY,
        )
        return result, metadata


__all__ = ["VolumeIndicators", "calculate_obv_series"]
