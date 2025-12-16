"""Volume indicator block."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from .base import IndicatorMetadata, IndicatorResult, collect_metadata
from modules.common.utils import validate_ohlcv_input


class VolumeIndicators:
    """Volume-driven signals."""

    CATEGORY = "volume"

    @staticmethod
    def apply(df: pd.DataFrame) -> IndicatorResult:
        # Validate input
        validate_ohlcv_input(df, required_columns=["close", "volume"])
        
        result = df.copy()
        before = result.columns.tolist()
        obv = ta.obv(result["close"], result["volume"])
        result["OBV"] = (
            obv.ffill().fillna(0.0)
            if obv is not None
            else pd.Series(0.0, index=result.index)
        )
        metadata = collect_metadata(
            before,
            result.columns,
            VolumeIndicators.CATEGORY,
        )
        return result, metadata


__all__ = ["VolumeIndicators"]
