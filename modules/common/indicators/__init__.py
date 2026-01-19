"""Indicator utilities subpackage.

This module provides basic technical indicators. For advanced quantitative
metrics (CCI, Fisher Transform, DMI difference, Z-Score, MAR), see
modules.common.quantitative_metrics.
"""

from modules.common.utils import validate_ohlcv_input

from .base import (
    IndicatorFunc,
    IndicatorMetadata,
    IndicatorResult,
    collect_metadata,
)
from .blocks import BLOCK_SPECS, BlockSpec
from .candlestick import CandlestickPatterns
from .momentum import (
    MomentumIndicators,
    calculate_bollinger_bands_series,
    calculate_kama,
    calculate_kama_series,
    calculate_macd_series,
    calculate_rsi_series,
    calculate_stochrsi_series,
)
from .price_derived import PriceDerivedIndicators
from .trend import (
    TrendIndicators,
    calculate_adx,
    calculate_adx_series,
    calculate_cci,
    calculate_dmi_difference,
    calculate_ma_series,
    calculate_trend_direction,
    calculate_weighted_ma,
)
from .volatility import (
    VolatilityIndicators,
    calculate_atr_range,
    calculate_atr_series,
    calculate_returns_volatility,
)
from .volume import VolumeIndicators, calculate_obv_series

__all__ = [
    "calculate_kama",
    "calculate_kama_series",
    "calculate_adx",
    "calculate_adx_series",
    "calculate_cci",
    "calculate_dmi_difference",
    "calculate_weighted_ma",
    "calculate_trend_direction",
    "calculate_ma_series",
    "calculate_rsi_series",
    "calculate_macd_series",
    "calculate_bollinger_bands_series",
    "calculate_stochrsi_series",
    "IndicatorMetadata",
    "IndicatorResult",
    "IndicatorFunc",
    "collect_metadata",
    "validate_ohlcv_input",
    "PriceDerivedIndicators",
    "TrendIndicators",
    "MomentumIndicators",
    "VolatilityIndicators",
    "calculate_returns_volatility",
    "calculate_atr_range",
    "calculate_atr_series",
    "VolumeIndicators",
    "calculate_obv_series",
    "CandlestickPatterns",
    "BlockSpec",
    "BLOCK_SPECS",
]
