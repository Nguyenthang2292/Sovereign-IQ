"""Indicator block registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .base import IndicatorFunc, IndicatorMetadata, collect_metadata
from .candlestick import CandlestickPatterns
from .momentum import MomentumIndicators
from .trend import TrendIndicators
from .volatility import VolatilityIndicators
from .volume import VolumeIndicators


@dataclass(frozen=True)
class BlockSpec:
    """Simple descriptor for building IndicatorBlocks."""

    name: str
    toggle_attr: str
    handler: IndicatorFunc


BLOCK_SPECS: Tuple[BlockSpec, ...] = (
    BlockSpec(
        name=TrendIndicators.CATEGORY,
        toggle_attr="include_trend",
        handler=TrendIndicators.apply,
    ),
    BlockSpec(
        name=MomentumIndicators.CATEGORY,
        toggle_attr="include_momentum",
        handler=MomentumIndicators.apply,
    ),
    BlockSpec(
        name=VolatilityIndicators.CATEGORY,
        toggle_attr="include_volatility",
        handler=VolatilityIndicators.apply,
    ),
    BlockSpec(
        name=VolumeIndicators.CATEGORY,
        toggle_attr="include_volume",
        handler=VolumeIndicators.apply,
    ),
    BlockSpec(
        name=CandlestickPatterns.CATEGORY,
        toggle_attr="include_candlestick",
        handler=CandlestickPatterns.apply,
    ),
)


__all__ = [
    "IndicatorMetadata",
    "IndicatorFunc",
    "collect_metadata",
    "TrendIndicators",
    "MomentumIndicators",
    "VolatilityIndicators",
    "VolumeIndicators",
    "CandlestickPatterns",
    "BlockSpec",
    "BLOCK_SPECS",
]
