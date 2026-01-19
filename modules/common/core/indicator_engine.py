"""
Technical indicator and candlestick pattern orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from modules.common.indicators.blocks import (
    BLOCK_SPECS,
    BlockSpec,
    IndicatorFunc,
    IndicatorMetadata,
    collect_metadata,
)


class IndicatorProfile(str, Enum):
    """Predefined indicator bundles."""

    CORE = "core"
    XGBOOST = "xgboost"
    DEEP_LEARNING = "deep_learning"


@dataclass
class IndicatorConfig:
    """Configuration for indicator engine."""

    include_price_derived: bool = True  # Price-derived features (returns, log_volume, etc.)
    include_trend: bool = True
    include_momentum: bool = True
    include_volatility: bool = True
    include_volume: bool = True
    include_candlestick: bool = False
    custom_indicators: Dict[str, IndicatorFunc] = field(default_factory=dict)

    @classmethod
    def for_profile(cls, profile: IndicatorProfile) -> IndicatorConfig:
        """Create config presets for known profiles."""
        if profile == IndicatorProfile.CORE:
            return cls(include_candlestick=False)
        if profile == IndicatorProfile.XGBOOST:
            return cls(include_candlestick=True)
        if profile == IndicatorProfile.DEEP_LEARNING:
            return cls(include_candlestick=False)
        return cls()


@dataclass
class CustomIndicator:
    """Runtime-registered indicator definition."""

    name: str
    handler: IndicatorFunc
    category: str

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, IndicatorMetadata]:
        result = self.handler(df)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        if isinstance(result, pd.DataFrame):
            return result, {}
        raise TypeError(
            f"Custom indicator '{self.name}' must return either a DataFrame or a (DataFrame, metadata dict) tuple."
        )


class IndicatorEngine:
    """Reusable indicator engine orchestrating technical features."""

    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()
        self._custom_registry: Dict[str, CustomIndicator] = {}
        self._block_specs: Sequence[BlockSpec] = BLOCK_SPECS

    def register_indicator(
        self,
        name: str,
        func: IndicatorFunc,
        category: Optional[str] = None,
    ) -> None:
        """Register a custom indicator handler."""
        self._custom_registry[name] = CustomIndicator(
            name=name,
            handler=func,
            category=category or f"custom:{name}",
        )

    def _run_builtin_blocks(
        self,
        df: pd.DataFrame,
        cfg: IndicatorConfig,
    ) -> Tuple[pd.DataFrame, IndicatorMetadata]:
        metadata: IndicatorMetadata = {}
        working_df = df

        for spec in self._block_specs:
            if not getattr(cfg, spec.toggle_attr, False):
                continue
            working_df, block_meta = spec.handler(working_df)
            metadata.update(block_meta)

        return working_df, metadata

    def _prepare_custom_registry(
        self,
        cfg: IndicatorConfig,
    ) -> Dict[str, CustomIndicator]:
        dynamic_custom = {
            name: CustomIndicator(
                name=name,
                handler=func,
                category=f"custom:{name}",
            )
            for name, func in cfg.custom_indicators.items()
            if name not in self._custom_registry
        }
        return {**self._custom_registry, **dynamic_custom}

    def compute_features(
        self,
        df: pd.DataFrame,
        config: Optional[IndicatorConfig] = None,
        return_metadata: bool = False,
    ) -> Tuple[pd.DataFrame, IndicatorMetadata] | pd.DataFrame:
        """Calculate enabled indicator blocks and optionally return metadata."""
        cfg = config or self.config
        working_df, metadata = self._run_builtin_blocks(df.copy(), cfg)

        for custom in self._prepare_custom_registry(cfg).values():
            before_cols = working_df.columns.tolist()
            working_df, custom_meta = custom.run(working_df)
            if not custom_meta:
                custom_meta = collect_metadata(
                    before_cols,
                    working_df.columns,
                    custom.category,
                )
            metadata.update(custom_meta)

        if return_metadata:
            return working_df, metadata
        return working_df
