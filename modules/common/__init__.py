"""Common utilities shared across all components.

This package intentionally uses lazy exports to avoid heavy import side effects
and circular dependencies during app startup.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import indicators, quantitative_metrics
    from .core.data_fetcher import DataFetcher
    from .core.exchange_manager import ExchangeManager
    from .core.indicator_engine import (
        CustomIndicator,
        IndicatorConfig,
        IndicatorEngine,
        IndicatorProfile,
    )
    from .models.position import Position
    from .ui.progress_bar import ProgressBar

__all__ = [
    "IndicatorEngine",
    "IndicatorConfig",
    "IndicatorProfile",
    "CustomIndicator",
    "DataFetcher",
    "ExchangeManager",
    "ProgressBar",
    "Position",
    "indicators",
    "quantitative_metrics",
]


def __getattr__(name: str) -> Any:
    if name == "DataFetcher":
        from .core.data_fetcher import DataFetcher

        return DataFetcher
    if name == "ExchangeManager":
        from .core.exchange_manager import ExchangeManager

        return ExchangeManager
    if name in {"CustomIndicator", "IndicatorConfig", "IndicatorEngine", "IndicatorProfile"}:
        from .core.indicator_engine import (
            CustomIndicator,
            IndicatorConfig,
            IndicatorEngine,
            IndicatorProfile,
        )

        return {
            "CustomIndicator": CustomIndicator,
            "IndicatorConfig": IndicatorConfig,
            "IndicatorEngine": IndicatorEngine,
            "IndicatorProfile": IndicatorProfile,
        }[name]
    if name == "Position":
        from .models.position import Position

        return Position
    if name == "ProgressBar":
        from .ui.progress_bar import ProgressBar

        return ProgressBar
    if name in {"indicators", "quantitative_metrics"}:
        from importlib import import_module

        return import_module(f"{__name__}.{name}")
    raise AttributeError(name)
