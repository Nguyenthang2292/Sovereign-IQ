"""
ATC Symbol Analyzer Package.

This package provides modular components for analyzing symbols using
Adaptive Trend Classification (ATC).
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from .data_provider import DataProvider
from .price_source_selector import PriceSourceSelector
from .symbol_analyzer import SymbolAnalyzer

if TYPE_CHECKING:
    from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig
    from modules.common.core.data_fetcher import DataFetcher

__all__ = [
    "DataProvider",
    "PriceSourceSelector",
    "SymbolAnalyzer",
    "analyze_symbol",
]


def analyze_symbol(
    symbol: str,
    data_fetcher: "DataFetcher",
    config: "ATCConfig",
) -> Optional[Dict[str, Any]]:
    """
    Analyze a single symbol using ATC (Backward Compatibility Wrapper).

    This function delegates to SymbolAnalyzer.analyze().

    Args:
        symbol: Symbol to analyze
        data_fetcher: DataFetcher instance
        config: ATCConfig containing all ATC parameters

    Returns:
        Dictionary containing analysis results or None if failed.
    """
    analyzer = SymbolAnalyzer(data_fetcher)
    return analyzer.analyze(symbol, config)
