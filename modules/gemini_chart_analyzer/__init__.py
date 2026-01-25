"""
Gemini Chart Analyzer Module

This module provides technical chart analysis functionality using Google Gemini AI:
- Generate charts from OHLCV data with indicators (MA, RSI, etc.)
- Save charts as images
- Send images to Google Gemini for analysis and produce LONG/SHORT signals with TP/SL
- Batch market scanning: scan entire market by batching symbols
"""

from modules.gemini_chart_analyzer.core.aggregators.signal_aggregator import SignalAggregator
from modules.gemini_chart_analyzer.core.analyzers.gemini_batch_chart_analyzer import GeminiBatchChartAnalyzer
from modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer import GeminiChartAnalyzer
from modules.gemini_chart_analyzer.core.analyzers.multi_timeframe_coordinator import MultiTimeframeCoordinator
from modules.gemini_chart_analyzer.core.generators.chart_batch_generator import ChartBatchGenerator
from modules.gemini_chart_analyzer.core.generators.chart_generator import ChartGenerator
from modules.gemini_chart_analyzer.core.generators.chart_multi_timeframe_batch_generator import (
    ChartMultiTimeframeBatchGenerator,
)
from modules.gemini_chart_analyzer.core.generators.simple_chart_generator import SimpleChartGenerator
from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import MarketBatchScanner
from modules.gemini_chart_analyzer.services.batch_scan_service import BatchScanConfig, run_batch_scan

__all__ = [
    "ChartGenerator",
    "GeminiChartAnalyzer",
    "SimpleChartGenerator",
    "ChartBatchGenerator",
    "GeminiBatchChartAnalyzer",
    "MarketBatchScanner",
    "MultiTimeframeCoordinator",
    "SignalAggregator",
    "ChartMultiTimeframeBatchGenerator",
    "BatchScanConfig",
    "run_batch_scan",
]
