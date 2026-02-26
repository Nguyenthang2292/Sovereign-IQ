"""
Gemini Gann Square Module

Combines Gann Square technical analysis with Google Gemini AI
to produce trading signals (LONG / SHORT / SKIP) with Entry, TP, SL.

Quick start (CLI):
    python -m modules.gemini_gann_square --symbol BTC/USDT --timeframe 4h

Or programmatic use:
    from modules.gemini_gann_square import GannSignalEngine
    engine = GannSignalEngine()
    result = engine.analyze(df, symbol="BTC/USDT", timeframe="4h")
    print(result.display())
"""

from modules.gemini_gann_square.core.gann_calculator import (
    GannCalculator,
    GannSquareResult,
    GannZone,
)
from modules.gemini_gann_square.core.gann_chart_generator import GannChartGenerator
from modules.gemini_gann_square.core.gann_signal_engine import (
    GannAnalysisResult,
    GannSignalEngine,
)
from modules.gemini_gann_square.core.swing_detector import SwingDetector, SwingPoint

__all__ = [
    "SwingDetector",
    "SwingPoint",
    "GannCalculator",
    "GannSquareResult",
    "GannZone",
    "GannChartGenerator",
    "GannSignalEngine",
    "GannAnalysisResult",
]
