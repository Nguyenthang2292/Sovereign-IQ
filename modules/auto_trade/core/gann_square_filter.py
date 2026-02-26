"""
Gann Square Filter Module

Optional final filter layer that iterates ranked candidates top-down until one passes
Gann analysis (or skip cycle if all rejected).

Usage:
    gann_filter = GannSquareFilter(timeframe="1h", limit=200, charts_dir="charts")
    result = gann_filter.run(ranked_signals)
    if result:
        # Process tradeable result
"""

import glob
import importlib
import os
from typing import TYPE_CHECKING, Any, List, Optional

from modules.auto_trade.core.signal_selector import FinalSignal
from modules.common.ui.logging import log_error, log_info, log_warn

if TYPE_CHECKING:
    from modules.gemini_gann_square.core.gann_signal_engine import GannAnalysisResult

_GANN_ENGINE_CLS: Any = None

try:
    _gann_module = importlib.import_module("modules.gemini_gann_square.core.gann_signal_engine")
    _GANN_ENGINE_CLS = getattr(_gann_module, "GannSignalEngine", None)
except ImportError as e:
    log_warn(f"GannSquareFilter: Could not import GannSignalEngine: {e}")


class GannSquareFilter:
    """Filters ranked signals through Gann Square analysis."""

    def __init__(
        self,
        timeframe: str = "1h",
        limit: int = 200,
        lookback: int = 5,
        charts_dir: str = "charts",
        gemini_api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the Gann Square filter.

        Args:
            timeframe: Chart timeframe for Gann analysis (e.g., '4h').
            limit: Number of candles to fetch for analysis.
            lookback: Zigzag pivot lookback window (default 5).
            charts_dir: Directory to save/clean chart PNGs.
            gemini_api_key: Optional Gemini API key (falls back to config).
        """
        self.timeframe = timeframe
        self.limit = limit
        self.lookback = lookback
        self.charts_dir = charts_dir
        self.gemini_api_key = gemini_api_key
        self.gann_engine: Optional[Any] = None

        if _GANN_ENGINE_CLS is not None:
            self.gann_engine = _GANN_ENGINE_CLS(
                lookback=self.lookback,
                gemini_api_key=gemini_api_key,
                chart_output_dir=charts_dir,
            )

    def _clean_charts_dir(self) -> None:
        """Delete all *.png files in charts_dir root (keep subdirs intact)."""
        if not os.path.exists(self.charts_dir):
            return

        pattern = os.path.join(self.charts_dir, "*.png")
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
            except OSError as e:
                log_warn(f"Failed to remove chart {file_path}: {e}")

    def run(self, ranked_signals: List[FinalSignal]) -> Optional["GannAnalysisResult"]:
        """
        Iterate ranked signals through Gann analysis.

        Args:
            ranked_signals: List of FinalSignal candidates (high → low score).

        Returns:
            GannAnalysisResult if a tradeable signal is found, None otherwise.
        """
        if not ranked_signals:
            log_info("GannSquareFilter: No ranked signals to process.")
            return None

        if self.gann_engine is None:
            log_warn("GannSquareFilter: GannSignalEngine not available, skipping filter.")
            return None

        for idx, signal in enumerate(ranked_signals):
            symbol = signal.symbol

            try:
                self._clean_charts_dir()

                log_info(f"GannSquareFilter: Analyzing {symbol} (rank {idx + 1}/{len(ranked_signals)})")

                result = self.gann_engine.analyze(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    limit=self.limit,
                )

                if result.is_tradeable():
                    log_info(f"GannSquare: {symbol} PASS — {result.signal} (confidence: {result.confidence_pct}%)")
                    return result

                if result.signal == "SKIP":
                    reasoning_preview = result.reasoning[:80] if result.reasoning else "No reasoning"
                    log_info(f"GannSquare: {symbol} SKIP — {reasoning_preview}")
                    continue

                log_info(f"GannSquare: {symbol} rejected (signal: {result.signal})")

            except Exception as e:
                log_error(f"GannSquareFilter: Error analyzing {symbol}: {e}")
                continue

        log_warn("GannSquareFilter: All candidates rejected by Gann analysis.")
        return None
