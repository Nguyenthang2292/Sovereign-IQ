"""
Gemini Integration for Auto Trading.

Uses Google Gemini to analyze chart patterns and validate signals.
"""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
from pathlib import Path
import time
import uuid
import tempfile
import shutil

from modules.auto_trade.core.atc_scanner import SignalResult
from modules.common.core.data_fetcher import DataFetcher
from modules.common.ui.logging import log_debug, log_error, log_info, log_warn
from modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer import GeminiChartAnalyzer
from modules.gemini_chart_analyzer.core.generators.chart_generator import ChartGenerator


@dataclass
class GeminiSignal:
    """Structured signal from Gemini analysis."""

    trend: str
    signal: str  # LONG, SHORT, NONE
    confidence: float
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""


class GeminiIntegration:
    """Integrates Gemini Chart Analyzer into the trading pipeline."""

    # Default indicator configuration
    DEFAULT_INDICATORS = {"MA": {"periods": [20, 50, 200]}, "RSI": {"period": 14}, "MACD": {}, "BB": {}}

    def __init__(
        self,
        data_fetcher: DataFetcher,
        api_key: Optional[str] = None,
        analysis_timeframe: str = "1h",
        history_limit: int = 200,
        indicators: Optional[Dict] = None,
    ):
        """
        Initialize Gemini Integration.

        Args:
            data_fetcher: Data fetcher instance.
            api_key: Optional API key (if not in config).
            analysis_timeframe: Timeframe for chart analysis (default: "1h").
            history_limit: Number of candles to fetch (default: 200).
            indicators: Indicator configuration (uses DEFAULT_INDICATORS if None).

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        self.data_fetcher = data_fetcher

        # Get API key from parameter or environment variable
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self._api_key:
            log_warn(
                "No Gemini API key provided. " "Set GEMINI_API_KEY environment variable or pass api_key parameter."
            )

        # Use OS temp directory for temp files
        self.temp_dir = Path(tempfile.gettempdir()) / "gemini_charts"
        self.temp_dir.mkdir(exist_ok=True)

        # Validate and set timeframe
        valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.analysis_timeframe = analysis_timeframe
        if self.analysis_timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {self.analysis_timeframe}. " f"Must be one of {valid_timeframes}")

        # Validate and set history limit
        self.history_limit = history_limit
        if self.history_limit <= 0:
            raise ValueError(f"history_limit must be positive, got {self.history_limit}")

        # Set indicators
        self.indicators = indicators if indicators else self.DEFAULT_INDICATORS.copy()

        # Initialize chart generator and analyzer
        self.chart_generator = ChartGenerator(figsize=(12, 8), style="dark_background")
        self.analyzer = GeminiChartAnalyzer(api_key=self._api_key)

        # Rate limiting
        self.request_times = deque(maxlen=60)  # Track last 60 requests
        self.max_requests_per_minute = 60  # Gemini's typical limit

        # Caching
        self._cache: Dict[str, Tuple[GeminiSignal, datetime]] = {}
        self.cache_ttl = timedelta(hours=1)  # Cache for 1 hour

        # Cleanup old temp files on init
        self._cleanup_old_temp_files()

    def is_available(self) -> bool:
        """Check if Gemini integration is available (has API key)."""
        return self._api_key is not None and len(self._api_key) > 0

    def _check_rate_limit(self) -> None:
        """Ensure we don't exceed rate limits."""
        now = time.time()

        # Remove requests older than 1 minute
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()

        # If at limit, wait
        if len(self.request_times) >= self.max_requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                log_info(f"Rate limit reached, waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        self.request_times.append(now)

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._cache.clear()
        log_debug("Gemini analysis cache cleared")

    def _cleanup_old_temp_files(self) -> None:
        """Clean up old temp chart files on initialization."""
        try:
            if self.temp_dir.exists():
                # Remove files older than 1 hour
                now = time.time()
                for file_path in self.temp_dir.glob("*.png"):
                    if file_path.stat().st_mtime < now - 3600:  # 1 hour ago
                        try:
                            file_path.unlink()
                            log_debug(f"Removed old temp file: {file_path.name}")
                        except Exception as e:
                            log_warn(f"Failed to remove old temp file {file_path.name}: {e}")
        except Exception as e:
            log_warn(f"Failed to cleanup old temp files: {e}")

    def analyze_candidate(self, signal: SignalResult) -> Optional[GeminiSignal]:
        """
        Analyze a candidate symbol using Gemini.

        Args:
            signal: The SignalResult candidate to analyze.

        Returns:
            GeminiSignal object if successful, None otherwise.
        """
        symbol = signal.symbol
        # Use configured timeframe for pattern recognition context
        timeframe = self.analysis_timeframe

        log_info(f"Gemini: Analyzing {symbol} chart...")

        # 1. Check cache first
        if symbol in self._cache:
            cached_signal, cached_time = self._cache[symbol]
            if datetime.now() - cached_time < self.cache_ttl:
                log_debug(f"Using cached Gemini analysis for {symbol}")
                return cached_signal

        # 2. Fetch Data (with rate limiting)
        self._check_rate_limit()
        df, exchange_used = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol=symbol, timeframe=timeframe, limit=self.history_limit, check_freshness=False
        )
        log_debug(f"Fetched data from exchange: {exchange_used}")

        if df is None or df.empty:
            log_error(f"Gemini: No data fetched for {symbol}")
            return None

        # 2. Generate Temporary Chart with unique ID in temp directory
        unique_id = uuid.uuid4().hex[:8]
        temp_filename = str(self.temp_dir / f"chart_{symbol.replace('/', '_')}_{timeframe}_{unique_id}.png")
        chart_path = None

        try:
            chart_path = self.chart_generator.create_chart(
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                output_path=temp_filename,
                show_volume=True,
                indicators=self.indicators,
            )

            # 3. Call Gemini
            analysis_text = self.analyzer.analyze_chart(
                image_path=chart_path,
                symbol=symbol,
                timeframe=timeframe,
                prompt_type="detailed",  # Expects JSON output
            )

            # 4. Parse Result
            result = self._parse_gemini_response(analysis_text)

            # Cache result
            if result:
                self._cache[symbol] = (result, datetime.now())

            return result

        except Exception as e:
            log_error(f"Gemini analysis failed for {symbol}: {e}")
            return None
        finally:
            # 5. Cleanup
            if chart_path and os.path.exists(chart_path):
                try:
                    os.remove(chart_path)
                except Exception as e:
                    log_warn(f"Failed to remove temp chart {chart_path}: {e}")

    def _parse_gemini_response(self, text: str) -> Optional[GeminiSignal]:
        """Extract and parse JSON from Gemini response text."""
        try:
            # Try markdown code block first
            match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
            if not match:
                # Fallback to plain JSON (non-greedy)
                match = re.search(r"\{[\s\S]*?\}", text)
            if not match:
                log_warn("Gemini response did not contain valid JSON structure.")
                return None

            json_str = match.group(1) if match.lastindex else match.group(0)
            data = json.loads(json_str)

            # Normalize fields
            signal_raw = str(data.get("signal", "NONE")).upper()
            if "LONG" in signal_raw:
                signal_type = "LONG"
            elif "SHORT" in signal_raw:
                signal_type = "SHORT"
            else:
                signal_type = "NONE"

            # Validate and normalize confidence
            confidence = float(data.get("confidence", 0.0))

            # Normalize if percentage (0-100)
            if confidence > 1.0:
                confidence = confidence / 100.0

            # Clamp to valid range
            confidence = max(0.0, min(1.0, confidence))

            # Validate signal logic (entry/stop_loss)
            entry = self._safe_float(data.get("entry"))
            stop_loss = self._safe_float(data.get("stop_loss"))

            if entry is not None and stop_loss is not None:
                signal = str(data.get("signal", "NONE")).upper()
                if signal == "LONG" and stop_loss >= entry:
                    log_warn(f"Invalid LONG signal: stop_loss >= entry ({stop_loss} >= {entry})")
                    return None
                elif signal == "SHORT" and stop_loss <= entry:
                    log_warn(f"Invalid SHORT signal: stop_loss <= entry ({stop_loss} <= {entry})")
                    return None

            # Extract support/resistance reasoning or patterns
            patterns = data.get("patterns", [])
            reasoning = ", ".join(patterns) if isinstance(patterns, list) else str(patterns)

            # Sanitize reasoning (prevent injection, limit length)
            reasoning = reasoning[:500]

            return GeminiSignal(
                trend=str(data.get("trend", "Unknown")),
                signal=signal_type,
                confidence=confidence,
                entry=entry,
                stop_loss=stop_loss,
                take_profit=self._safe_float(data.get("take_profit")),
                reasoning=reasoning,
            )

        except json.JSONDecodeError as e:
            log_error(f"Failed to decode Gemini JSON: {e}")
            return None
        except Exception as e:
            log_error(f"Error parsing Gemini response: {e}")
            return None

    def _safe_float(self, val: Any) -> Optional[float]:
        """Convert value to float safely."""
        if val is None:
            return None
        if isinstance(val, (float, int)):
            return float(val)
        try:
            return float(str(val).replace(",", "").strip())
        except ValueError:
            return None
