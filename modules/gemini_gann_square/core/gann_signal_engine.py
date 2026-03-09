"""
Gann Signal Engine — Orchestrator.

Coordinates the full pipeline:
  1. Fetch OHLCV data
  2. Detect Swing High / Low (pivot zigzag)
  3. Calculate Gann Square zones
  4. Generate chart PNG with Gann overlay
  5. Build Gemini prompt with context
  6. Call Gemini for AI analysis
  7. Parse response → GannAnalysisResult
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from modules.common.ui.logging import log_error, log_info, log_warn
from modules.gemini_chart_analyzer.core.analyzers.vision_analyzer_chain import VisionAnalyzerChain

from .gann_calculator import GannCalculator, GannSquareResult, SignalCode
from .gann_chart_generator import GannChartGenerator
from .swing_detector import SwingDetector


@dataclass
class GannAnalysisResult:
    """Final output of the full Gann + Gemini pipeline."""

    symbol: str
    timeframe: str

    # Gann Square data
    gann_result: GannSquareResult

    # Chart path
    chart_path: str

    # Gemini parsed output
    zone_confirmed: int
    trend_confirmed: str
    override_reason: str
    signal: SignalCode
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    confidence_pct: int
    reasoning: str

    # Raw Gemini response (for debugging)
    raw_gemini_response: str = ""
    gemini_parse_error: str = ""

    def is_tradeable(self) -> bool:
        return self.signal in ("LONG", "SHORT")

    def display(self) -> str:
        """Format result for terminal display."""
        trend_arrow = "⬇" if self.gann_result.trend == "DOWN" else "⬆"
        signal_icon = {"LONG": "⬆ LONG", "SHORT": "⬇ SHORT", "SKIP": "⊘ SKIP"}.get(self.signal, self.signal)

        lines = [
            "═" * 50,
            "  GEMINI GANN SQUARE ANALYSIS",
            f"  Symbol: {self.symbol}  │  Timeframe: {self.timeframe}",
            "═" * 50,
            f"  Trend    : {self.gann_result.trend} {trend_arrow}",
            f"  Swing High: {self.gann_result.swing_high.price:,.4f}  ({self.gann_result.swing_high.timestamp})",
            f"  Swing Low : {self.gann_result.swing_low.price:,.4f}  ({self.gann_result.swing_low.timestamp})",
            f"  Range     : {self.gann_result.price_range:,.4f}",
            "",
            f"  Current Zone (code)  : Zone {self.gann_result.current_zone}  →  {self.gann_result.signal_code}",
            f"  Current Zone (Gemini): Zone {self.zone_confirmed}  →  {self.signal}",
        ]

        if self.override_reason:
            lines.append(f"  ⚠ Override: {self.override_reason}")

        lines += [
            "─" * 50,
            "  🤖 GEMINI FINAL RECOMMENDATION",
            "─" * 50,
            f"  Signal     : {signal_icon}",
        ]

        if self.is_tradeable():
            lines += [
                f"  Entry      : {self.entry_price:,.4f}",
                f"  Stop Loss  : {self.stop_loss:,.4f}",
                f"  TP1        : {self.take_profit_1:,.4f}",
                f"  TP2        : {self.take_profit_2:,.4f}",
            ]

        lines += [
            f"  Confidence : {self.confidence_pct}%",
            f"  Reasoning  : {self.reasoning}",
            "═" * 50,
            f"  Chart: {self.chart_path}",
            "═" * 50,
        ]

        return "\n".join(lines)


class GannSignalEngine:
    """
    Full Gann Square + Gemini analysis pipeline.

    Usage:
        engine = GannSignalEngine()
        result = engine.analyze(df, symbol="BTC/USDT", timeframe="4h")
        print(result.display())
    """

    _PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

    def __init__(
        self,
        lookback: int = 5,
        gemini_api_key: Optional[str] = None,
        qwen_api_key: Optional[str] = None,
        chart_output_dir: str = "charts",
    ) -> None:
        """
        Initialize the engine.

        Args:
            lookback: Zigzag pivot lookback window (default 5).
            gemini_api_key: Optional Gemini API key (falls back to config).
            qwen_api_key: Optional Qwen (Dashscope) API key (falls back to config).
            chart_output_dir: Directory to save chart PNGs.
        """
        self.swing_detector = SwingDetector(lookback=lookback)
        self.gann_calculator = GannCalculator()
        self.chart_generator = GannChartGenerator(output_dir=chart_output_dir)
        self.gemini_analyzer = VisionAnalyzerChain(
            gemini_api_key=gemini_api_key,
            qwen_api_key=qwen_api_key,
        )

    def analyze(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200,
        chart_output_path: Optional[str] = None,
    ) -> GannAnalysisResult:
        """
        Run the full Gann + Gemini analysis pipeline.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT').
            timeframe: Chart timeframe (e.g., '4h').
            limit: Number of candles to fetch.
            chart_output_path: Optional explicit path for the output chart.

        Returns:
            GannAnalysisResult with full analysis.

        Raises:
            ValueError: If swing points cannot be detected.
        """
        log_info(f"[GannSignalEngine] Starting analysis: {symbol} {timeframe}")

        # Imports are deferred here to avoid circular dependencies between
        # modules.common and modules.gemini_gann_square at import time.
        from modules.common.core.data_fetcher import DataFetcher
        from modules.common.core.exchange_manager import ExchangeManager

        exchange_manager = ExchangeManager()
        fetcher = DataFetcher(exchange_manager)
        df, exchange_id = fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            check_freshness=False,
        )

        if df is None or df.empty:
            raise ValueError(f"No OHLCV data returned for {symbol} {timeframe}.")

        log_info(f"Fetched {len(df)} candles from {exchange_id}.")

        # Step 1: Detect Swing High / Low
        swing_high, swing_low = self.swing_detector.get_significant_swings(df)
        if swing_high is None or swing_low is None:
            raise ValueError(
                f"Could not detect swing points for {symbol} {timeframe}. "
                f"Try increasing data limit or reducing lookback window."
            )
        log_info(f"  Swing High: {swing_high}")
        log_info(f"  Swing Low : {swing_low}")

        # Step 2: Calculate Gann Square zones
        current_price = float(df["close"].iloc[-1])
        gann_result = self.gann_calculator.calculate(swing_high, swing_low, current_price)
        log_info(f"  {gann_result.summary()}")

        # Step 3: Generate chart PNG
        chart_path = self.chart_generator.create_chart(
            df, gann_result, symbol, timeframe, output_path=chart_output_path
        )

        # Step 4: Build prompt with context
        prompt = self._build_prompt(symbol, timeframe, current_price, gann_result)

        # Step 5: Call Gemini with the chart image
        log_info("  Sending chart to Gemini for analysis...")
        raw_response = self.gemini_analyzer.analyze_chart(
            image_path=chart_path,
            symbol=symbol,
            timeframe=timeframe,
            prompt_type="custom",
            custom_prompt=prompt,
        )

        # Step 6: Parse Gemini response
        parsed, parse_error = self._parse_gemini_response(raw_response, gann_result)

        return GannAnalysisResult(
            symbol=symbol,
            timeframe=timeframe,
            gann_result=gann_result,
            chart_path=chart_path,
            raw_gemini_response=raw_response,
            gemini_parse_error=parse_error,
            **parsed,
        )

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _build_prompt(
        self,
        symbol: str,
        timeframe: str,
        current_price: float,
        gann: GannSquareResult,
    ) -> str:
        """Load prompt template and inject context variables."""
        prompt_file = self._PROMPTS_DIR / "gann_analysis.txt"
        try:
            template = prompt_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            log_warn(f"Prompt file not found: {prompt_file}, using fallback prompt.")
            return self._fallback_prompt(
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                gann=gann,
            )

        zones = gann.zones
        current_idx = gann.current_index
        replacements = {
            "{SYMBOL}": self._sanitize_prompt_text(symbol),
            "{TIMEFRAME}": self._sanitize_prompt_text(timeframe),
            "{TREND}": gann.trend,
            "{SWING_HIGH_PRICE}": f"{gann.swing_high.price:,.4f}",
            "{SWING_HIGH_TIME}": str(gann.swing_high.timestamp),
            "{SWING_LOW_PRICE}": f"{gann.swing_low.price:,.4f}",
            "{SWING_LOW_TIME}": str(gann.swing_low.timestamp),
            "{PRICE_RANGE}": f"{gann.price_range:,.4f}",
            "{CURRENT_PRICE}": f"{current_price:,.4f}",
            "{CURRENT_ZONE}": str(gann.current_zone),
            "{PRECALC_SIGNAL}": gann.signal_code,
        }
        for i, zone in enumerate(zones, start=1):
            replacements[f"{{ZONE{i}_UPPER}}"] = f"{zone.upper_price_at(current_idx):,.4f}"
            replacements[f"{{ZONE{i}_LOWER}}"] = f"{zone.lower_price_at(current_idx):,.4f}"
            replacements[f"{{ZONE{i}_SIGNAL}}"] = zone.signal

        template = re.sub(
            r"\{[A-Z0-9_]+\}",
            lambda match: replacements.get(match.group(0), match.group(0)),
            template,
        )

        return template

    def _parse_gemini_response(self, raw: str, gann: GannSquareResult) -> tuple[dict, str]:
        """
        Extract JSON from Gemini response.

        Returns (parsed_dict, error_string).
        Falls back to system-calculated values on parse failure.
        """
        try:
            # Extract JSON block (Gemini may wrap in ```json ... ```)
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in Gemini response.")

            data = json.loads(json_match.group())

            raw_signal = str(data.get("signal", gann.signal_code)).upper().strip()
            signal: SignalCode = raw_signal if raw_signal in ("LONG", "SHORT", "SKIP") else gann.signal_code

            zone_confirmed = self._safe_int(data.get("zone_confirmed", gann.current_zone), fallback=gann.current_zone)
            if zone_confirmed < 1 or zone_confirmed > 4:
                zone_confirmed = gann.current_zone

            confidence_pct = self._safe_int(data.get("confidence_pct", 0), fallback=0)
            confidence_pct = max(0, min(100, confidence_pct))

            entry_price = self._safe_price(data.get("entry_price", 0.0))
            stop_loss = self._safe_price(data.get("stop_loss", 0.0))
            take_profit_1 = self._safe_price(data.get("take_profit_1", 0.0))
            take_profit_2 = self._safe_price(data.get("take_profit_2", 0.0))

            if signal == "SKIP":
                entry_price = 0.0
                stop_loss = 0.0
                take_profit_1 = 0.0
                take_profit_2 = 0.0

            return {
                "zone_confirmed": zone_confirmed,
                "trend_confirmed": str(data.get("trend_confirmed", gann.trend)),
                "override_reason": str(data.get("override_reason", "")),
                "signal": signal,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit_1": take_profit_1,
                "take_profit_2": take_profit_2,
                "confidence_pct": confidence_pct,
                "reasoning": str(data.get("reasoning", "")),
            }, ""

        except Exception as e:
            log_error(f"Failed to parse Gemini response: {e}")
            log_warn("Falling back to system-calculated signal.")
            return {
                "zone_confirmed": gann.current_zone,
                "trend_confirmed": gann.trend,
                "override_reason": "",
                "signal": gann.signal_code,
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit_1": 0.0,
                "take_profit_2": 0.0,
                "confidence_pct": 0,
                "reasoning": "Gemini response parse failed. Using system-calculated signal.",
            }, str(e)

    def _fallback_prompt(
        self,
        symbol: str,
        timeframe: str,
        current_price: float,
        gann: GannSquareResult,
    ) -> str:
        safe_symbol = self._sanitize_prompt_text(symbol)
        safe_timeframe = self._sanitize_prompt_text(timeframe)
        return (
            f"Analyze this {safe_symbol} {safe_timeframe} chart with Gann Square zones. "
            f"Detected trend: {gann.trend}. Current price is {current_price:,.4f} in Zone {gann.current_zone} of 4. "
            f"Pre-calculated signal: {gann.signal_code}. "
            "Respond ONLY with a valid JSON object with these exact keys: "
            "zone_confirmed (int 1-4), trend_confirmed (UP or DOWN), "
            "override_reason (string, empty if no override), "
            "signal (LONG or SHORT or SKIP), entry_price (float), "
            "stop_loss (float), take_profit_1 (float), take_profit_2 (float), "
            "confidence_pct (int 0-100), reasoning (string)."
        )

    @staticmethod
    def _sanitize_prompt_text(value: str) -> str:
        text = str(value)
        text = text.replace("{", "(").replace("}", ")")
        text = " ".join(text.splitlines())
        return text.strip()

    @staticmethod
    def _safe_int(value: object, fallback: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _safe_price(value: object) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(parsed) or parsed < 0:
            return 0.0
        return parsed
