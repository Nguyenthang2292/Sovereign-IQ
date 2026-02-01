"""
Signal Selector Module

Responsible for aggregating signals from diverse sources (ATC, XGBoost, Gemini),
resolving conflicts, and selecting the optimal trade setup.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from modules.auto_trade.core.atc_scanner import SignalResult
from modules.auto_trade.core.gemini_integration import GeminiSignal
from modules.common.ui.logging import log_info, log_warn


@dataclass
class FinalSignal:
    """The authoritative signal for trade execution."""

    symbol: str
    signal_type: str  # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int = 2
    confidence: float = 0.0
    sources: Dict[str, Any] = field(default_factory=dict)  # Metadata from sources
    timestamp: float = field(default_factory=time.time)


class SignalSelector:
    """Aggregates and selects the best trading signal."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        # Weights for scoring (total should be ~1.0 generally, but used relatively here)
        self.weight_xgboost = self.config.get("weight_xgboost", 0.4)
        self.weight_gemini = self.config.get("weight_gemini", 0.6)
        self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.7)

    def select_best_signal(
        self, xgboost_signals: List[SignalResult], gemini_signals: Dict[str, GeminiSignal]
    ) -> Optional[FinalSignal]:
        """
        Evaluate all candidates and return the single best signal.

        Args:
            xgboost_signals: Results from the XGBoost filter step.
            gemini_signals: Results from the Gemini analysis step (keyed by symbol).

        Returns:
            The FinalSignal to execute, or None if no suitable signal found.
        """
        candidates: List[FinalSignal] = []

        for signal in xgboost_signals:
            symbol = signal.symbol
            gemini_data = gemini_signals.get(symbol)

            # If no Gemini data, we might skip or rely solely on XGBoost based on policy.
            # Current policy: Require at least XGBoost confirmation, boost with Gemini.

            final_signal = self._evaluate_candidate(signal, gemini_data)
            if final_signal and final_signal.confidence >= self.min_confidence_threshold:
                candidates.append(final_signal)

        if not candidates:
            log_info("Signal Selector: No candidates met the criteria.")
            return None

        # Sort by confidence descending
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        best_signal = candidates[0]

        log_info(
            f"Signal Selector: Selected {best_signal.symbol} ({best_signal.signal_type}) "
            f"with confidence {best_signal.confidence:.2f}"
        )

        return best_signal

    def _evaluate_candidate(
        self, xb_signal: SignalResult, gemini_signal: Optional[GeminiSignal]
    ) -> Optional[FinalSignal]:
        """Combine signal sources into a FinalSignal candidate."""

        # Base confidence from XGBoost (parsed from details if available, or default)
        try:
            xb_conf = float(xb_signal.details.get("xgboost_conf", 0.0))
        except (ValueError, TypeError):
            xb_conf = 0.0

        # Gemini confidence
        gemini_conf = 0.0
        gemini_dir = "NONE"

        entry = 0.0
        tp = 0.0
        sl = 0.0

        if gemini_signal:
            gemini_conf = gemini_signal.confidence
            gemini_dir = gemini_signal.signal

            # Prefer Gemini's precise levels if available
            entry = gemini_signal.entry if gemini_signal.entry else 0.0
            tp = gemini_signal.take_profit if gemini_signal.take_profit else 0.0
            sl = gemini_signal.stop_loss if gemini_signal.stop_loss else 0.0

        # Conflict Check
        # If Gemini explicitly contradicts XGBoost (e.g., LONG vs SHORT), penalize or discard.
        if gemini_signal and gemini_dir != "NONE":
            if xb_signal.signal_type != gemini_dir:
                log_warn(
                    f"Signal Conflict for {xb_signal.symbol}: "
                    f"XGBoost={xb_signal.signal_type}, Gemini={gemini_dir}. Discarding."
                )
                return None

        # Calculate Final Confidence
        # If Gemini is present, we normalize the combined score.
        # If Gemini is missing, we strictly use XGBoost score.
        if gemini_signal:
            final_conf = (xb_conf * self.weight_xgboost) + (gemini_conf * self.weight_gemini)
            # Normalize to 0-1 range roughly, ensuring we don't exceed 1.0
            final_conf = min(1.0, final_conf)
        else:
            final_conf = xb_conf  # Fallback to just XGBoost confidence

        # Safety check for Price Levels
        # If Gemini didn't provide levels (or wasn't run), we need a fallback calculator (not implemented here yet).
        # For Phase 2.5, we will return 0.0 levels if Gemini is missing,
        # relying on Execution Phase (Phase 3) to calculate them if they are 0.

        return FinalSignal(
            symbol=xb_signal.symbol,
            signal_type=xb_signal.signal_type,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            confidence=final_conf,
            sources={
                "xgboost_score": xb_conf,
                "gemini_score": gemini_conf,
                "gemini_reasoning": gemini_signal.reasoning if gemini_signal else "N/A",
            },
        )
