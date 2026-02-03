"""
Signal Selector Module

Responsible for aggregating signals from diverse sources (ATC, XGBoost, Gemini),
resolving conflicts, and selecting the optimal trade setup.

Usage:
    selector = SignalSelector(config={"weight_xgboost": 0.4, "weight_gemini": 0.6})

    # 1. Prepare signals
    xb_signals = [SignalResult("BTC/USDT", 0.9, "LONG", {"xgboost_conf": "0.8"})]
    gemini_signals = {
        "BTC/USDT": GeminiSignal(
            trend="UP", signal="LONG", confidence=0.85,
            entry=50000, stop_loss=49000, take_profit=52000
        )
    }

    # 2. Select best signal
    final_signal = selector.select_best_signal(xb_signals, gemini_signals)

    if final_signal:
        print(f"Trade {final_signal.symbol} {final_signal.signal_type}")
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TypedDict

# Use backward-compatible import from root config package
from config import SIGNAL_SELECTOR_DEFAULTS
from modules.auto_trade.core.atc_scanner import SignalResult
from modules.auto_trade.core.gemini_integration import GeminiSignal
from modules.common.ui.logging import log_info, log_warn


class SignalSources(TypedDict, total=False):
    xgboost_score: float
    gemini_score: float
    gemini_reasoning: str


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
    score: float = 0.0  # Quality score 0-100
    sources: SignalSources = field(default_factory=dict)  # Metadata from sources
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Validate signal integrity."""
        # 1. Leverage Validation
        if not (1 <= self.leverage <= 10):
            raise ValueError(f"Invalid leverage: {self.leverage}. Must be 1-10")

        # 2. Price Level Validation
        if self.entry_price <= 0 or self.stop_loss <= 0 or self.take_profit <= 0:
            raise ValueError(
                f"Invalid zero/negative price levels for {self.symbol}: "
                f"Entry={self.entry_price}, SL={self.stop_loss}, TP={self.take_profit}"
            )

        # 3. Price Relationship Validation
        if self.signal_type == "LONG":
            if not (self.stop_loss < self.entry_price < self.take_profit):
                raise ValueError(
                    f"Invalid LONG price structure for {self.symbol}: "
                    f"SL({self.stop_loss}) < Entry({self.entry_price}) < TP({self.take_profit}) is false"
                )
        elif self.signal_type == "SHORT":
            if not (self.take_profit < self.entry_price < self.stop_loss):
                raise ValueError(
                    f"Invalid SHORT price structure for {self.symbol}: "
                    f"TP({self.take_profit}) < Entry({self.entry_price}) < SL({self.stop_loss}) is false"
                )


class SignalSelector:
    """Aggregates and selects the best trading signal."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        # Weights for scoring (total should be ~1.0 generally, but used relatively here)
        self.weight_xgboost = self.config.get("weight_xgboost", SIGNAL_SELECTOR_DEFAULTS["weight_xgboost"])
        self.weight_gemini = self.config.get("weight_gemini", SIGNAL_SELECTOR_DEFAULTS["weight_gemini"])
        self.min_confidence_threshold = self.config.get(
            "min_confidence_threshold", SIGNAL_SELECTOR_DEFAULTS["min_confidence_threshold"]
        )
        # Default policy: Signals must have valid price levels (usually from Gemini)
        self.require_gemini_levels = self.config.get("require_gemini_levels", True)

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

        # Calculate Risk/Reward for logging
        rr_ratio = self._calculate_risk_reward_ratio(best_signal)

        log_info(
            f"Signal Selector: Selected {best_signal.symbol} ({best_signal.signal_type}) "
            f"with confidence {best_signal.confidence:.2f}, R/R: {rr_ratio:.2f}"
        )

        return best_signal

    def _evaluate_candidate(
        self, xb_signal: SignalResult, gemini_signal: Optional[GeminiSignal]
    ) -> Optional[FinalSignal]:
        """Combine signal sources into a FinalSignal candidate."""

        # Base confidence from XGBoost (now stored as float, not string)
        xb_conf = xb_signal.details.get("xgboost_conf", 0.0)
        if not isinstance(xb_conf, (int, float)):
            # Handle legacy string format if present
            try:
                xb_conf = float(xb_conf)
            except (ValueError, TypeError) as e:
                log_warn(
                    f"Failed to parse XGBoost confidence for {xb_signal.symbol}: {e}. Details: {xb_signal.details}"
                )
                xb_conf = 0.0
        else:
            xb_conf = float(xb_conf)

        # Gemini confidence
        gemini_conf = 0.0
        gemini_dir = "NONE"

        entry = 0.0
        tp = 0.0
        sl = 0.0
        gemini_reasoning = "N/A"

        if gemini_signal:
            gemini_conf = gemini_signal.confidence
            gemini_dir = gemini_signal.signal
            gemini_reasoning = gemini_signal.reasoning

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
        # Ensure weights are normalized
        total_weight = self.weight_xgboost + self.weight_gemini

        if gemini_signal:
            final_conf = ((xb_conf * self.weight_xgboost) + (gemini_conf * self.weight_gemini)) / total_weight
            # Normalize to 0-1 range roughly, ensuring we don't exceed 1.0 (though total_weight helps)
            final_conf = min(1.0, final_conf)
        else:
            final_conf = min(1.0, xb_conf)  # Fallback to just XGBoost confidence, normalized

        # Safety check for Price Levels
        # Strict Validation Phase: If levels are 0, we cannot create a valid FinalSignal.
        if entry == 0.0 or tp == 0.0 or sl == 0.0:
            log_warn("Gemini analysis required for accurate levels.")
            # If we allow missing Gemini levels (via config), we could theoretically calculate fallbacks here.
            # But currently, 'require_gemini_levels' defaults to True, enforcing this check.
            return None

        try:
            # Calculate Score (0-100)
            # Base it on confidence (60%), R/R ratio (20%), and source consistency (20%)
            # This is a preliminary scoring model.

            # 1. Confidence Component (0-60)
            score = final_conf * 60

            # 2. Risk/Reward Component (0-20)
            # Cap RR at 3.0 for max score
            rr_ratio = 0.0
            if sl > 0 and entry > 0 and tp > 0:
                if xb_signal.signal_type == "LONG":
                    risk = abs(entry - sl)
                    reward = abs(tp - entry)
                else:
                    risk = abs(sl - entry)
                    reward = abs(entry - tp)

                if risk > 0:
                    rr_ratio = reward / risk

            rr_score = min(rr_ratio, 3.0) / 3.0 * 20
            score += rr_score

            # 3. Consistency Component (0-20)
            # If both XGBoost and Gemini agree (which they do if we are here), add consistency points
            # If Gemini is missing, we only get partial points
            if gemini_signal:
                score += 20
            else:
                score += 5  # Small bonus for just XGBoost existing

            return FinalSignal(
                symbol=xb_signal.symbol,
                signal_type=xb_signal.signal_type,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                confidence=final_conf,
                score=score,
                sources={
                    "xgboost_score": xb_conf,
                    "gemini_score": gemini_conf,
                    "gemini_reasoning": gemini_reasoning,
                },
            )
        except ValueError as e:
            log_warn(f"Validation failed for {xb_signal.symbol}: {e}")
            return None

    def _calculate_risk_reward_ratio(self, signal: FinalSignal) -> float:
        """Calculate risk/reward ratio for the signal."""
        if signal.signal_type == "LONG":
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price)
        else:  # SHORT
            risk = abs(signal.stop_loss - signal.entry_price)
            reward = abs(signal.entry_price - signal.take_profit)

        return reward / risk if risk > 0 else 0.0
