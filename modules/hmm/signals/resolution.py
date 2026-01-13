"""
HMM Signal Resolution Module

Functions for conflict resolution and dynamic threshold adjustment.
"""

from typing import TYPE_CHECKING, Dict, List, Literal, Tuple

from config.hmm import (
    HMM_CONFLICT_RESOLUTION_THRESHOLD,
    HMM_FEATURES,
    HMM_VOLATILITY_CONFIG,
)
from modules.common.utils import log_warn

if TYPE_CHECKING:
    from modules.hmm.signals.strategy import HMMStrategy, HMMStrategyResult

# Signal constants (moved from config for better organization)
Signal = Literal[-1, 0, 1]
LONG: Signal = 1
HOLD: Signal = 0
SHORT: Signal = -1

# Export for use in other modules
__all__ = [
    "Signal",
    "LONG",
    "HOLD",
    "SHORT",
    "calculate_dynamic_threshold",
    "resolve_signal_conflict",
    "resolve_multi_strategy_conflicts",
]


def calculate_dynamic_threshold(base_threshold: float, volatility: float) -> float:
    """
    Adjust threshold dynamically based on market volatility.

    Args:
        base_threshold: Base threshold value
        volatility: Market volatility

    Returns:
        Adjusted threshold value
    """
    if not HMM_FEATURES["dynamic_threshold_enabled"]:
        return base_threshold

    high_threshold = HMM_VOLATILITY_CONFIG["high_threshold"]
    adjustments = HMM_VOLATILITY_CONFIG["adjustments"]

    if volatility > high_threshold:
        # High volatility: increase threshold (more conservative)
        return base_threshold * adjustments["high"]
    elif volatility < high_threshold * 0.5:
        # Low volatility: decrease threshold (more aggressive)
        return base_threshold * adjustments["low"]
    else:
        # Normal volatility: use base threshold
        return base_threshold


def resolve_signal_conflict(
    signal_high_order: Signal,
    signal_kama: Signal,
    high_order_prob: float,
    kama_confidence: float,
) -> Tuple[Signal, Signal]:
    """
    Resolve conflicts when signals from different models disagree.

    Args:
        signal_high_order: Signal from High-Order HMM
        signal_kama: Signal from KAMA model
        high_order_prob: Probability from High-Order HMM
        kama_confidence: Confidence from KAMA model

    Returns:
        Tuple of (resolved_high_order_signal, resolved_kama_signal)
    """
    if not HMM_FEATURES["conflict_resolution_enabled"]:
        return signal_high_order, signal_kama

    # No conflict if signals agree or one is HOLD
    if signal_high_order == signal_kama or signal_high_order == HOLD or signal_kama == HOLD:
        return signal_high_order, signal_kama

    # Conflict detected: signals disagree
    # Compare confidence levels
    high_order_confidence = high_order_prob

    # If High-Order has significantly higher confidence, downgrade KAMA
    if high_order_confidence > kama_confidence * HMM_CONFLICT_RESOLUTION_THRESHOLD:
        # High-Order is more confident, keep it, downgrade KAMA to HOLD
        return signal_high_order, HOLD
    elif kama_confidence > high_order_confidence * HMM_CONFLICT_RESOLUTION_THRESHOLD:
        # KAMA is more confident, keep it, downgrade High-Order to HOLD
        return HOLD, signal_kama
    else:
        # Confidence levels are similar, conflict unresolved -> SAFETY FIRST: HOLD both
        log_warn(
            f"Signal conflict unresolved - High Order: {signal_high_order} "
            f"(conf: {high_order_confidence:.3f}), KAMA: {signal_kama} "
            f"(conf: {kama_confidence:.3f}). Defaulting to HOLD."
        )
        return HOLD, HOLD


def resolve_multi_strategy_conflicts(
    strategies: List["HMMStrategy"], results: Dict[str, "HMMStrategyResult"]
) -> Dict[str, Signal]:
    """
    Resolve conflicts between multiple strategy signals.

    Uses strategy weights and confidences to resolve conflicts.
    Strategies with higher confidence and weight take precedence.

    Args:
        strategies: List of HMM strategies
        results: Dictionary mapping strategy names to results

    Returns:
        Dictionary mapping strategy names to resolved signals
    """
    if not HMM_FEATURES.get("conflict_resolution_enabled", True):
        # Return original signals without resolution
        return {name: results[name].signal for name in results.keys()}

    resolved_signals = {}

    # Group strategies by signal
    signal_groups = {LONG: [], SHORT: [], HOLD: []}

    for strategy in strategies:
        if strategy.name not in results:
            continue

        result = results[strategy.name]
        signal = result.signal
        signal_groups[signal].append((strategy, result))

    # Check for conflicts (both LONG and SHORT signals exist)
    has_conflict = len(signal_groups[LONG]) > 0 and len(signal_groups[SHORT]) > 0

    if not has_conflict:
        # No conflict, return original signals
        for strategy in strategies:
            if strategy.name in results:
                resolved_signals[strategy.name] = results[strategy.name].signal
        return resolved_signals

    # Conflict detected: resolve using weighted confidence
    # Calculate total weighted confidence for each signal type
    long_weighted_conf = sum(s.weight * r.probability for s, r in signal_groups[LONG])
    short_weighted_conf = sum(s.weight * r.probability for s, r in signal_groups[SHORT])

    # Determine winning signal
    if long_weighted_conf > short_weighted_conf * HMM_CONFLICT_RESOLUTION_THRESHOLD:
        # LONG wins: keep LONG signals, downgrade SHORT to HOLD
        winning_signal = LONG
        losing_signal = SHORT
    elif short_weighted_conf > long_weighted_conf * HMM_CONFLICT_RESOLUTION_THRESHOLD:
        # SHORT wins: keep SHORT signals, downgrade LONG to HOLD
        winning_signal = SHORT
        losing_signal = LONG
    else:
        # Confidence levels are similar, conflict unresolved -> SAFETY FIRST: HOLD all
        log_warn(
            f"Multi-strategy conflict unresolved - LONG weighted conf: {long_weighted_conf:.3f}, "
            f"SHORT weighted conf: {short_weighted_conf:.3f}. Defaulting all to HOLD."
        )
        for strategy in strategies:
            if strategy.name in results:
                resolved_signals[strategy.name] = HOLD
        return resolved_signals

    # Apply resolution: keep winning signals, downgrade losing signals
    for strategy in strategies:
        if strategy.name not in results:
            continue

        result = results[strategy.name]
        original_signal = result.signal

        if original_signal == winning_signal:
            # Keep winning signal
            resolved_signals[strategy.name] = original_signal
        elif original_signal == losing_signal:
            # Downgrade losing signal to HOLD
            resolved_signals[strategy.name] = HOLD
        else:
            # HOLD signals remain HOLD
            resolved_signals[strategy.name] = HOLD

    return resolved_signals
