"""
HMM Signal Resolution Module

Functions for conflict resolution and dynamic threshold adjustment.
"""

from typing import Tuple, Literal
from modules.common.utils import log_warn
from modules.config import (
    HMM_FEATURES,
    HMM_CONFLICT_RESOLUTION_THRESHOLD,
    HMM_VOLATILITY_CONFIG,
)

# Signal constants (moved from config for better organization)
Signal = Literal[-1, 0, 1]
LONG: Signal = 1
HOLD: Signal = 0
SHORT: Signal = -1

# Export for use in other modules
__all__ = ['Signal', 'LONG', 'HOLD', 'SHORT', 'calculate_dynamic_threshold', 'resolve_signal_conflict']


def calculate_dynamic_threshold(
    base_threshold: float, volatility: float
) -> float:
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

