"""
HMM Signal Combiner Module

Combines High-Order HMM and HMM-KAMA models to generate trading signals.
"""

from typing import Tuple, Literal, Optional
import pandas as pd

from modules.hmm.kama import hmm_kama
from modules.hmm.high_order import hmm_high_order
from modules.hmm.signal_utils import validate_dataframe, calculate_market_volatility
from modules.hmm.signal_scoring import normalize_scores
from modules.hmm.signal_confidence import calculate_kama_confidence, calculate_combined_confidence
from modules.hmm.signal_resolution import (
    calculate_dynamic_threshold,
    resolve_signal_conflict,
    Signal,
    LONG,
    HOLD,
    SHORT,
)
from modules.common.utils import log_error, log_info

# Export Signal type for backward compatibility
__all__ = ['hmm_signals', 'Signal']

from modules.config import (
    HMM_PROBABILITY_THRESHOLD,
    HMM_SIGNAL_PRIMARY_WEIGHT,
    HMM_SIGNAL_TRANSITION_WEIGHT,
    HMM_SIGNAL_ARM_WEIGHT,
    HMM_SIGNAL_MIN_THRESHOLD,
    HMM_HIGH_ORDER_MAX_SCORE,
    HMM_FEATURES,
    HMM_HIGH_ORDER_STRENGTH,
    HMM_STATE_STRENGTH,
)


def hmm_signals(
    df: pd.DataFrame,
    # HMM-KAMA parameters
    window_kama: Optional[int] = None,
    fast_kama: Optional[int] = None,
    slow_kama: Optional[int] = None,
    window_size: Optional[int] = None,
    # HMM High Order parameters
    orders_argrelextrema: Optional[int] = None,
    strict_mode: Optional[bool] = None,
) -> Tuple[Signal, Signal]:
    """
    Generate HMM trading signals from high-order HMM and HMM-KAMA models.

    Args:
        df: DataFrame containing OHLCV data with columns: open, high, low, close
        window_kama: KAMA window size (default: from config)
        fast_kama: Fast KAMA parameter (default: from config)
        slow_kama: Slow KAMA parameter (default: from config)
        window_size: Rolling window size (default: from config)
        orders_argrelextrema: Order for swing detection (default: from config)
        strict_mode: Use strict mode for swing-to-state conversion (default: from config)

    Returns:
        Tuple of (high_order_hmm_signal, hmm_kama_signal)
        Each signal is one of: LONG (1), HOLD (0), SHORT (-1)
    """
    # Input validation
    if not validate_dataframe(df):
        return HOLD, HOLD

    # Run HMM models
    try:
        hmm_kama_result = hmm_kama(
            df,
            window_kama=window_kama,
            fast_kama=fast_kama,
            slow_kama=slow_kama,
            window_size=window_size,
        )
        high_order_hmm_result = hmm_high_order(
            df,
            eval_mode=True,
            orders_argrelextrema=orders_argrelextrema,
            strict_mode=strict_mode,
        )
    except (ValueError, AttributeError, KeyError) as e:
        log_error(f"Data validation error in HMM model: {type(e).__name__}: {str(e)}")
        return HOLD, HOLD
    except Exception as e:
        log_error(f"Unexpected error in HMM model: {type(e).__name__}: {str(e)}")
        return HOLD, HOLD

    # High-Order HMM signal evaluation with enhanced scoring
    next_state: int = high_order_hmm_result.next_state_with_high_order_hmm
    probability: float = high_order_hmm_result.next_state_probability

    # Calculate High-Order HMM score with strength multipliers
    # Score = probability * state_strength_multiplier (as per proposal)
    high_order_score_long = 0.0
    high_order_score_short = 0.0
    
    if HMM_FEATURES["high_order_scoring_enabled"] and probability >= HMM_PROBABILITY_THRESHOLD:
        if next_state == -1:  # BEARISH
            high_order_score_short = probability * HMM_HIGH_ORDER_STRENGTH["bearish"]
        elif next_state == 1:  # BULLISH
            high_order_score_long = probability * HMM_HIGH_ORDER_STRENGTH["bullish"]
        # next_state == 0 (NEUTRAL): không cộng score

    signal_high_order_hmm: Signal = HOLD
    if probability >= HMM_PROBABILITY_THRESHOLD:
        if next_state == -1:
            signal_high_order_hmm = SHORT
        elif next_state == 1:
            signal_high_order_hmm = LONG

    # HMM-KAMA scoring system
    score_long: float = 0.0
    score_short: float = 0.0

    # Calculate base confidence for confidence-weighted scoring
    base_confidence = 1.0
    if HMM_FEATURES["confidence_enabled"]:
        # Use High-Order probability as base confidence indicator
        base_confidence = max(probability, 0.5)  # Minimum 0.5

    # Primary signal scoring with state strength multipliers
    primary_state = hmm_kama_result.next_state_with_hmm_kama
    
    # Apply state strength multipliers
    state_strength = 1.0
    if HMM_FEATURES["state_strength_enabled"]:
        if primary_state in {0, 3}:  # Strong states
            state_strength = HMM_STATE_STRENGTH["strong"]
        elif primary_state in {1, 2}:  # Weak states
            state_strength = HMM_STATE_STRENGTH["weak"]
    
    if primary_state in {1, 3}:  # Bullish states
        base_weight = HMM_SIGNAL_PRIMARY_WEIGHT * state_strength
        weight = base_weight * base_confidence if HMM_FEATURES["confidence_enabled"] else base_weight
        score_long += weight
    elif primary_state in {0, 2}:  # Bearish states
        base_weight = HMM_SIGNAL_PRIMARY_WEIGHT * state_strength
        weight = base_weight * base_confidence if HMM_FEATURES["confidence_enabled"] else base_weight
        score_short += weight

    # Transition state indicators (weight: HMM_SIGNAL_TRANSITION_WEIGHT each)
    # States: -1 (bearish), 0 (neutral), 1 (bullish)
    transition_states: list[int] = [
        hmm_kama_result.current_state_of_state_using_std,
        hmm_kama_result.current_state_of_state_using_hmm,
        hmm_kama_result.current_state_of_state_using_kmeans,
    ]
    for state in transition_states:
        weight = HMM_SIGNAL_TRANSITION_WEIGHT * base_confidence if HMM_FEATURES["confidence_enabled"] else HMM_SIGNAL_TRANSITION_WEIGHT
        if state == 1:  # Bullish transition
            score_long += weight
        elif state == -1:  # Bearish transition
            score_short += weight
        # state == 0: Neutral, no bonus

    # ARM-based state scoring (weight: HMM_SIGNAL_ARM_WEIGHT each)
    arm_states: list[int] = [
        hmm_kama_result.state_high_probabilities_using_arm_apriori,
        hmm_kama_result.state_high_probabilities_using_arm_fpgrowth,
    ]
    for state in arm_states:
        weight = HMM_SIGNAL_ARM_WEIGHT * base_confidence if HMM_FEATURES["confidence_enabled"] else HMM_SIGNAL_ARM_WEIGHT
        if state in {1, 3}:  # Bullish states
            score_long += weight
        elif state in {0, 2}:  # Bearish states
            score_short += weight

    # Add High-Order HMM scores to total
    score_long += high_order_score_long
    score_short += high_order_score_short

    # Normalize scores
    normalized_long, normalized_short = normalize_scores(
        score_long, score_short, max(high_order_score_long, high_order_score_short)
    )

    # Calculate KAMA confidence
    kama_confidence = calculate_kama_confidence(score_long, score_short)

    # Check signal agreement
    signal_agreement = (
        signal_high_order_hmm == LONG and normalized_long > normalized_short
    ) or (
        signal_high_order_hmm == SHORT and normalized_short > normalized_long
    ) or (
        signal_high_order_hmm == HOLD
    )

    # Calculate combined confidence
    combined_confidence = calculate_combined_confidence(
        probability, kama_confidence, signal_agreement
    )

    # Calculate market volatility for dynamic threshold
    volatility = calculate_market_volatility(df)
    
    # Calculate base threshold
    if HMM_FEATURES["normalization_enabled"]:
        max_possible = (
            HMM_SIGNAL_PRIMARY_WEIGHT +
            (HMM_SIGNAL_TRANSITION_WEIGHT * 3) +
            (HMM_SIGNAL_ARM_WEIGHT * 2) +
            HMM_HIGH_ORDER_MAX_SCORE
        )
        base_threshold = (HMM_SIGNAL_MIN_THRESHOLD / max_possible) * 100
    else:
        base_threshold = HMM_SIGNAL_MIN_THRESHOLD
    
    # Apply dynamic threshold adjustment
    adjusted_threshold = calculate_dynamic_threshold(base_threshold, volatility)

    # Signal decision with adjusted threshold (use normalized scores if enabled)
    final_score_long = normalized_long if HMM_FEATURES["normalization_enabled"] else score_long
    final_score_short = normalized_short if HMM_FEATURES["normalization_enabled"] else score_short

    signal_hmm_kama: Signal = HOLD
    if final_score_long >= adjusted_threshold and final_score_long > final_score_short:
        signal_hmm_kama = LONG
    elif final_score_short >= adjusted_threshold and final_score_short > final_score_long:
        signal_hmm_kama = SHORT

    # Conflict resolution
    original_signal_high_order = signal_high_order_hmm
    original_signal_kama = signal_hmm_kama
    signal_high_order_hmm, signal_hmm_kama = resolve_signal_conflict(
        signal_high_order_hmm,
        signal_hmm_kama,
        probability,
        kama_confidence,
    )

    # Log active signals with detailed information including confidence
    if signal_high_order_hmm != HOLD or signal_hmm_kama != HOLD:
        signal_map: dict[Signal, str] = {LONG: "LONG", HOLD: "HOLD", SHORT: "SHORT"}
        confidence_info = ""
        if HMM_FEATURES["combined_confidence_enabled"]:
            confidence_info = f", Combined Confidence: {combined_confidence:.3f}"
        
        score_display = (
            f"(normalized L:{normalized_long:.1f}/S:{normalized_short:.1f})"
            if HMM_FEATURES["normalization_enabled"]
            else f"(raw L:{score_long:.1f}/S:{score_short:.1f})"
        )
        
        threshold_info = ""
        if HMM_FEATURES["dynamic_threshold_enabled"]:
            if adjusted_threshold != base_threshold:
                threshold_info = f", Threshold: {adjusted_threshold:.2f} (adj from {base_threshold:.2f})"
            else:
                threshold_info = f", Threshold: {adjusted_threshold:.2f}"
        
        volatility_info = ""
        if HMM_FEATURES["dynamic_threshold_enabled"]:
            volatility_info = f", Volatility: {volatility:.4f}"
        
        conflict_info = ""
        if HMM_FEATURES["conflict_resolution_enabled"]:
            if (original_signal_high_order != signal_high_order_hmm or 
                original_signal_kama != signal_hmm_kama):
                conflict_info = " [CONFLICT RESOLVED]"
        
        log_info(
            f"HMM Signals{conflict_info} - High Order: {signal_map[signal_high_order_hmm]} "
            f"(state: {next_state}, prob: {probability:.3f}), "
            f"KAMA: {signal_map[signal_hmm_kama]} "
            f"(primary: {primary_state}, {score_display}, "
            f"KAMA conf: {kama_confidence:.3f}{confidence_info}{threshold_info}{volatility_info})"
        )

    return signal_high_order_hmm, signal_hmm_kama
