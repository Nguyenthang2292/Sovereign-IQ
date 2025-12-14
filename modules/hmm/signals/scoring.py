"""
HMM Signal Scoring Module

Functions for score normalization and calculation.
Supports both legacy hardcoded scoring and new strategy-based scoring.
"""

from typing import Tuple, List, Dict
from modules.hmm.signals.strategy import HMMStrategy, HMMStrategyResult
from modules.hmm.signals.resolution import LONG, HOLD, SHORT
from config import (
    HMM_FEATURES,
    HMM_SIGNAL_PRIMARY_WEIGHT,
    HMM_SIGNAL_TRANSITION_WEIGHT,
    HMM_SIGNAL_ARM_WEIGHT,
    HMM_HIGH_ORDER_MAX_SCORE,
    HMM_PROBABILITY_THRESHOLD,
    HMM_HIGH_ORDER_STRENGTH,
)


def normalize_scores(
    score_long: float, score_short: float, high_order_score: float = 0.0
) -> Tuple[float, float]:
    """
    Normalize scores to 0-100 range for easier comparison.
    
    Args:
        score_long: Long signal score
        score_short: Short signal score
        high_order_score: High-Order HMM score (optional)
        
    Returns:
        Tuple of (normalized_long, normalized_short)
    """
    if not HMM_FEATURES["normalization_enabled"]:
        return score_long, score_short
    
    # Calculate maximum possible score
    max_possible_score = (
        HMM_SIGNAL_PRIMARY_WEIGHT +
        (HMM_SIGNAL_TRANSITION_WEIGHT * 3) +  # 3 transition states
        (HMM_SIGNAL_ARM_WEIGHT * 2) +  # 2 ARM states
        HMM_HIGH_ORDER_MAX_SCORE
    )
    
    if max_possible_score == 0:
        return 0.0, 0.0
    
    # Normalize to 0-100 range
    normalized_long = (score_long / max_possible_score) * 100
    normalized_short = (score_short / max_possible_score) * 100
    
    return normalized_long, normalized_short


def calculate_strategy_scores(
    strategies: List[HMMStrategy],
    results: Dict[str, HMMStrategyResult]
) -> Tuple[float, float]:
    """
    Calculate aggregated scores from multiple strategy results.
    
    Args:
        strategies: List of HMM strategies
        results: Dictionary mapping strategy names to results
        
    Returns:
        Tuple of (score_long, score_short) aggregated from all strategies
    """
    score_long = 0.0
    score_short = 0.0
    
    for strategy in strategies:
        if strategy.name not in results:
            continue
        
        result = results[strategy.name]
        
        # Only count strategies with probability above threshold
        if result.probability < HMM_PROBABILITY_THRESHOLD:
            continue
        
        # Calculate strategy contribution based on signal and probability
        strategy_weight = strategy.weight
        signal_strength = result.probability
        
        # Apply strength multipliers for high-order strategies
        if HMM_FEATURES.get("high_order_scoring_enabled", False):
            if result.signal == SHORT:
                strength_multiplier = HMM_HIGH_ORDER_STRENGTH.get("bearish", 1.0)
                score_short += signal_strength * strategy_weight * strength_multiplier
            elif result.signal == LONG:
                strength_multiplier = HMM_HIGH_ORDER_STRENGTH.get("bullish", 1.0)
                score_long += signal_strength * strategy_weight * strength_multiplier
        else:
            # Simple scoring: weight * probability
            if result.signal == SHORT:
                score_short += signal_strength * strategy_weight
            elif result.signal == LONG:
                score_long += signal_strength * strategy_weight
    
    return score_long, score_short


def normalize_strategy_scores(
    score_long: float,
    score_short: float,
    strategies: List[HMMStrategy],
    results: Dict[str, HMMStrategyResult]
) -> Tuple[float, float]:
    """
    Normalize strategy-based scores to 0-100 range.
    
    Args:
        score_long: Long signal score
        score_short: Short signal score
        strategies: List of strategies (for calculating max possible score)
        results: Dictionary mapping strategy names to results
        
    Returns:
        Tuple of (normalized_long, normalized_short)
    """
    if not HMM_FEATURES.get("normalization_enabled", True):
        return score_long, score_short
    
    # Calculate maximum possible score from all enabled strategies
    max_possible_score = 0.0
    for strategy in strategies:
        if strategy.enabled and strategy.name in results:
            max_possible_score += strategy.weight * HMM_HIGH_ORDER_MAX_SCORE
    
    if max_possible_score == 0:
        return 0.0, 0.0
    
    # Normalize to 0-100 range
    normalized_long = (score_long / max_possible_score) * 100
    normalized_short = (score_short / max_possible_score) * 100
    
    return normalized_long, normalized_short

