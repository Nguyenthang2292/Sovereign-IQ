"""
HMM Signal Scoring Module

Functions for score normalization and calculation.
"""

from typing import Tuple
from config import (
    HMM_FEATURES,
    HMM_SIGNAL_PRIMARY_WEIGHT,
    HMM_SIGNAL_TRANSITION_WEIGHT,
    HMM_SIGNAL_ARM_WEIGHT,
    HMM_HIGH_ORDER_MAX_SCORE,
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

hort

