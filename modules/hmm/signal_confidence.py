"""
HMM Signal Confidence Module

Functions for calculating confidence scores from HMM models.
"""

from modules.config import (
    HMM_SIGNAL_MIN_THRESHOLD,
    HMM_FEATURES,
    HMM_HIGH_ORDER_WEIGHT,
    HMM_KAMA_WEIGHT,
    HMM_AGREEMENT_BONUS,
)


def calculate_kama_confidence(
    score_long: float, score_short: float
) -> float:
    """
    Calculate confidence score from KAMA model based on score distribution.
    
    Args:
        score_long: Long signal score
        score_short: Short signal score
        
    Returns:
        Confidence value between 0.0 and 1.0
    """
    total_score = score_long + score_short
    if total_score == 0:
        return 0.5  # Neutral confidence
    
    # Confidence = ratio of dominant signal
    max_score = max(score_long, score_short)
    confidence = max_score / total_score
    
    # Boost confidence if scores are well-separated
    score_diff = abs(score_long - score_short)
    if score_diff > HMM_SIGNAL_MIN_THRESHOLD:
        confidence = min(confidence * 1.1, 1.0)
    
    return confidence


def calculate_combined_confidence(
    high_order_prob: float,
    kama_confidence: float,
    signal_agreement: bool,
) -> float:
    """
    Calculate combined confidence from both HMM models.
    
    Args:
        high_order_prob: Probability from High-Order HMM
        kama_confidence: Confidence from KAMA model
        signal_agreement: Whether both models agree on direction
        
    Returns:
        Combined confidence value between 0.0 and 1.0
    """
    if not HMM_FEATURES["combined_confidence_enabled"]:
        return (high_order_prob + kama_confidence) / 2
    
    # Weighted average of both models
    base_confidence = (
        high_order_prob * HMM_HIGH_ORDER_WEIGHT +
        kama_confidence * HMM_KAMA_WEIGHT
    )
    
    # Agreement bonus: if signals agree, increase confidence
    if signal_agreement:
        base_confidence *= HMM_AGREEMENT_BONUS
    
    return min(base_confidence, 1.0)  # Cap at 1.0

