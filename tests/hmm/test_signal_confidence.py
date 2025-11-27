"""
Test script for modules.hmm.signal_confidence - Confidence calculations.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import patch

from modules.hmm.signal_confidence import (
    calculate_kama_confidence,
    calculate_combined_confidence,
)
from modules.config import (
    HMM_SIGNAL_MIN_THRESHOLD,
    HMM_FEATURES,
    HMM_HIGH_ORDER_WEIGHT,
    HMM_KAMA_WEIGHT,
    HMM_AGREEMENT_BONUS,
)


def test_calculate_kama_confidence_zero_scores():
    """Test calculate_kama_confidence with zero scores."""
    confidence = calculate_kama_confidence(0.0, 0.0)
    assert confidence == 0.5  # Neutral confidence


def test_calculate_kama_confidence_long_dominant():
    """Test calculate_kama_confidence with long score dominant."""
    confidence = calculate_kama_confidence(10.0, 2.0)
    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.5  # Long is dominant


def test_calculate_kama_confidence_short_dominant():
    """Test calculate_kama_confidence with short score dominant."""
    confidence = calculate_kama_confidence(2.0, 10.0)
    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.5  # Short is dominant


def test_calculate_kama_confidence_equal_scores():
    """Test calculate_kama_confidence with equal scores."""
    confidence = calculate_kama_confidence(5.0, 5.0)
    assert confidence == 0.5  # Equal scores = 0.5 confidence


def test_calculate_kama_confidence_well_separated():
    """Test calculate_kama_confidence with well-separated scores."""
    score_diff = HMM_SIGNAL_MIN_THRESHOLD + 1.0
    confidence = calculate_kama_confidence(score_diff + 5.0, 5.0)
    assert confidence > 0.5
    assert confidence <= 1.0


def test_calculate_kama_confidence_boost_applied():
    """Test that confidence boost is applied for well-separated scores."""
    score_long = HMM_SIGNAL_MIN_THRESHOLD + 10.0
    score_short = 1.0
    
    confidence = calculate_kama_confidence(score_long, score_short)
    
    # Without boost: confidence would be score_long / (score_long + score_short)
    # With boost: should be higher
    base_confidence = score_long / (score_long + score_short)
    assert confidence >= base_confidence


def test_calculate_combined_confidence_disabled():
    """Test calculate_combined_confidence when disabled."""
    with patch('modules.hmm.signal_confidence.HMM_FEATURES', {"combined_confidence_enabled": False}):
        high_order_prob = 0.8
        kama_confidence = 0.6
        signal_agreement = True
        
        combined = calculate_combined_confidence(
            high_order_prob, kama_confidence, signal_agreement
        )
        
        # Should be simple average
        expected = (high_order_prob + kama_confidence) / 2
        assert abs(combined - expected) < 0.01


def test_calculate_combined_confidence_enabled():
    """Test calculate_combined_confidence when enabled."""
    if not HMM_FEATURES["combined_confidence_enabled"]:
        pytest.skip("Combined confidence is disabled in config")
    
    high_order_prob = 0.8
    kama_confidence = 0.6
    signal_agreement = False
    
    combined = calculate_combined_confidence(
        high_order_prob, kama_confidence, signal_agreement
        )
    
    assert 0.0 <= combined <= 1.0
    # Should be weighted average
    expected_base = (
        high_order_prob * HMM_HIGH_ORDER_WEIGHT +
        kama_confidence * HMM_KAMA_WEIGHT
    )
    assert abs(combined - expected_base) < 0.01


def test_calculate_combined_confidence_with_agreement():
    """Test calculate_combined_confidence with signal agreement."""
    if not HMM_FEATURES["combined_confidence_enabled"]:
        pytest.skip("Combined confidence is disabled in config")
    
    high_order_prob = 0.8
    kama_confidence = 0.6
    signal_agreement = True
    
    combined_with_agreement = calculate_combined_confidence(
        high_order_prob, kama_confidence, signal_agreement
    )
    
    combined_without_agreement = calculate_combined_confidence(
        high_order_prob, kama_confidence, False
    )
    
    # With agreement should be higher (bonus applied)
    assert combined_with_agreement >= combined_without_agreement


def test_calculate_combined_confidence_capped_at_one():
    """Test that combined confidence is capped at 1.0."""
    if not HMM_FEATURES["combined_confidence_enabled"]:
        pytest.skip("Combined confidence is disabled in config")
    
    high_order_prob = 1.0
    kama_confidence = 1.0
    signal_agreement = True
    
    combined = calculate_combined_confidence(
        high_order_prob, kama_confidence, signal_agreement
    )
    
    assert combined <= 1.0


def test_calculate_combined_confidence_edge_cases():
    """Test calculate_combined_confidence with edge case values."""
    if not HMM_FEATURES["combined_confidence_enabled"]:
        pytest.skip("Combined confidence is disabled in config")
    
    # Test with zero probabilities
    combined = calculate_combined_confidence(0.0, 0.0, False)
    assert 0.0 <= combined <= 1.0
    
    # Test with maximum probabilities
    combined = calculate_combined_confidence(1.0, 1.0, True)
    assert combined <= 1.0

