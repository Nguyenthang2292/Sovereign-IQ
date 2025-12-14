"""
Test script for modules.hmm.signal_resolution - Conflict resolution and threshold adjustment.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import patch

from modules.hmm.signals.resolution import (
    calculate_dynamic_threshold,
    resolve_signal_conflict,
    Signal,
    LONG,
    HOLD,
    SHORT,
)
from config import (
    HMM_FEATURES,
    HMM_CONFLICT_RESOLUTION_THRESHOLD,
    HMM_VOLATILITY_CONFIG,
)


def test_calculate_dynamic_threshold_disabled():
    """Test calculate_dynamic_threshold when disabled."""
    with patch('modules.hmm.signals.resolution.HMM_FEATURES', {"dynamic_threshold_enabled": False}):
        base_threshold = 10.0
        volatility = 0.05  # High volatility
        
        adjusted = calculate_dynamic_threshold(base_threshold, volatility)
        
        assert adjusted == base_threshold


def test_calculate_dynamic_threshold_high_volatility():
    """Test calculate_dynamic_threshold with high volatility."""
    if not HMM_FEATURES["dynamic_threshold_enabled"]:
        pytest.skip("Dynamic threshold is disabled in config")
    
    base_threshold = 10.0
    high_threshold = HMM_VOLATILITY_CONFIG["high_threshold"]
    high_volatility = high_threshold + 0.01
    
    adjusted = calculate_dynamic_threshold(base_threshold, high_volatility)
    
    # Should be increased (more conservative)
    assert adjusted > base_threshold
    expected = base_threshold * HMM_VOLATILITY_CONFIG["adjustments"]["high"]
    assert abs(adjusted - expected) < 0.01


def test_calculate_dynamic_threshold_low_volatility():
    """Test calculate_dynamic_threshold with low volatility."""
    if not HMM_FEATURES["dynamic_threshold_enabled"]:
        pytest.skip("Dynamic threshold is disabled in config")
    
    base_threshold = 10.0
    high_threshold = HMM_VOLATILITY_CONFIG["high_threshold"]
    low_volatility = high_threshold * 0.4  # Below 0.5 threshold
    
    adjusted = calculate_dynamic_threshold(base_threshold, low_volatility)
    
    # Should be decreased (more aggressive)
    assert adjusted < base_threshold
    expected = base_threshold * HMM_VOLATILITY_CONFIG["adjustments"]["low"]
    assert abs(adjusted - expected) < 0.01


def test_calculate_dynamic_threshold_normal_volatility():
    """Test calculate_dynamic_threshold with normal volatility."""
    if not HMM_FEATURES["dynamic_threshold_enabled"]:
        pytest.skip("Dynamic threshold is disabled in config")
    
    base_threshold = 10.0
    high_threshold = HMM_VOLATILITY_CONFIG["high_threshold"]
    normal_volatility = high_threshold * 0.75  # Between thresholds
    
    adjusted = calculate_dynamic_threshold(base_threshold, normal_volatility)
    
    # Should remain unchanged
    assert adjusted == base_threshold


def test_resolve_signal_conflict_disabled():
    """Test resolve_signal_conflict when disabled."""
    with patch('modules.hmm.signals.resolution.HMM_FEATURES', {"conflict_resolution_enabled": False}):
        signal_high_order = LONG
        signal_kama = SHORT
        high_order_prob = 0.8
        kama_confidence = 0.6
        
        resolved_high, resolved_kama = resolve_signal_conflict(
            signal_high_order, signal_kama, high_order_prob, kama_confidence
        )
        
        assert resolved_high == signal_high_order
        assert resolved_kama == signal_kama


def test_resolve_signal_conflict_no_conflict_agreement():
    """Test resolve_signal_conflict when signals agree."""
    if not HMM_FEATURES["conflict_resolution_enabled"]:
        pytest.skip("Conflict resolution is disabled in config")
    
    signal_high_order = LONG
    signal_kama = LONG
    high_order_prob = 0.8
    kama_confidence = 0.6
    
    resolved_high, resolved_kama = resolve_signal_conflict(
        signal_high_order, signal_kama, high_order_prob, kama_confidence
    )
    
    assert resolved_high == LONG
    assert resolved_kama == LONG


def test_resolve_signal_conflict_no_conflict_hold():
    """Test resolve_signal_conflict when one signal is HOLD."""
    if not HMM_FEATURES["conflict_resolution_enabled"]:
        pytest.skip("Conflict resolution is disabled in config")
    
    signal_high_order = LONG
    signal_kama = HOLD
    high_order_prob = 0.8
    kama_confidence = 0.6
    
    resolved_high, resolved_kama = resolve_signal_conflict(
        signal_high_order, signal_kama, high_order_prob, kama_confidence
    )
    
    assert resolved_high == LONG
    assert resolved_kama == HOLD


def test_resolve_signal_conflict_high_order_wins():
    """Test resolve_signal_conflict when High-Order has higher confidence."""
    if not HMM_FEATURES["conflict_resolution_enabled"]:
        pytest.skip("Conflict resolution is disabled in config")
    
    signal_high_order = LONG
    signal_kama = SHORT
    high_order_prob = 0.9  # High confidence
    kama_confidence = 0.5  # Lower confidence
    
    # High-Order confidence should be > kama_confidence * threshold
    threshold = HMM_CONFLICT_RESOLUTION_THRESHOLD
    if high_order_prob > kama_confidence * threshold:
        resolved_high, resolved_kama = resolve_signal_conflict(
            signal_high_order, signal_kama, high_order_prob, kama_confidence
        )
        
        assert resolved_high == LONG
        assert resolved_kama == HOLD  # KAMA downgraded


def test_resolve_signal_conflict_kama_wins():
    """Test resolve_signal_conflict when KAMA has higher confidence."""
    if not HMM_FEATURES["conflict_resolution_enabled"]:
        pytest.skip("Conflict resolution is disabled in config")
    
    signal_high_order = LONG
    signal_kama = SHORT
    high_order_prob = 0.5  # Lower confidence
    kama_confidence = 0.9  # High confidence
    
    # KAMA confidence should be > high_order_prob * threshold
    threshold = HMM_CONFLICT_RESOLUTION_THRESHOLD
    if kama_confidence > high_order_prob * threshold:
        resolved_high, resolved_kama = resolve_signal_conflict(
            signal_high_order, signal_kama, high_order_prob, kama_confidence
        )
        
        assert resolved_high == HOLD  # High-Order downgraded
        assert resolved_kama == SHORT


def test_resolve_signal_conflict_similar_confidence():
    """Test resolve_signal_conflict when confidences are similar."""
    if not HMM_FEATURES["conflict_resolution_enabled"]:
        pytest.skip("Conflict resolution is disabled in config")
    
    signal_high_order = LONG
    signal_kama = SHORT
    high_order_prob = 0.7
    kama_confidence = 0.65  # Similar confidence
    
    # Neither should be significantly higher
    threshold = HMM_CONFLICT_RESOLUTION_THRESHOLD
    if not (high_order_prob > kama_confidence * threshold or 
            kama_confidence > high_order_prob * threshold):
        resolved_high, resolved_kama = resolve_signal_conflict(
            signal_high_order, signal_kama, high_order_prob, kama_confidence
        )
        
        # When confidence is similar and conflict unresolved, both should default to HOLD (safety first)
        assert resolved_high == HOLD
        assert resolved_kama == HOLD

