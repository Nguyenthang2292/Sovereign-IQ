
from pathlib import Path
import sys

"""
Test script for modules.hmm.signal_scoring - Score normalization.
"""


# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unittest.mock import patch

import pytest

from config import (
    HMM_FEATURES,
    HMM_HIGH_ORDER_MAX_SCORE,
    HMM_SIGNAL_ARM_WEIGHT,
    HMM_SIGNAL_PRIMARY_WEIGHT,
    HMM_SIGNAL_TRANSITION_WEIGHT,
)
from modules.hmm.signals.scoring import normalize_scores


def test_normalize_scores_enabled():
    """Test normalize_scores when normalization is enabled."""
    if not HMM_FEATURES["normalization_enabled"]:
        pytest.skip("Normalization is disabled in config")

    score_long = 5.0
    score_short = 3.0

    normalized_long, normalized_short = normalize_scores(score_long, score_short)

    assert isinstance(normalized_long, float)
    assert isinstance(normalized_short, float)
    assert 0 <= normalized_long <= 100
    assert 0 <= normalized_short <= 100


def test_normalize_scores_disabled():
    """Test normalize_scores when normalization is disabled."""
    with patch("modules.hmm.signals.scoring.HMM_FEATURES", {"normalization_enabled": False}):
        score_long = 5.0
        score_short = 3.0

        normalized_long, normalized_short = normalize_scores(score_long, score_short)

        assert normalized_long == score_long
        assert normalized_short == score_short


def test_normalize_scores_zero_scores():
    """Test normalize_scores with zero scores."""
    if not HMM_FEATURES["normalization_enabled"]:
        pytest.skip("Normalization is disabled in config")

    normalized_long, normalized_short = normalize_scores(0.0, 0.0)

    assert normalized_long == 0.0
    assert normalized_short == 0.0


def test_normalize_scores_max_possible():
    """Test normalize_scores with maximum possible scores."""
    if not HMM_FEATURES["normalization_enabled"]:
        pytest.skip("Normalization is disabled in config")

    max_possible = (
        HMM_SIGNAL_PRIMARY_WEIGHT
        + (HMM_SIGNAL_TRANSITION_WEIGHT * 3)
        + (HMM_SIGNAL_ARM_WEIGHT * 2)
        + HMM_HIGH_ORDER_MAX_SCORE
    )

    normalized_long, normalized_short = normalize_scores(max_possible, 0.0)

    assert abs(normalized_long - 100.0) < 0.01
    assert normalized_short == 0.0


def test_normalize_scores_half_max():
    """Test normalize_scores with half of maximum scores."""
    if not HMM_FEATURES["normalization_enabled"]:
        pytest.skip("Normalization is disabled in config")

    max_possible = (
        HMM_SIGNAL_PRIMARY_WEIGHT
        + (HMM_SIGNAL_TRANSITION_WEIGHT * 3)
        + (HMM_SIGNAL_ARM_WEIGHT * 2)
        + HMM_HIGH_ORDER_MAX_SCORE
    )

    half_max = max_possible / 2
    normalized_long, normalized_short = normalize_scores(half_max, half_max)

    assert abs(normalized_long - 50.0) < 0.01
    assert abs(normalized_short - 50.0) < 0.01


def test_normalize_scores_proportional():
    """Test that normalized scores maintain proportional relationship."""
    if not HMM_FEATURES["normalization_enabled"]:
        pytest.skip("Normalization is disabled in config")

    score_long = 6.0
    score_short = 3.0

    normalized_long, normalized_short = normalize_scores(score_long, score_short)

    # Long should be twice short
    if normalized_short > 0:
        ratio = normalized_long / normalized_short
        assert abs(ratio - 2.0) < 0.01


def test_normalize_scores_with_high_order_score():
    """Test normalize_scores with high_order_score parameter."""
    if not HMM_FEATURES["normalization_enabled"]:
        pytest.skip("Normalization is disabled in config")

    score_long = 5.0
    score_short = 3.0
    high_order_score = 1.0

    normalized_long, normalized_short = normalize_scores(score_long, score_short, high_order_score)

    assert 0 <= normalized_long <= 100
    assert 0 <= normalized_short <= 100
