"""
Tests for Bayesian Kelly Calculator.
"""

import pytest
import numpy as np
from modules.position_sizing.core.kelly_calculator import BayesianKellyCalculator


def test_calculate_kelly_fraction_basic():
    """Test basic Kelly fraction calculation."""
    calculator = BayesianKellyCalculator()
    
    # High win rate, good win/loss ratio
    kelly = calculator.calculate_kelly_fraction(
        win_rate=0.6,
        avg_win=0.05,  # 5% average win
        avg_loss=0.02,  # 2% average loss
        num_trades=100,
    )
    
    assert kelly >= 0.0
    assert kelly <= 0.25  # Max Kelly fraction


def test_calculate_kelly_fraction_low_win_rate():
    """Test Kelly calculation with low win rate (should return 0)."""
    calculator = BayesianKellyCalculator(min_win_rate=0.4)
    
    kelly = calculator.calculate_kelly_fraction(
        win_rate=0.3,  # Below minimum
        avg_win=0.05,
        avg_loss=0.02,
        num_trades=100,
    )
    
    assert kelly == 0.0


def test_calculate_kelly_fraction_insufficient_trades():
    """Test Kelly calculation with insufficient trades (should return 0)."""
    calculator = BayesianKellyCalculator(min_trades=10)
    
    kelly = calculator.calculate_kelly_fraction(
        win_rate=0.6,
        avg_win=0.05,
        avg_loss=0.02,
        num_trades=5,  # Below minimum
    )
    
    assert kelly == 0.0


def test_calculate_kelly_fraction_negative_result():
    """Test Kelly calculation that would be negative (should return 0)."""
    calculator = BayesianKellyCalculator()
    
    # Very low win rate relative to win/loss ratio
    kelly = calculator.calculate_kelly_fraction(
        win_rate=0.3,
        avg_win=0.01,  # Small wins
        avg_loss=0.05,  # Large losses
        num_trades=100,
    )
    
    assert kelly == 0.0


def test_calculate_kelly_from_metrics():
    """Test Kelly calculation from metrics dictionary."""
    calculator = BayesianKellyCalculator()
    
    metrics = {
        'win_rate': 0.6,
        'avg_win': 0.05,
        'avg_loss': 0.02,
        'num_trades': 100,
    }
    
    kelly = calculator.calculate_kelly_from_metrics(metrics)
    
    assert kelly >= 0.0
    assert kelly <= 0.25


def test_adjust_for_confidence():
    """Test confidence adjustment."""
    calculator = BayesianKellyCalculator()
    
    base_kelly = 0.2
    
    # High confidence - should keep most of it
    adjusted_high = calculator.adjust_for_confidence(base_kelly, confidence=0.95)
    assert adjusted_high <= base_kelly
    
    # Low confidence - should reduce more
    adjusted_low = calculator.adjust_for_confidence(base_kelly, confidence=0.8)
    assert adjusted_low <= adjusted_high


def test_get_posterior_distribution():
    """Test posterior distribution calculation."""
    calculator = BayesianKellyCalculator()
    
    posterior = calculator.get_posterior_distribution(
        num_wins=60,
        num_losses=40,
    )
    
    assert 'alpha' in posterior
    assert 'beta' in posterior
    assert 'mean' in posterior
    assert 'mode' in posterior
    assert 'confidence_interval' in posterior
    
    assert posterior['alpha'] > 0
    assert posterior['beta'] > 0
    assert 0.0 <= posterior['mean'] <= 1.0

