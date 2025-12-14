"""
Test script for modules.hmm.signals.combiner - Signal combiner.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from modules.hmm.signals.combiner import combine_signals, HMMSignalCombiner
from modules.hmm.signals.registry import HMMStrategyRegistry
from modules.hmm.signals.strategy import HMMStrategyResult
from modules.hmm.signals.resolution import (
    LONG,
    HOLD,
    SHORT,
)


def _sample_ohlcv_dataframe(length: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV DataFrame for testing."""
    idx = pd.date_range("2024-01-01", periods=length, freq="h")
    np.random.seed(42)
    base_price = 100.0
    prices = []
    for i in range(length):
        change = np.random.normal(0, 0.5)
        base_price += change
        prices.append(base_price)
    
    df = pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
    }, index=idx)
    return df


def test_combine_signals_invalid_dataframe():
    """Test combine_signals with invalid DataFrame."""
    df = pd.DataFrame()  # Empty DataFrame
    
    result = combine_signals(df)
    
    assert result["combined_signal"] == HOLD
    assert result["confidence"] == 0.0
    assert len(result["signals"]) == 0


def test_combine_signals_insufficient_data():
    """Test combine_signals with insufficient data."""
    df = _sample_ohlcv_dataframe(10)  # Less than 20 rows
    
    result = combine_signals(df)
    
    assert result["combined_signal"] == HOLD
    assert result["confidence"] == 0.0


@patch('modules.hmm.core.kama.hmm_kama')
@patch('modules.hmm.core.swings.hmm_swings')
@patch('modules.hmm.core.high_order.true_high_order_hmm')
def test_combine_signals_model_error(mock_true_high_order, mock_swings, mock_kama):
    """Test combine_signals when model raises error."""
    df = _sample_ohlcv_dataframe(50)
    
    # Mock models to raise error
    mock_kama.side_effect = ValueError("Test error")
    
    result = combine_signals(df)
    
    # Should return HOLD for failed strategies
    assert result["combined_signal"] in [LONG, HOLD, SHORT]
    assert "signals" in result
    assert "errors" in result["metadata"]


@patch('modules.hmm.core.kama.hmm_kama')
@patch('modules.hmm.core.swings.hmm_swings')
@patch('modules.hmm.core.high_order.true_high_order_hmm')
def test_combine_signals_basic(mock_true_high_order, mock_swings, mock_kama):
    """Test combine_signals with basic valid data."""
    df = _sample_ohlcv_dataframe(50)
    
    # Mock HMM_KAMA result
    from modules.hmm.core.kama import HMM_KAMA
    mock_kama_result = HMM_KAMA(
        next_state_with_hmm_kama=3,  # Bullish strong
        current_state_of_state_using_std=1,
        current_state_of_state_using_hmm=1,
        current_state_of_state_using_kmeans=1,
        state_high_probabilities_using_arm_apriori=3,
        state_high_probabilities_using_arm_fpgrowth=3,
    )
    mock_kama.return_value = mock_kama_result
    
    # Mock Swings HMM result
    from modules.hmm.core.swings import HMM_SWINGS
    mock_swings_result = HMM_SWINGS(
        next_state_with_high_order_hmm=1,  # BULLISH
        next_state_duration=5,
        next_state_probability=0.8
    )
    mock_swings.return_value = mock_swings_result
    
    # Mock True High-Order HMM result
    mock_true_high_order_result = HMM_SWINGS(
        next_state_with_high_order_hmm=1,  # BULLISH
        next_state_duration=5,
        next_state_probability=0.75
    )
    mock_true_high_order.return_value = mock_true_high_order_result
    
    result = combine_signals(df)
    
    # Should return valid result structure
    assert "signals" in result
    assert "combined_signal" in result
    assert "confidence" in result
    assert "votes" in result
    assert "metadata" in result
    assert result["combined_signal"] in [LONG, HOLD, SHORT]
    assert 0.0 <= result["confidence"] <= 1.0


@patch('modules.hmm.core.kama.hmm_kama')
@patch('modules.hmm.core.swings.hmm_swings')
@patch('modules.hmm.core.high_order.true_high_order_hmm')
def test_combine_signals_bearish_signals(mock_true_high_order, mock_swings, mock_kama):
    """Test combine_signals with bearish signals."""
    df = _sample_ohlcv_dataframe(50)
    
    # Mock HMM_KAMA result - bearish
    from modules.hmm.core.kama import HMM_KAMA
    mock_kama_result = HMM_KAMA(
        next_state_with_hmm_kama=0,  # Bearish strong
        current_state_of_state_using_std=-1,
        current_state_of_state_using_hmm=-1,
        current_state_of_state_using_kmeans=-1,
        state_high_probabilities_using_arm_apriori=0,
        state_high_probabilities_using_arm_fpgrowth=0,
    )
    mock_kama.return_value = mock_kama_result
    
    # Mock Swings HMM result - bearish
    from modules.hmm.core.swings import HMM_SWINGS
    mock_swings_result = HMM_SWINGS(
        next_state_with_high_order_hmm=-1,  # BEARISH
        next_state_duration=5,
        next_state_probability=0.8
    )
    mock_swings.return_value = mock_swings_result
    
    # Mock True High-Order HMM result - bearish
    mock_true_high_order_result = HMM_SWINGS(
        next_state_with_high_order_hmm=-1,  # BEARISH
        next_state_duration=5,
        next_state_probability=0.75
    )
    mock_true_high_order.return_value = mock_true_high_order_result
    
    result = combine_signals(df)
    
    # Should potentially return SHORT
    assert result["combined_signal"] in [LONG, HOLD, SHORT]
    assert "signals" in result


@patch('modules.hmm.core.kama.hmm_kama')
@patch('modules.hmm.core.swings.hmm_swings')
@patch('modules.hmm.core.high_order.true_high_order_hmm')
def test_combine_signals_with_parameters(mock_true_high_order, mock_swings, mock_kama):
    """Test combine_signals with custom parameters."""
    df = _sample_ohlcv_dataframe(50)
    
    # Mock results
    from modules.hmm.core.kama import HMM_KAMA
    from modules.hmm.core.swings import HMM_SWINGS
    
    mock_kama_result = HMM_KAMA(3, 1, 1, 1, 3, 3)
    mock_kama.return_value = mock_kama_result
    
    mock_swings_result = HMM_SWINGS(1, 5, 0.8)
    mock_swings.return_value = mock_swings_result
    
    mock_true_high_order_result = HMM_SWINGS(1, 5, 0.75)
    mock_true_high_order.return_value = mock_true_high_order_result
    
    # Call with custom parameters
    result = combine_signals(
        df,
        window_kama=10,
        fast_kama=2,
        slow_kama=30,
        window_size=50,
        orders_argrelextrema=5,
        strict_mode=True,
    )
    
    # Verify result structure
    assert "signals" in result
    assert "combined_signal" in result
    assert result["combined_signal"] in [LONG, HOLD, SHORT]


def test_hmm_signal_combiner_class():
    """Test HMMSignalCombiner class directly."""
    df = _sample_ohlcv_dataframe(50)
    
    # Create registry and combiner
    registry = HMMStrategyRegistry()
    combiner = HMMSignalCombiner(registry=registry)
    
    # Load strategies from config
    from config import HMM_STRATEGIES
    registry.load_from_config(HMM_STRATEGIES)
    
    result = combiner.combine(df)
    
    # Verify result structure
    assert "signals" in result
    assert "combined_signal" in result
    assert "confidence" in result
    assert "votes" in result
    assert "metadata" in result
    assert result["combined_signal"] in [LONG, HOLD, SHORT]


def test_registry_strategy_management():
    """Test strategy registry functionality."""
    from modules.hmm.core.swings import SwingsHMMStrategy
    
    registry = HMMStrategyRegistry()
    
    # Register a strategy
    strategy = SwingsHMMStrategy(name="test_swings", weight=1.0, enabled=True)
    registry.register(strategy)
    
    # Test retrieval
    assert "test_swings" in registry
    assert registry.get("test_swings") == strategy
    assert len(registry.get_enabled()) == 1
    
    # Test disable
    strategy.enabled = False
    assert len(registry.get_enabled()) == 0
    
    # Test unregister
    assert registry.unregister("test_swings")
    assert "test_swings" not in registry
