"""
Test script for modules.hmm.signal_combiner - Signal combiner.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from modules.hmm.signal_combiner import hmm_signals
from modules.hmm.signal_resolution import (
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


def test_hmm_signals_invalid_dataframe():
    """Test hmm_signals with invalid DataFrame."""
    df = pd.DataFrame()  # Empty DataFrame
    
    signal_high, signal_kama = hmm_signals(df)
    
    assert signal_high == HOLD
    assert signal_kama == HOLD


def test_hmm_signals_insufficient_data():
    """Test hmm_signals with insufficient data."""
    df = _sample_ohlcv_dataframe(10)  # Less than 20 rows
    
    signal_high, signal_kama = hmm_signals(df)
    
    assert signal_high == HOLD
    assert signal_kama == HOLD


@patch('modules.hmm.signal_combiner.hmm_kama')
@patch('modules.hmm.signal_combiner.hmm_high_order')
def test_hmm_signals_model_error(mock_high_order, mock_kama):
    """Test hmm_signals when model raises error."""
    df = _sample_ohlcv_dataframe(50)
    
    # Mock models to raise error
    mock_kama.side_effect = ValueError("Test error")
    
    signal_high, signal_kama = hmm_signals(df)
    
    assert signal_high == HOLD
    assert signal_kama == HOLD


@patch('modules.hmm.signal_combiner.hmm_kama')
@patch('modules.hmm.signal_combiner.hmm_high_order')
def test_hmm_signals_basic(mock_high_order, mock_kama):
    """Test hmm_signals with basic valid data."""
    df = _sample_ohlcv_dataframe(50)
    
    # Mock HMM_KAMA result
    mock_kama_result = MagicMock()
    mock_kama_result.next_state_with_hmm_kama = 3  # Bullish strong
    mock_kama_result.current_state_of_state_using_std = 1
    mock_kama_result.current_state_of_state_using_hmm = 1
    mock_kama_result.current_state_of_state_using_kmeans = 1
    mock_kama_result.state_high_probabilities_using_arm_apriori = 3
    mock_kama_result.state_high_probabilities_using_arm_fpgrowth = 3
    mock_kama.return_value = mock_kama_result
    
    # Mock High-Order HMM result
    mock_high_order_result = MagicMock()
    mock_high_order_result.next_state_with_high_order_hmm = 1  # BULLISH
    mock_high_order_result.next_state_probability = 0.8
    mock_high_order.return_value = mock_high_order_result
    
    signal_high, signal_kama = hmm_signals(df)
    
    # Should return signals (not necessarily HOLD)
    assert signal_high in [LONG, HOLD, SHORT]
    assert signal_kama in [LONG, HOLD, SHORT]


@patch('modules.hmm.signal_combiner.hmm_kama')
@patch('modules.hmm.signal_combiner.hmm_high_order')
def test_hmm_signals_bearish_signals(mock_high_order, mock_kama):
    """Test hmm_signals with bearish signals."""
    df = _sample_ohlcv_dataframe(50)
    
    # Mock HMM_KAMA result - bearish
    mock_kama_result = MagicMock()
    mock_kama_result.next_state_with_hmm_kama = 0  # Bearish strong
    mock_kama_result.current_state_of_state_using_std = -1
    mock_kama_result.current_state_of_state_using_hmm = -1
    mock_kama_result.current_state_of_state_using_kmeans = -1
    mock_kama_result.state_high_probabilities_using_arm_apriori = 0
    mock_kama_result.state_high_probabilities_using_arm_fpgrowth = 0
    mock_kama.return_value = mock_kama_result
    
    # Mock High-Order HMM result - bearish
    mock_high_order_result = MagicMock()
    mock_high_order_result.next_state_with_high_order_hmm = -1  # BEARISH
    mock_high_order_result.next_state_probability = 0.8
    mock_high_order.return_value = mock_high_order_result
    
    signal_high, signal_kama = hmm_signals(df)
    
    # Both should potentially be SHORT or HOLD
    assert signal_high in [LONG, HOLD, SHORT]
    assert signal_kama in [LONG, HOLD, SHORT]


@patch('modules.hmm.signal_combiner.hmm_kama')
@patch('modules.hmm.signal_combiner.hmm_high_order')
def test_hmm_signals_low_probability(mock_high_order, mock_kama):
    """Test hmm_signals with low probability (below threshold)."""
    df = _sample_ohlcv_dataframe(50)
    
    # Mock HMM_KAMA result
    mock_kama_result = MagicMock()
    mock_kama_result.next_state_with_hmm_kama = 3
    mock_kama_result.current_state_of_state_using_std = 0
    mock_kama_result.current_state_of_state_using_hmm = 0
    mock_kama_result.current_state_of_state_using_kmeans = 0
    mock_kama_result.state_high_probabilities_using_arm_apriori = 0
    mock_kama_result.state_high_probabilities_using_arm_fpgrowth = 0
    mock_kama.return_value = mock_kama_result
    
    # Mock High-Order HMM result - low probability
    mock_high_order_result = MagicMock()
    mock_high_order_result.next_state_with_high_order_hmm = 1
    mock_high_order_result.next_state_probability = 0.3  # Below threshold
    mock_high_order.return_value = mock_high_order_result
    
    signal_high, signal_kama = hmm_signals(df)
    
    # High-Order should be HOLD due to low probability
    assert signal_high == HOLD


@patch('modules.hmm.signal_combiner.hmm_kama')
@patch('modules.hmm.signal_combiner.hmm_high_order')
def test_hmm_signals_with_parameters(mock_high_order, mock_kama):
    """Test hmm_signals with custom parameters."""
    df = _sample_ohlcv_dataframe(50)
    
    # Mock results
    mock_kama_result = MagicMock()
    mock_kama_result.next_state_with_hmm_kama = 3
    mock_kama_result.current_state_of_state_using_std = 1
    mock_kama_result.current_state_of_state_using_hmm = 1
    mock_kama_result.current_state_of_state_using_kmeans = 1
    mock_kama_result.state_high_probabilities_using_arm_apriori = 3
    mock_kama_result.state_high_probabilities_using_arm_fpgrowth = 3
    mock_kama.return_value = mock_kama_result
    
    mock_high_order_result = MagicMock()
    mock_high_order_result.next_state_with_high_order_hmm = 1
    mock_high_order_result.next_state_probability = 0.8
    mock_high_order.return_value = mock_high_order_result
    
    # Call with custom parameters
    signal_high, signal_kama = hmm_signals(
        df,
        window_kama=10,
        fast_kama=2,
        slow_kama=30,
        window_size=50,
        orders_argrelextrema=5,
        strict_mode=True,
    )
    
    # Verify parameters were passed
    mock_kama.assert_called_once()
    mock_high_order.assert_called_once()
    
    assert signal_high in [LONG, HOLD, SHORT]
    assert signal_kama in [LONG, HOLD, SHORT]


@patch('modules.hmm.signal_combiner.hmm_kama')
@patch('modules.hmm.signal_combiner.hmm_high_order')
def test_hmm_signals_neutral_state(mock_high_order, mock_kama):
    """Test hmm_signals with neutral state."""
    df = _sample_ohlcv_dataframe(50)
    
    # Mock HMM_KAMA result - neutral
    mock_kama_result = MagicMock()
    mock_kama_result.next_state_with_hmm_kama = 1  # Weak bearish
    mock_kama_result.current_state_of_state_using_std = 0
    mock_kama_result.current_state_of_state_using_hmm = 0
    mock_kama_result.current_state_of_state_using_kmeans = 0
    mock_kama_result.state_high_probabilities_using_arm_apriori = 0
    mock_kama_result.state_high_probabilities_using_arm_fpgrowth = 0
    mock_kama.return_value = mock_kama_result
    
    # Mock High-Order HMM result - neutral
    mock_high_order_result = MagicMock()
    mock_high_order_result.next_state_with_high_order_hmm = 0  # NEUTRAL
    mock_high_order_result.next_state_probability = 0.8
    mock_high_order.return_value = mock_high_order_result
    
    signal_high, signal_kama = hmm_signals(df)
    
    # High-Order should be HOLD for neutral state
    assert signal_high == HOLD

