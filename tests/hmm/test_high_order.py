"""
Test script for modules.hmm.hmm_high_order - High Order HMM analysis.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

# Check if pomegranate is available
try:
    import pomegranate
    POMEGRANATE_AVAILABLE = True
except ImportError:
    POMEGRANATE_AVAILABLE = False

# Try to import modules
if POMEGRANATE_AVAILABLE:
    from modules.hmm.high_order import (
        HIGH_ORDER_HMM,
        BULLISH,
        NEUTRAL,
        BEARISH,
        convert_swing_to_state,
        optimize_n_states,
        create_hmm_model,
        train_model,
        predict_next_hidden_state_forward_backward,
        predict_next_observation,
        average_swing_distance,
        evaluate_model_accuracy,
        hmm_high_order,
    )
else:
    # Define dummy values for tests that don't need pomegranate
    HIGH_ORDER_HMM = None
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    convert_swing_to_state = None
    optimize_n_states = None
    create_hmm_model = None
    train_model = None
    predict_next_hidden_state_forward_backward = None
    predict_next_observation = None
    average_swing_distance = None
    evaluate_model_accuracy = None
    hmm_high_order = None


def _sample_ohlcv_dataframe(length: int = 200) -> pd.DataFrame:
    """Generate a sample OHLCV DataFrame with swing points."""
    idx = pd.date_range("2024-01-01", periods=length, freq="1h")
    np.random.seed(42)
    
    # Create price series with some trend
    base_price = 100.0
    trend = np.linspace(0, 20, length)
    noise = np.random.randn(length) * 2
    prices = base_price + trend + noise
    
    # Generate OHLCV
    high = prices + np.abs(np.random.randn(length) * 1)
    low = prices - np.abs(np.random.randn(length) * 1)
    open_price = prices + np.random.randn(length) * 0.5
    close = prices
    volume = np.random.rand(length) * 1000
    
    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=idx)


# Note: _get_param function was removed during refactoring
# These tests are no longer applicable as we now use explicit parameters


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_convert_swing_to_state_strict_mode():
    """Test convert_swing_to_state with strict_mode=True."""
    swing_highs = pd.DataFrame({
        "high": [110, 105, 100, 95, 90],
    }, index=pd.date_range("2024-01-01", periods=5, freq="1h"))
    
    swing_lows = pd.DataFrame({
        "low": [100, 95, 90, 85, 80],
    }, index=pd.date_range("2024-01-01", periods=5, freq="1h"))
    
    states = convert_swing_to_state(swing_highs, swing_lows, strict_mode=True)
    
    assert isinstance(states, list)
    assert len(states) > 0
    assert all(s in [0, 1, 2] for s in states)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_convert_swing_to_state_non_strict_mode():
    """Test convert_swing_to_state with strict_mode=False."""
    swing_highs = pd.DataFrame({
        "high": [110, 105, 100],
    }, index=pd.date_range("2024-01-01", periods=3, freq="1h"))
    
    swing_lows = pd.DataFrame({
        "low": [100, 95, 90],
    }, index=pd.date_range("2024-01-02", periods=3, freq="1h"))
    
    states = convert_swing_to_state(swing_highs, swing_lows, strict_mode=False)
    
    assert isinstance(states, list)
    assert all(s in [0, 1, 2] for s in states)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_convert_swing_to_state_empty_dataframes():
    """Test convert_swing_to_state with empty DataFrames."""
    empty_highs = pd.DataFrame({"high": []})
    empty_lows = pd.DataFrame({"low": []})
    
    states = convert_swing_to_state(empty_highs, empty_lows)
    
    assert states == []


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_create_hmm_model_default():
    """Test create_hmm_model with default parameters."""
    model = create_hmm_model()
    
    assert model is not None
    assert hasattr(model, "distributions")
    assert hasattr(model, "edges")


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_create_hmm_model_custom_states():
    """Test create_hmm_model with custom number of states."""
    model = create_hmm_model(n_symbols=3, n_states=4)
    
    assert model is not None
    assert len(model.distributions) == 4


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_train_model():
    """Test train_model function."""
    model = create_hmm_model(n_symbols=3, n_states=2)
    observations = [np.array([0, 1, 2, 0, 1, 2]).reshape(-1, 1)]
    
    trained_model = train_model(model, observations)
    
    assert trained_model is not None
    assert trained_model is model  # Should return the same model


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_average_swing_distance():
    """Test average_swing_distance calculation."""
    swing_highs = pd.DataFrame({
        "high": [110, 105, 100],
    }, index=pd.date_range("2024-01-01", periods=3, freq="2h"))
    
    swing_lows = pd.DataFrame({
        "low": [100, 95, 90],
    }, index=pd.date_range("2024-01-02", periods=3, freq="2h"))
    
    avg_distance = average_swing_distance(swing_highs, swing_lows)
    
    assert avg_distance > 0
    assert isinstance(avg_distance, (int, float))


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_evaluate_model_accuracy():
    """Test evaluate_model_accuracy function."""
    model = create_hmm_model(n_symbols=3, n_states=2)
    train_states = [0, 1, 2, 0, 1]
    test_states = [2, 0, 1]
    
    # Train model first
    train_observations = [np.array(train_states).reshape(-1, 1)]
    model = train_model(model, train_observations)
    
    accuracy = evaluate_model_accuracy(model, train_states, test_states)
    
    assert 0.0 <= accuracy <= 1.0
    assert isinstance(accuracy, float)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_hmm_high_order_basic():
    """Test hmm_high_order with basic valid data."""
    df = _sample_ohlcv_dataframe(200)
    
    result = hmm_high_order(df, train_ratio=0.8, eval_mode=False)
    
    assert isinstance(result, HIGH_ORDER_HMM)
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]
    assert result.next_state_duration > 0
    assert 0.0 <= result.next_state_probability <= 1.0


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_hmm_high_order_with_params():
    """Test hmm_high_order with explicit parameters."""
    df = _sample_ohlcv_dataframe(200)
    
    result = hmm_high_order(
        df, 
        train_ratio=0.8, 
        eval_mode=False, 
        orders_argrelextrema=3,
        strict_mode=True
    )
    
    assert isinstance(result, HIGH_ORDER_HMM)
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_hmm_high_order_empty_dataframe():
    """Test hmm_high_order with empty DataFrame."""
    empty_df = pd.DataFrame()
    
    result = hmm_high_order(empty_df)
    
    assert isinstance(result, HIGH_ORDER_HMM)
    assert result.next_state_with_high_order_hmm == NEUTRAL


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_hmm_high_order_missing_columns():
    """Test hmm_high_order with missing required columns."""
    df = pd.DataFrame({"close": [100, 101, 102]})
    
    result = hmm_high_order(df)
    
    assert isinstance(result, HIGH_ORDER_HMM)
    assert result.next_state_with_high_order_hmm == NEUTRAL


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_hmm_high_order_insufficient_data():
    """Test hmm_high_order with insufficient data."""
    df = _sample_ohlcv_dataframe(10)  # Too few data points
    
    result = hmm_high_order(df)
    
    assert isinstance(result, HIGH_ORDER_HMM)
    # Should return NEUTRAL or handle gracefully
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_hmm_high_order_eval_mode():
    """Test hmm_high_order with eval_mode=True."""
    df = _sample_ohlcv_dataframe(200)
    
    result = hmm_high_order(df, train_ratio=0.8, eval_mode=True)
    
    assert isinstance(result, HIGH_ORDER_HMM)
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_hmm_high_order_different_train_ratios():
    """Test hmm_high_order with different train_ratio values."""
    df = _sample_ohlcv_dataframe(200)
    
    result1 = hmm_high_order(df, train_ratio=0.7, eval_mode=False)
    result2 = hmm_high_order(df, train_ratio=0.9, eval_mode=False)
    
    assert isinstance(result1, HIGH_ORDER_HMM)
    assert isinstance(result2, HIGH_ORDER_HMM)
    assert result1.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]
    assert result2.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_optimize_n_states():
    """Test optimize_n_states function."""
    observations = [np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]).reshape(-1, 1)]
    
    optimal_states = optimize_n_states(observations, min_states=2, max_states=5, n_folds=3)
    
    assert 2 <= optimal_states <= 5
    assert isinstance(optimal_states, int)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_optimize_n_states_insufficient_data():
    """Test optimize_n_states with insufficient data."""
    observations = [np.array([0, 1]).reshape(-1, 1)]
    
    with pytest.raises(ValueError):
        optimize_n_states(observations, min_states=2, max_states=5, n_folds=3)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_predict_next_observation():
    """Test predict_next_observation function."""
    model = create_hmm_model(n_symbols=3, n_states=2)
    observations = [np.array([0, 1, 2, 0, 1]).reshape(-1, 1)]
    model = train_model(model, observations)
    
    next_obs_proba = predict_next_observation(model, observations)
    
    assert len(next_obs_proba) == 3
    assert np.isclose(np.sum(next_obs_proba), 1.0, atol=1e-6)
    assert all(0.0 <= p <= 1.0 for p in next_obs_proba)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_dataclass():
    """Test HIGH_ORDER_HMM dataclass."""
    result = HIGH_ORDER_HMM(
        next_state_with_high_order_hmm=BULLISH,
        next_state_duration=10,
        next_state_probability=0.75
    )
    
    assert result.next_state_with_high_order_hmm == BULLISH
    assert result.next_state_duration == 10
    assert result.next_state_probability == 0.75


def test_constants():
    """Test that constants are defined correctly."""
    # These constants are defined even if pomegranate is not available
    assert BULLISH == 1
    assert NEUTRAL == 0
    assert BEARISH == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

