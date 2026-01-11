
from pathlib import Path
import sys

"""
Test script for modules.hmm.high_order - True High-Order HMM implementation.
"""


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
    from modules.hmm.core.high_order import (
        N_BASE_STATES,
        N_SYMBOLS,
        TrueHighOrderHMM,
        compute_emission_probabilities_from_data_high_order,
        compute_start_probabilities_from_data_high_order,
        compute_transition_matrix_from_data_high_order,
        create_high_order_hmm_model,
        decode_expanded_state,
        expand_state_sequence,
        get_expanded_state_count,
        map_expanded_to_base_state,
        optimize_n_states_high_order,
        optimize_order_k,
        predict_next_observation_high_order,
        true_high_order_hmm,
    )
    from modules.hmm.core.swings import (
        BEARISH,
        BULLISH,
        HMM_SWINGS,
        NEUTRAL,
    )
else:
    # Define dummy values for tests that don't need pomegranate
    HMM_SWINGS = None
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    get_expanded_state_count = None
    expand_state_sequence = None
    decode_expanded_state = None
    map_expanded_to_base_state = None
    compute_transition_matrix_from_data_high_order = None
    compute_emission_probabilities_from_data_high_order = None
    compute_start_probabilities_from_data_high_order = None
    create_high_order_hmm_model = None
    optimize_n_states_high_order = None
    optimize_order_k = None
    predict_next_observation_high_order = None
    TrueHighOrderHMM = None
    true_high_order_hmm = None
    N_BASE_STATES = 3
    N_SYMBOLS = 3


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

    return pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_get_expanded_state_count():
    """Test get_expanded_state_count function."""
    assert get_expanded_state_count(3, 1) == 3
    assert get_expanded_state_count(3, 2) == 9
    assert get_expanded_state_count(3, 3) == 27
    assert get_expanded_state_count(3, 4) == 81


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_expand_state_sequence():
    """Test expand_state_sequence function."""
    states = [0, 1, 2]

    # Order 1: should return states as-is (but as integers)
    expanded_1 = expand_state_sequence(states, order=1)
    assert expanded_1 == [0, 1, 2]

    # Order 2: should create expanded states
    expanded_2 = expand_state_sequence(states, order=2)
    assert len(expanded_2) == 2  # len(states) - order + 1
    # First expanded state: (0, 1) = 0*3^1 + 1*3^0 = 1
    # Second expanded state: (1, 2) = 1*3^1 + 2*3^0 = 5
    assert expanded_2[0] == 1
    assert expanded_2[1] == 5

    # Order 3: should create single expanded state
    expanded_3 = expand_state_sequence(states, order=3)
    assert len(expanded_3) == 1
    # (0, 1, 2) = 0*3^2 + 1*3^1 + 2*3^0 = 0 + 3 + 2 = 5
    assert expanded_3[0] == 5


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_decode_expanded_state():
    """Test decode_expanded_state function."""
    # Order 2
    assert decode_expanded_state(0, order=2) == (0, 0)
    assert decode_expanded_state(1, order=2) == (0, 1)
    assert decode_expanded_state(5, order=2) == (1, 2)
    assert decode_expanded_state(8, order=2) == (2, 2)

    # Order 3
    assert decode_expanded_state(5, order=3) == (0, 1, 2)
    assert decode_expanded_state(0, order=3) == (0, 0, 0)
    assert decode_expanded_state(26, order=3) == (2, 2, 2)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_map_expanded_to_base_state():
    """Test map_expanded_to_base_state function."""
    # Order 2: should return last state in sequence
    assert map_expanded_to_base_state(0, order=2) == 0  # (0, 0) -> 0
    assert map_expanded_to_base_state(1, order=2) == 1  # (0, 1) -> 1
    assert map_expanded_to_base_state(5, order=2) == 2  # (1, 2) -> 2

    # Order 3: should return last state
    assert map_expanded_to_base_state(5, order=3) == 2  # (0, 1, 2) -> 2
    assert map_expanded_to_base_state(0, order=3) == 0  # (0, 0, 0) -> 0


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_compute_transition_matrix_from_data_high_order():
    """Test compute_transition_matrix_from_data_high_order function."""
    states = [0, 1, 2, 0, 1, 2]

    # Order 1: should work like original
    matrix_1 = compute_transition_matrix_from_data_high_order(states, n_states=3, order=1)
    assert matrix_1.shape == (3, 3)
    assert np.allclose(matrix_1.sum(axis=1), 1.0)

    # Order 2: should create expanded transition matrix
    matrix_2 = compute_transition_matrix_from_data_high_order(states, n_states=9, order=2)
    assert matrix_2.shape == (9, 9)
    assert np.allclose(matrix_2.sum(axis=1), 1.0)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_compute_emission_probabilities_from_data_high_order():
    """Test compute_emission_probabilities_from_data_high_order function."""
    states = [0, 1, 2, 0, 1, 2]

    # Order 1
    emissions_1 = compute_emission_probabilities_from_data_high_order(states, n_states=3, n_symbols=3, order=1)
    assert len(emissions_1) == 3

    # Order 2
    emissions_2 = compute_emission_probabilities_from_data_high_order(states, n_states=9, n_symbols=3, order=2)
    assert len(emissions_2) == 9


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_create_high_order_hmm_model():
    """Test create_high_order_hmm_model function."""
    states = [0, 1, 2, 0, 1, 2, 0, 1]

    # Order 1
    model_1 = create_high_order_hmm_model(n_symbols=3, n_states=3, order=1, states_data=states, use_data_driven=True)
    assert model_1 is not None

    # Order 2
    model_2 = create_high_order_hmm_model(n_symbols=3, n_states=9, order=2, states_data=states, use_data_driven=True)
    assert model_2 is not None


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_true_high_order_hmm_basic():
    """Test true_high_order_hmm with basic valid data."""
    df = _sample_ohlcv_dataframe(200)

    result = true_high_order_hmm(df, train_ratio=0.8, eval_mode=False, min_order=2, max_order=3)

    assert isinstance(result, HMM_SWINGS)
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]
    assert result.next_state_duration > 0
    assert 0.0 <= result.next_state_probability <= 1.0


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_true_high_order_hmm_with_params():
    """Test true_high_order_hmm with explicit parameters."""
    df = _sample_ohlcv_dataframe(200)

    result = true_high_order_hmm(
        df, train_ratio=0.8, eval_mode=False, orders_argrelextrema=3, strict_mode=True, min_order=2, max_order=3
    )

    assert isinstance(result, HMM_SWINGS)
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_true_high_order_hmm_empty_dataframe():
    """Test true_high_order_hmm with empty DataFrame."""
    empty_df = pd.DataFrame()

    result = true_high_order_hmm(empty_df)

    assert isinstance(result, HMM_SWINGS)
    assert result.next_state_with_high_order_hmm == NEUTRAL


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_true_high_order_hmm_class():
    """Test TrueHighOrderHMM class directly."""
    df = _sample_ohlcv_dataframe(200)

    analyzer = TrueHighOrderHMM(
        min_order=2,
        max_order=3,
        train_ratio=0.8,
    )

    result = analyzer.analyze(df, eval_mode=False)

    assert isinstance(result, HMM_SWINGS)
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]
    assert analyzer.optimal_order is not None
    assert analyzer.optimal_n_states is not None


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_state_space_expansion_consistency():
    """Test that state space expansion and decoding are consistent."""
    states = [0, 1, 2, 0, 1, 2]

    for order in [2, 3]:
        expanded = expand_state_sequence(states, order)

        for expanded_state in expanded:
            decoded = decode_expanded_state(expanded_state, order)
            # Verify we can reconstruct the expanded state
            reconstructed = 0
            for j, state in enumerate(decoded):
                reconstructed += state * (N_BASE_STATES ** (order - 1 - j))
            assert reconstructed == expanded_state


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_predict_next_observation_high_order():
    """Test predict_next_observation_high_order function."""
    # Use longer sequence for testing
    states = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

    # Test with order=1 (simpler case)
    model = create_high_order_hmm_model(n_symbols=3, n_states=3, order=1, states_data=states, use_data_driven=True)

    observations = [np.array(states).reshape(-1, 1)]

    # Train model
    from modules.hmm.core.high_order import train_model

    model = train_model(model, observations)

    # Predict
    next_obs_proba = predict_next_observation_high_order(model, observations, order=1, n_base_states=3)

    assert len(next_obs_proba) == 3
    # Check that probabilities are valid (non-negative and sum to 1)
    assert all(p >= 0 for p in next_obs_proba), f"Probabilities should be non-negative: {next_obs_proba}"
    prob_sum = np.sum(next_obs_proba)
    assert np.allclose(prob_sum, 1.0, atol=1e-6), f"Probabilities should sum to 1, got {prob_sum}"
    assert all(0 <= p <= 1 for p in next_obs_proba)
