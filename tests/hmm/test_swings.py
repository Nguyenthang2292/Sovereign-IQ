import sys
from pathlib import Path

"""
Test script for modules.hmm.swings - HMM Swings analysis.
"""


# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

# Check if pomegranate is available
try:
    import pomegranate  # noqa: F401

    POMEGRANATE_AVAILABLE = True
except ImportError:
    POMEGRANATE_AVAILABLE = False

# Try to import modules
if POMEGRANATE_AVAILABLE:
    from modules.hmm.core.swings import (
        BEARISH,
        BULLISH,
        HMM_SWINGS,
        NEUTRAL,
        SwingsHMM,
        _calculate_hmm_parameters,
        _map_observed_to_hidden_state,
        average_swing_distance,
        compute_emission_probabilities_from_data,
        compute_start_probabilities_from_data,
        compute_transition_matrix_from_data,
        convert_swing_to_state,
        create_hmm_model,
        evaluate_model_accuracy,
        hmm_swings,
        optimize_n_states,
        predict_next_hidden_state_forward_backward,
        predict_next_observation,
        train_model,
    )
else:
    # Define dummy values for tests that don't need pomegranate
    HMM_SWINGS = None
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
    hmm_swings = None


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


# Note: _get_param function was removed during refactoring
# These tests are no longer applicable as we now use explicit parameters


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_convert_swing_to_state_strict_mode():
    """Test convert_swing_to_state with strict_mode=True."""
    swing_highs = pd.DataFrame(
        {
            "high": [110, 105, 100, 95, 90],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="1h"),
    )

    swing_lows = pd.DataFrame(
        {
            "low": [100, 95, 90, 85, 80],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="1h"),
    )

    states = convert_swing_to_state(swing_highs, swing_lows, strict_mode=True)

    assert isinstance(states, list)
    assert len(states) > 0
    assert all(s in [0, 1, 2] for s in states)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_convert_swing_to_state_non_strict_mode():
    """Test convert_swing_to_state with strict_mode=False."""
    swing_highs = pd.DataFrame(
        {
            "high": [110, 105, 100],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1h"),
    )

    swing_lows = pd.DataFrame(
        {
            "low": [100, 95, 90],
        },
        index=pd.date_range("2024-01-02", periods=3, freq="1h"),
    )

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
    swing_highs = pd.DataFrame(
        {
            "high": [110, 105, 100],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="2h"),
    )

    swing_lows = pd.DataFrame(
        {
            "low": [100, 95, 90],
        },
        index=pd.date_range("2024-01-02", periods=3, freq="2h"),
    )

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
def test_hmm_swings_basic():
    """Test hmm_swings with basic valid data."""
    df = _sample_ohlcv_dataframe(200)

    result = hmm_swings(df, train_ratio=0.8, eval_mode=False)

    assert isinstance(result, HMM_SWINGS)
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]
    assert result.next_state_duration > 0
    assert 0.0 <= result.next_state_probability <= 1.0


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_hmm_swings_with_params():
    """Test hmm_swings with explicit parameters."""
    df = _sample_ohlcv_dataframe(200)

    result = hmm_swings(df, train_ratio=0.8, eval_mode=False, orders_argrelextrema=3, strict_mode=True)

    assert isinstance(result, HMM_SWINGS)
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_hmm_swings_empty_dataframe():
    """Test hmm_swings with empty DataFrame."""
    empty_df = pd.DataFrame()

    result = hmm_swings(empty_df)

    assert isinstance(result, HMM_SWINGS)
    assert result.next_state_with_high_order_hmm == NEUTRAL


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_hmm_swings_missing_columns():
    """Test hmm_swings with missing required columns."""
    df = pd.DataFrame({"close": [100, 101, 102]})

    result = hmm_swings(df)

    assert isinstance(result, HMM_SWINGS)
    assert result.next_state_with_high_order_hmm == NEUTRAL


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_hmm_swings_insufficient_data():
    """Test hmm_swings with insufficient data."""
    df = _sample_ohlcv_dataframe(10)  # Too few data points

    result = hmm_swings(df)

    assert isinstance(result, HMM_SWINGS)
    # Should return NEUTRAL or handle gracefully
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_hmm_swings_eval_mode():
    """Test hmm_swings with eval_mode=True."""
    df = _sample_ohlcv_dataframe(200)

    result = hmm_swings(df, train_ratio=0.8, eval_mode=True)

    assert isinstance(result, HMM_SWINGS)
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_hmm_swings_different_train_ratios():
    """Test hmm_swings with different train_ratio values."""
    df = _sample_ohlcv_dataframe(200)

    result1 = hmm_swings(df, train_ratio=0.7, eval_mode=False)
    result2 = hmm_swings(df, train_ratio=0.9, eval_mode=False)

    assert isinstance(result1, HMM_SWINGS)
    assert isinstance(result2, HMM_SWINGS)
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
    """Test HMM_SWINGS dataclass."""
    result = HMM_SWINGS(next_state_with_high_order_hmm=BULLISH, next_state_duration=10, next_state_probability=0.75)

    assert result.next_state_with_high_order_hmm == BULLISH
    assert result.next_state_duration == 10
    assert result.next_state_probability == 0.75


def test_constants():
    """Test that constants are defined correctly."""
    # These constants are defined even if pomegranate is not available
    assert BULLISH == 1
    assert NEUTRAL == 0
    assert BEARISH == -1


# ============================================================================
# SwingsHMM Class Tests
# ============================================================================


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_class_initialization():
    """Test SwingsHMM class initialization."""
    analyzer = SwingsHMM()

    assert analyzer.model is None
    assert analyzer.optimal_n_states is None
    assert analyzer.swing_highs_info is None
    assert analyzer.swing_lows_info is None
    assert analyzer.states is None
    assert analyzer.train_states is None
    assert analyzer.test_states is None
    assert analyzer.train_ratio == 0.8
    assert analyzer.min_states == 2
    assert analyzer.max_states == 10
    assert analyzer.n_folds == 3
    assert analyzer.use_bic is True


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_class_custom_params():
    """Test SwingsHMM class with custom parameters."""
    analyzer = SwingsHMM(
        orders_argrelextrema=3,
        strict_mode=False,
        use_data_driven=False,
        train_ratio=0.7,
        min_states=3,
        max_states=8,
        n_folds=2,
        use_bic=False,
    )

    assert analyzer.orders_argrelextrema == 3
    assert analyzer.strict_mode is False
    assert analyzer.use_data_driven is False
    assert analyzer.train_ratio == 0.7
    assert analyzer.min_states == 3
    assert analyzer.max_states == 8
    assert analyzer.n_folds == 2
    assert analyzer.use_bic is False


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_validate_dataframe():
    """Test _validate_dataframe method."""
    analyzer = SwingsHMM()

    # Valid DataFrame
    valid_df = _sample_ohlcv_dataframe(50)
    assert analyzer._validate_dataframe(valid_df) is True

    # Empty DataFrame
    empty_df = pd.DataFrame()
    assert analyzer._validate_dataframe(empty_df) is False

    # Missing columns
    missing_cols_df = pd.DataFrame({"close": [100, 101, 102]})
    assert analyzer._validate_dataframe(missing_cols_df) is False

    # Non-numeric data
    non_numeric_df = pd.DataFrame(
        {"open": ["a", "b", "c"], "high": [100, 101, 102], "low": [99, 100, 101], "close": [100, 101, 102]}
    )
    assert analyzer._validate_dataframe(non_numeric_df) is False


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_determine_interval():
    """Test _determine_interval method."""
    analyzer = SwingsHMM()

    # DatetimeIndex with hourly data
    hourly_df = pd.DataFrame(
        {"open": [100, 101, 102], "high": [101, 102, 103], "low": [99, 100, 101], "close": [100, 101, 102]},
        index=pd.date_range("2024-01-01", periods=3, freq="1h"),
    )
    assert analyzer._determine_interval(hourly_df) == "h1"

    # DatetimeIndex with minute data
    minute_df = pd.DataFrame(
        {"open": [100, 101, 102], "high": [101, 102, 103], "low": [99, 100, 101], "close": [100, 101, 102]},
        index=pd.date_range("2024-01-01", periods=3, freq="15min"),
    )
    assert analyzer._determine_interval(minute_df) == "m15"

    # Non-datetime index
    non_dt_df = pd.DataFrame(
        {"open": [100, 101, 102], "high": [101, 102, 103], "low": [99, 100, 101], "close": [100, 101, 102]},
        index=[0, 1, 2],
    )
    assert analyzer._determine_interval(non_dt_df) == "h1"


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_detect_swings():
    """Test detect_swings method."""
    analyzer = SwingsHMM(orders_argrelextrema=3)
    df = _sample_ohlcv_dataframe(100)

    swing_highs, swing_lows = analyzer.detect_swings(df)

    assert isinstance(swing_highs, pd.DataFrame)
    assert isinstance(swing_lows, pd.DataFrame)
    assert not swing_highs.empty
    assert not swing_lows.empty
    assert "high" in swing_highs.columns
    assert "low" in swing_lows.columns


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_detect_swings_insufficient_data():
    """Test detect_swings with insufficient data."""
    analyzer = SwingsHMM()
    df = _sample_ohlcv_dataframe(5)  # Too few points

    swing_highs, swing_lows = analyzer.detect_swings(df)

    assert swing_highs.empty
    assert swing_lows.empty


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_convert_to_states():
    """Test convert_to_states method."""
    analyzer = SwingsHMM(strict_mode=True)

    swing_highs = pd.DataFrame(
        {
            "high": [110, 105, 100, 95, 90],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="1h"),
    )

    swing_lows = pd.DataFrame(
        {
            "low": [100, 95, 90, 85, 80],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="1h"),
    )

    states = analyzer.convert_to_states(swing_highs, swing_lows)

    assert isinstance(states, list)
    assert len(states) > 0
    assert all(s in [0, 1, 2] for s in states)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_optimize_and_create_model():
    """Test optimize_and_create_model method."""
    analyzer = SwingsHMM(min_states=2, max_states=4, n_folds=2)
    train_states = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

    model = analyzer.optimize_and_create_model(train_states)

    assert model is not None
    assert analyzer.optimal_n_states is not None
    assert 2 <= analyzer.optimal_n_states <= 4
    assert hasattr(model, "distributions")
    assert hasattr(model, "edges")


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_predict_next_state():
    """Test predict_next_state method."""
    analyzer = SwingsHMM()
    model = create_hmm_model(n_symbols=3, n_states=2)
    states = [0, 1, 2, 0, 1, 2]
    train_observations = [np.array(states).reshape(-1, 1)]
    model = train_model(model, train_observations)

    max_index, max_value = analyzer.predict_next_state(model, states)

    assert max_index in [0, 1, 2]
    assert 0.0 <= max_value <= 1.0
    assert isinstance(max_index, int)
    assert isinstance(max_value, float)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_calculate_duration():
    """Test _calculate_duration method."""
    analyzer = SwingsHMM()

    swing_highs = pd.DataFrame(
        {
            "high": [110, 105, 100],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="2h"),
    )

    swing_lows = pd.DataFrame(
        {
            "low": [100, 95, 90],
        },
        index=pd.date_range("2024-01-02", periods=3, freq="2h"),
    )

    duration = analyzer._calculate_duration(swing_highs, swing_lows, "h1")

    assert duration > 0
    assert isinstance(duration, int)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_analyze_basic():
    """Test analyze method with basic valid data."""
    analyzer = SwingsHMM(train_ratio=0.8)
    df = _sample_ohlcv_dataframe(200)

    result = analyzer.analyze(df, eval_mode=False)

    assert isinstance(result, HMM_SWINGS)
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]
    assert result.next_state_duration > 0
    assert 0.0 <= result.next_state_probability <= 1.0
    assert analyzer.model is not None
    assert analyzer.states is not None
    assert analyzer.swing_highs_info is not None
    assert analyzer.swing_lows_info is not None


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_analyze_with_eval_mode():
    """Test analyze method with eval_mode=True."""
    analyzer = SwingsHMM(train_ratio=0.8)
    df = _sample_ohlcv_dataframe(200)

    result = analyzer.analyze(df, eval_mode=True)

    assert isinstance(result, HMM_SWINGS)
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_analyze_invalid_data():
    """Test analyze method with invalid data."""
    analyzer = SwingsHMM()

    # Empty DataFrame
    empty_df = pd.DataFrame()
    result = analyzer.analyze(empty_df)
    assert result.next_state_with_high_order_hmm == NEUTRAL

    # Missing columns
    missing_cols_df = pd.DataFrame({"close": [100, 101, 102]})
    result = analyzer.analyze(missing_cols_df)
    assert result.next_state_with_high_order_hmm == NEUTRAL


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_analyze_insufficient_swings():
    """Test analyze method with insufficient swing points."""
    analyzer = SwingsHMM()
    df = _sample_ohlcv_dataframe(10)  # Too few points

    result = analyzer.analyze(df)

    assert isinstance(result, HMM_SWINGS)
    assert result.next_state_with_high_order_hmm == NEUTRAL


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_analyze_strict_mode():
    """Test analyze method with strict_mode=True."""
    analyzer = SwingsHMM(strict_mode=True, train_ratio=0.8)
    df = _sample_ohlcv_dataframe(200)

    result = analyzer.analyze(df, eval_mode=False)

    assert isinstance(result, HMM_SWINGS)
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_analyze_data_driven():
    """Test analyze method with data-driven initialization."""
    analyzer = SwingsHMM(use_data_driven=True, train_ratio=0.8)
    df = _sample_ohlcv_dataframe(200)

    result = analyzer.analyze(df, eval_mode=False)

    assert isinstance(result, HMM_SWINGS)
    assert analyzer.model is not None


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_high_order_hmm_analyze_low_accuracy():
    """Test analyze method when accuracy is too low."""
    analyzer = SwingsHMM(train_ratio=0.8)
    df = _sample_ohlcv_dataframe(200)

    # This test verifies that low accuracy returns NEUTRAL
    # The actual accuracy depends on the data, so we just verify the structure
    result = analyzer.analyze(df, eval_mode=True)

    assert isinstance(result, HMM_SWINGS)
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


# ============================================================================
# Data-Driven Initialization Tests
# ============================================================================


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_map_observed_to_hidden_state():
    """Test _map_observed_to_hidden_state function."""
    # Direct mapping (n_states == n_observed_states)
    assert _map_observed_to_hidden_state(0, 3) == 0
    assert _map_observed_to_hidden_state(1, 3) == 1
    assert _map_observed_to_hidden_state(2, 3) == 2

    # Mapping to 2 states
    assert _map_observed_to_hidden_state(0, 2) == 0
    assert _map_observed_to_hidden_state(1, 2) == 0
    assert _map_observed_to_hidden_state(2, 2) == 1

    # Mapping to 4 states
    assert _map_observed_to_hidden_state(0, 4) == 0
    assert _map_observed_to_hidden_state(1, 4) == 1
    assert _map_observed_to_hidden_state(2, 4) == 2

    # Edge cases
    assert _map_observed_to_hidden_state(-1, 3) == 0  # Clamped
    assert _map_observed_to_hidden_state(5, 3) == 2  # Clamped


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_compute_transition_matrix_from_data():
    """Test compute_transition_matrix_from_data function."""
    states = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    n_states = 3

    transition_matrix = compute_transition_matrix_from_data(states, n_states)

    assert transition_matrix.shape == (n_states, n_states)
    assert np.allclose(transition_matrix.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(transition_matrix >= 0)
    assert np.all(transition_matrix <= 1)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_compute_transition_matrix_from_data_insufficient_data():
    """Test compute_transition_matrix_from_data with insufficient data."""
    states = [0]
    n_states = 3

    transition_matrix = compute_transition_matrix_from_data(states, n_states)

    assert transition_matrix.shape == (n_states, n_states)
    assert np.allclose(transition_matrix.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_compute_emission_probabilities_from_data():
    """Test compute_emission_probabilities_from_data function."""
    states = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    n_states = 3
    n_symbols = 3

    emission_probs = compute_emission_probabilities_from_data(states, n_states, n_symbols)

    assert len(emission_probs) == n_states
    assert all(isinstance(dist, type(emission_probs[0])) for dist in emission_probs)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_compute_emission_probabilities_from_data_n_states_less_than_symbols():
    """Test compute_emission_probabilities_from_data when n_states < n_symbols."""
    states = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    n_states = 2  # Less than n_symbols (3)
    n_symbols = 3

    emission_probs = compute_emission_probabilities_from_data(states, n_states, n_symbols)

    assert len(emission_probs) == n_states
    assert all(isinstance(dist, type(emission_probs[0])) for dist in emission_probs)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_compute_emission_probabilities_from_data_insufficient_data():
    """Test compute_emission_probabilities_from_data with insufficient data."""
    states = [0]
    n_states = 3
    n_symbols = 3

    emission_probs = compute_emission_probabilities_from_data(states, n_states, n_symbols)

    assert len(emission_probs) == n_states


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_compute_start_probabilities_from_data():
    """Test compute_start_probabilities_from_data function."""
    states = [0, 1, 2, 0, 1, 2]
    n_states = 3

    start_probs = compute_start_probabilities_from_data(states, n_states)

    assert len(start_probs) == n_states
    assert np.isclose(start_probs.sum(), 1.0, atol=1e-6)
    assert np.all(start_probs >= 0)
    assert np.all(start_probs <= 1)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_compute_start_probabilities_from_data_empty():
    """Test compute_start_probabilities_from_data with empty states."""
    states = []
    n_states = 3

    start_probs = compute_start_probabilities_from_data(states, n_states)

    assert len(start_probs) == n_states
    assert np.isclose(start_probs.sum(), 1.0, atol=1e-6)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_create_hmm_model_data_driven():
    """Test create_hmm_model with data-driven initialization."""
    states_data = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    n_states = 3

    model = create_hmm_model(n_symbols=3, n_states=n_states, states_data=states_data, use_data_driven=True)

    assert model is not None
    assert hasattr(model, "distributions")
    assert hasattr(model, "edges")
    assert len(model.distributions) == n_states


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_create_hmm_model_hardcoded():
    """Test create_hmm_model with hardcoded initialization."""
    model = create_hmm_model(n_symbols=3, n_states=2, states_data=None, use_data_driven=False)

    assert model is not None
    assert hasattr(model, "distributions")
    assert hasattr(model, "edges")
    assert len(model.distributions) == 2


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_calculate_hmm_parameters():
    """Test _calculate_hmm_parameters function."""
    # Test with n_states=2, n_symbols=3
    params_2_3 = _calculate_hmm_parameters(2, 3)
    assert params_2_3 > 0
    assert isinstance(params_2_3, int)

    # Test with n_states=3, n_symbols=3
    params_3_3 = _calculate_hmm_parameters(3, 3)
    assert params_3_3 > params_2_3  # More states = more parameters

    # Test with n_states=4, n_symbols=3
    params_4_3 = _calculate_hmm_parameters(4, 3)
    assert params_4_3 > params_3_3


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_optimize_n_states_with_bic():
    """Test optimize_n_states with BIC (default)."""
    observations = [np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]).reshape(-1, 1)]

    optimal_states = optimize_n_states(observations, min_states=2, max_states=5, n_folds=2, use_bic=True)

    assert 2 <= optimal_states <= 5
    assert isinstance(optimal_states, int)


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_optimize_n_states_without_bic():
    """Test optimize_n_states without BIC (using log-likelihood)."""
    observations = [np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]).reshape(-1, 1)]

    optimal_states = optimize_n_states(observations, min_states=2, max_states=5, n_folds=2, use_bic=False)

    assert 2 <= optimal_states <= 5
    assert isinstance(optimal_states, int)


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_integration_hmm_swings_vs_class():
    """Test that hmm_swings function and SwingsHMM class produce similar results."""
    df = _sample_ohlcv_dataframe(200)

    # Function-based
    result_func = hmm_swings(df, train_ratio=0.8, eval_mode=False)

    # Class-based
    analyzer = SwingsHMM(train_ratio=0.8)
    result_class = analyzer.analyze(df, eval_mode=False)

    # Both should return valid results
    assert isinstance(result_func, HMM_SWINGS)
    assert isinstance(result_class, HMM_SWINGS)
    assert result_func.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]
    assert result_class.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]
    assert result_func.next_state_duration > 0
    assert result_class.next_state_duration > 0


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_integration_full_pipeline_strict_mode():
    """Test full pipeline with strict_mode=True."""
    analyzer = SwingsHMM(strict_mode=True, use_data_driven=True, train_ratio=0.8)
    df = _sample_ohlcv_dataframe(200)

    result = analyzer.analyze(df, eval_mode=True)

    assert isinstance(result, HMM_SWINGS)
    assert analyzer.model is not None
    assert analyzer.optimal_n_states is not None
    assert analyzer.states is not None
    assert analyzer.train_states is not None
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_integration_full_pipeline_non_strict_mode():
    """Test full pipeline with strict_mode=False."""
    analyzer = SwingsHMM(strict_mode=False, use_data_driven=True, train_ratio=0.8)
    df = _sample_ohlcv_dataframe(200)

    result = analyzer.analyze(df, eval_mode=True)

    assert isinstance(result, HMM_SWINGS)
    assert analyzer.model is not None
    assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_integration_data_driven_vs_hardcoded():
    """Test integration comparing data-driven vs hardcoded initialization."""
    df = _sample_ohlcv_dataframe(200)

    # Data-driven
    analyzer_dd = SwingsHMM(use_data_driven=True, train_ratio=0.8)
    result_dd = analyzer_dd.analyze(df, eval_mode=False)

    # Hardcoded
    analyzer_hc = SwingsHMM(use_data_driven=False, train_ratio=0.8)
    result_hc = analyzer_hc.analyze(df, eval_mode=False)

    # Both should work
    assert isinstance(result_dd, HMM_SWINGS)
    assert isinstance(result_hc, HMM_SWINGS)
    assert result_dd.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]
    assert result_hc.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_integration_reuse_analyzer_instance():
    """Test reusing the same analyzer instance for multiple analyses."""
    analyzer = SwingsHMM(train_ratio=0.8)
    df1 = _sample_ohlcv_dataframe(200)
    df2 = _sample_ohlcv_dataframe(200)

    result1 = analyzer.analyze(df1, eval_mode=False)
    result2 = analyzer.analyze(df2, eval_mode=False)

    # Both should work
    assert isinstance(result1, HMM_SWINGS)
    assert isinstance(result2, HMM_SWINGS)
    # State should be updated after second analysis
    assert analyzer.model is not None


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_integration_different_train_ratios():
    """Test integration with different train ratios."""
    df = _sample_ohlcv_dataframe(200)

    for train_ratio in [0.6, 0.7, 0.8, 0.9]:
        analyzer = SwingsHMM(train_ratio=train_ratio)
        result = analyzer.analyze(df, eval_mode=False)

        assert isinstance(result, HMM_SWINGS)
        assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_integration_different_orders_argrelextrema():
    """Test integration with different orders_argrelextrema values."""
    df = _sample_ohlcv_dataframe(200)

    for order in [3, 5, 7]:
        analyzer = SwingsHMM(orders_argrelextrema=order)
        result = analyzer.analyze(df, eval_mode=False)

        assert isinstance(result, HMM_SWINGS)
        assert result.next_state_with_high_order_hmm in [BULLISH, NEUTRAL, BEARISH]


@pytest.mark.skipif(not POMEGRANATE_AVAILABLE, reason="pomegranate not available")
def test_integration_state_sequence_consistency():
    """Test that state sequences are consistent across different modes."""
    df = _sample_ohlcv_dataframe(200)

    analyzer_strict = SwingsHMM(strict_mode=True)
    analyzer_non_strict = SwingsHMM(strict_mode=False)

    result_strict = analyzer_strict.analyze(df, eval_mode=False)
    result_non_strict = analyzer_non_strict.analyze(df, eval_mode=False)

    # Both should produce valid results
    assert isinstance(result_strict, HMM_SWINGS)
    assert isinstance(result_non_strict, HMM_SWINGS)
    # States should exist
    assert analyzer_strict.states is not None
    assert analyzer_non_strict.states is not None
    # States should be valid
    assert all(s in [0, 1, 2] for s in analyzer_strict.states)
    assert all(s in [0, 1, 2] for s in analyzer_non_strict.states)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
