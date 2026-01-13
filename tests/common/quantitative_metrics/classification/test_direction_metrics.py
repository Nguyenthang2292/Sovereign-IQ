"""
Tests for direction_metrics module.
"""

import numpy as np
import pandas as pd

from modules.common.quantitative_metrics import calculate_direction_metrics


def test_calculate_direction_metrics_returns_dict_with_all_keys():
    """Test that calculate_direction_metrics returns dict with expected keys."""
    # Create a spread series that will generate active signals
    spread = pd.Series([0.1, -0.05, 0.15, -0.1, 0.2, -0.15, 0.05, -0.2] * 20)

    result = calculate_direction_metrics(spread)

    assert isinstance(result, dict)
    assert "classification_accuracy" in result
    assert "classification_precision" in result
    assert "classification_recall" in result
    assert "classification_f1" in result


def test_calculate_direction_metrics_insufficient_data():
    """Test that calculate_direction_metrics returns None values for insufficient data."""
    spread = pd.Series([0.1, -0.05, 0.15])  # Less than lookback

    result = calculate_direction_metrics(spread)

    assert result["classification_accuracy"] is None
    assert result["classification_precision"] is None
    assert result["classification_recall"] is None
    assert result["classification_f1"] is None


def test_calculate_direction_metrics_none_input():
    """Test that calculate_direction_metrics handles None input."""
    result = calculate_direction_metrics(None)

    assert result["classification_accuracy"] is None
    assert result["classification_precision"] is None
    assert result["classification_recall"] is None
    assert result["classification_f1"] is None


def test_calculate_direction_metrics_non_series_input():
    """Test that calculate_direction_metrics handles non-Series input."""
    result = calculate_direction_metrics([1, 2, 3, 4, 5])

    assert result["classification_accuracy"] is None
    assert result["classification_precision"] is None
    assert result["classification_recall"] is None
    assert result["classification_f1"] is None


def test_calculate_direction_metrics_default_parameters():
    """Test that calculate_direction_metrics works with default parameters."""
    spread = pd.Series([0.1, -0.05, 0.15, -0.1, 0.2, -0.15, 0.05, -0.2] * 20)

    result = calculate_direction_metrics(spread)

    # Should return a dict, values may be None if insufficient active signals
    assert isinstance(result, dict)


def test_calculate_direction_metrics_invalid_zscore_lookback():
    """Test that calculate_direction_metrics handles invalid zscore_lookback <= 0."""
    spread = pd.Series([0.1, -0.05, 0.15, -0.1, 0.2, -0.15, 0.05, -0.2] * 20)

    # Test zscore_lookback = 0
    result = calculate_direction_metrics(spread, zscore_lookback=0)
    assert result["classification_accuracy"] is None
    assert result["classification_precision"] is None
    assert result["classification_recall"] is None
    assert result["classification_f1"] is None

    # Test zscore_lookback < 0
    result = calculate_direction_metrics(spread, zscore_lookback=-10)
    assert result["classification_accuracy"] is None
    assert result["classification_precision"] is None
    assert result["classification_recall"] is None
    assert result["classification_f1"] is None


def test_calculate_direction_metrics_invalid_classification_zscore():
    """Test that calculate_direction_metrics handles invalid classification_zscore <= 0."""
    spread = pd.Series([0.1, -0.05, 0.15, -0.1, 0.2, -0.15, 0.05, -0.2] * 20)

    # Test classification_zscore = 0
    result = calculate_direction_metrics(spread, classification_zscore=0.0)
    assert result["classification_accuracy"] is None
    assert result["classification_precision"] is None
    assert result["classification_recall"] is None
    assert result["classification_f1"] is None

    # Test classification_zscore < 0
    result = calculate_direction_metrics(spread, classification_zscore=-0.5)
    assert result["classification_accuracy"] is None
    assert result["classification_precision"] is None
    assert result["classification_recall"] is None
    assert result["classification_f1"] is None


def test_calculate_direction_metrics_constant_spread():
    """Test that calculate_direction_metrics handles constant spread (std = 0)."""
    # Constant spread will have std = 0, leading to NaN in zscore
    spread = pd.Series([1.0] * 200)

    result = calculate_direction_metrics(spread, zscore_lookback=40, classification_zscore=0.5)

    # Should return None because zscore will be NaN (division by zero std)
    assert result["classification_accuracy"] is None
    assert result["classification_precision"] is None
    assert result["classification_recall"] is None
    assert result["classification_f1"] is None


def test_calculate_direction_metrics_all_values_same():
    """Test that calculate_direction_metrics handles spread with all same values."""
    spread = pd.Series([5.0] * 200)

    result = calculate_direction_metrics(spread, zscore_lookback=40, classification_zscore=0.5)

    # Should return None because std = 0 and zscore will be NaN
    assert result["classification_accuracy"] is None
    assert result["classification_precision"] is None
    assert result["classification_recall"] is None
    assert result["classification_f1"] is None


def test_calculate_direction_metrics_actual_direction_all_zeros():
    """Test that calculate_direction_metrics handles spread that never changes."""
    # Spread that increases then decreases by same amount, net change = 0
    spread = pd.Series([1.0, 2.0, 1.0, 2.0, 1.0, 2.0] * 50)

    result = calculate_direction_metrics(spread, zscore_lookback=40, classification_zscore=0.5)

    # Should return None because actual_direction will have many zeros
    # But if there are some non-zero directions, it might still work
    # This test verifies the edge case check works
    assert isinstance(result, dict)
    assert "classification_accuracy" in result


def test_calculate_direction_metrics_only_long_signals():
    """Test that calculate_direction_metrics handles case with only Long signals."""
    # Create spread that only generates Long signals (zscore < -threshold)
    # Use negative trend with volatility
    base = -np.linspace(0, 10, 400)  # Negative trend
    noise = np.random.RandomState(42).normal(0, 0.1, 400)
    spread = pd.Series(base + noise)

    result = calculate_direction_metrics(spread, zscore_lookback=40, classification_zscore=0.5)

    # Should still compute metrics (macro-averaging with only Long class)
    assert isinstance(result, dict)
    if result["classification_f1"] is not None:
        assert 0 <= result["classification_f1"] <= 1
        assert 0 <= result["classification_precision"] <= 1
        assert 0 <= result["classification_recall"] <= 1


def test_calculate_direction_metrics_only_short_signals():
    """Test that calculate_direction_metrics handles case with only Short signals."""
    # Create spread that only generates Short signals (zscore > threshold)
    # Use positive trend with volatility
    base = np.linspace(0, 10, 400)  # Positive trend
    noise = np.random.RandomState(42).normal(0, 0.1, 400)
    spread = pd.Series(base + noise)

    result = calculate_direction_metrics(spread, zscore_lookback=40, classification_zscore=0.5)

    # Should still compute metrics (macro-averaging with only Short class)
    assert isinstance(result, dict)
    if result["classification_f1"] is not None:
        assert 0 <= result["classification_f1"] <= 1
        assert 0 <= result["classification_precision"] <= 1
        assert 0 <= result["classification_recall"] <= 1


def test_calculate_direction_metrics_actual_direction_zeros_in_active():
    """Test that calculate_direction_metrics handles actual_direction = 0 in active signals."""
    # Create spread where some active signals have actual_direction = 0 (unchanged)
    # This tests the accuracy calculation when actual_direction can be 0
    base = np.sin(np.linspace(0, 20, 400))
    noise = np.random.RandomState(42).normal(0, 0.01, 400)  # Small noise
    spread = pd.Series(base + noise)

    result = calculate_direction_metrics(spread, zscore_lookback=40, classification_zscore=0.5)

    # Should handle actual_direction = 0 correctly (counts as wrong prediction)
    assert isinstance(result, dict)
    if result["classification_accuracy"] is not None:
        assert 0 <= result["classification_accuracy"] <= 1


def test_calculate_direction_metrics_insufficient_common_indices():
    """Test that calculate_direction_metrics handles insufficient common indices after alignment."""
    # Create spread with many NaN values that result in very few common indices
    spread_values = np.sin(np.linspace(0, 20, 100))
    spread_values[10:90] = np.nan  # Leave only 20 values
    spread = pd.Series(spread_values)

    result = calculate_direction_metrics(spread, zscore_lookback=10, classification_zscore=0.5)

    # Should return None if insufficient common indices (< 2)
    assert isinstance(result, dict)
    # May return None if not enough data after alignment
    assert "classification_accuracy" in result


def test_calculate_direction_metrics_empty_after_dropna():
    """Test that calculate_direction_metrics handles spread that becomes empty after dropna."""
    # Create spread with all NaN values
    spread = pd.Series([np.nan] * 200)

    result = calculate_direction_metrics(spread, zscore_lookback=40, classification_zscore=0.5)

    # Should return None because spread_clean will be empty
    assert result["classification_accuracy"] is None
    assert result["classification_precision"] is None
    assert result["classification_recall"] is None
    assert result["classification_f1"] is None


def test_calculate_direction_metrics_insufficient_after_dropna():
    """Test that calculate_direction_metrics handles insufficient data after removing NaN."""
    # Create spread with enough values but many NaN, leaving insufficient after dropna
    spread_values = np.sin(np.linspace(0, 20, 200))
    spread_values[50:150] = np.nan  # Leave only 100 values
    spread = pd.Series(spread_values)

    result = calculate_direction_metrics(spread, zscore_lookback=120, classification_zscore=0.5)

    # Should return None because len(spread_clean) < zscore_lookback
    assert result["classification_accuracy"] is None
    assert result["classification_precision"] is None
    assert result["classification_recall"] is None
    assert result["classification_f1"] is None
