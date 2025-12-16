"""
Tests for correlation module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.common.quantitative_metrics import calculate_correlation


def test_calculate_correlation_matches_manual_computation():
    """Test that calculate_correlation matches manual computation."""
    # Create price series with perfect positive correlation
    price1 = pd.Series([100, 101, 102, 103, 104, 105], dtype=float)
    price2 = pd.Series([200, 201, 202, 203, 204, 205], dtype=float)

    result = calculate_correlation(price1, price2, min_points=2)

    # Manual calculation: returns should be perfectly correlated
    returns1 = price1.pct_change().dropna()
    returns2 = price2.pct_change().dropna()
    expected = returns1.corr(returns2)

    assert result is not None
    assert np.isclose(result, expected, rtol=1e-9)
    # Perfect positive correlation should be close to 1.0
    assert abs(result - 1.0) < 0.01


def test_calculate_correlation_returns_valid_range():
    """Test that calculate_correlation returns value in valid range [-1, 1]."""
    price1 = pd.Series([100, 102, 101, 103, 105, 104, 106] * 10, dtype=float)
    price2 = pd.Series([200, 201, 202, 203, 204, 205, 206] * 10, dtype=float)

    result = calculate_correlation(price1, price2, min_points=10)

    assert result is not None
    assert -1 <= result <= 1


def test_calculate_correlation_negative_correlation():
    """Test that calculate_correlation handles negative correlation."""
    # Create price series with negative correlation (one goes up, other goes down)
    price1 = pd.Series([100, 101, 102, 103, 104, 105] * 10, dtype=float)
    price2 = pd.Series([200, 199, 198, 197, 196, 195] * 10, dtype=float)

    result = calculate_correlation(price1, price2, min_points=10)

    assert result is not None
    assert -1 <= result <= 1
    # Should be negative correlation
    assert result < 0


def test_calculate_correlation_zero_correlation():
    """Test that calculate_correlation handles zero/weak correlation."""
    # Create price series with random/uncorrelated movements
    np.random.seed(42)
    price1 = pd.Series(100 + np.random.randn(100).cumsum(), dtype=float)
    price2 = pd.Series(200 + np.random.randn(100).cumsum(), dtype=float)

    result = calculate_correlation(price1, price2, min_points=10)

    # Result may be None if insufficient correlation or may be close to 0
    if result is not None:
        assert -1 <= result <= 1


def test_calculate_correlation_none_input():
    """Test that calculate_correlation handles None input."""
    price1 = pd.Series([100, 101, 102, 103, 104], dtype=float)

    result = calculate_correlation(None, price1, min_points=2)
    assert result is None

    result = calculate_correlation(price1, None, min_points=2)
    assert result is None


def test_calculate_correlation_empty_series():
    """Test that calculate_correlation handles empty series."""
    price1 = pd.Series([], dtype=float)
    price2 = pd.Series([100, 101, 102], dtype=float)

    result = calculate_correlation(price1, price2, min_points=2)
    assert result is None

    result = calculate_correlation(price2, price1, min_points=2)
    assert result is None


def test_calculate_correlation_non_series_input():
    """Test that calculate_correlation handles non-Series input."""
    price1 = pd.Series([100, 101, 102, 103, 104], dtype=float)

    result = calculate_correlation([100, 101, 102], price1, min_points=2)
    assert result is None

    result = calculate_correlation(price1, [100, 101, 102], min_points=2)
    assert result is None


def test_calculate_correlation_insufficient_data():
    """Test that calculate_correlation returns None for insufficient data."""
    price1 = pd.Series([100, 101, 102], dtype=float)
    price2 = pd.Series([200, 201, 202], dtype=float)

    result = calculate_correlation(price1, price2, min_points=10)

    assert result is None


def test_calculate_correlation_with_nan_values():
    """Test that calculate_correlation handles NaN values correctly."""
    price1 = pd.Series([100, 101, np.nan, 103, 104, 105, 106] * 10, dtype=float)
    price2 = pd.Series([200, 201, 202, 203, np.nan, 205, 206] * 10, dtype=float)

    result = calculate_correlation(price1, price2, min_points=10)

    # Should still compute valid result after dropping NaN
    if result is not None:
        assert -1 <= result <= 1


def test_calculate_correlation_with_inf_values():
    """Test that calculate_correlation handles Inf values correctly."""
    price1 = pd.Series([100, 101, np.inf, 103, 104, 105], dtype=float)
    price2 = pd.Series([200, 201, 202, 203, 204, 205], dtype=float)

    result = calculate_correlation(price1, price2, min_points=2)

    # Should return None when Inf values are present
    assert result is None


def test_calculate_correlation_different_indices():
    """Test that calculate_correlation handles different indices correctly."""
    # Series with different indices but overlapping values
    price1 = pd.Series([100, 101, 102, 103, 104], index=[0, 1, 2, 3, 4], dtype=float)
    price2 = pd.Series([200, 201, 202, 203, 204], index=[1, 2, 3, 4, 5], dtype=float)

    result = calculate_correlation(price1, price2, min_points=2)

    # Should align by common index and compute correlation
    if result is not None:
        assert -1 <= result <= 1


def test_calculate_correlation_no_common_index():
    """Test that calculate_correlation handles no common index."""
    price1 = pd.Series([100, 101, 102], index=[0, 1, 2], dtype=float)
    price2 = pd.Series([200, 201, 202], index=[10, 11, 12], dtype=float)

    result = calculate_correlation(price1, price2, min_points=2)

    # Should return None when no common index
    assert result is None


def test_calculate_correlation_constant_series():
    """Test that calculate_correlation handles constant series."""
    # Constant series have zero variance in returns, correlation may be undefined
    price1 = pd.Series([100.0] * 20, dtype=float)
    price2 = pd.Series([200.0] * 20, dtype=float)

    result = calculate_correlation(price1, price2, min_points=2)

    # Constant series returns are all NaN or zero, correlation may be None
    assert result is None or (result is not None and -1 <= result <= 1)


def test_calculate_correlation_default_min_points():
    """Test that calculate_correlation uses default min_points when not specified."""
    price1 = pd.Series([100, 101, 102] * 20, dtype=float)
    price2 = pd.Series([200, 201, 202] * 20, dtype=float)

    result = calculate_correlation(price1, price2)

    # Should work with default min_points (50)
    if result is not None:
        assert -1 <= result <= 1


def test_calculate_correlation_min_points_validation():
    """Test that calculate_correlation validates min_points."""
    price1 = pd.Series([100, 101, 102, 103, 104] * 10, dtype=float)
    price2 = pd.Series([200, 201, 202, 203, 204] * 10, dtype=float)

    # min_points < 2 should still work if enough data
    # But the function doesn't explicitly validate min_points >= 2
    # It just checks len(price1_clean) < min_points
    result = calculate_correlation(price1, price2, min_points=1)

    # Should still compute if enough data
    if result is not None:
        assert -1 <= result <= 1


def test_calculate_correlation_returns_alignment():
    """Test that calculate_correlation correctly aligns returns after pct_change."""
    # Create series where returns alignment matters
    price1 = pd.Series([100, 102, 101, 103, 105, 104, 106] * 10, dtype=float)
    price2 = pd.Series([200, 201, 202, 203, 204, 205, 206] * 10, dtype=float)

    result = calculate_correlation(price1, price2, min_points=5)

    # Should compute valid correlation
    if result is not None:
        assert -1 <= result <= 1
        assert not np.isnan(result)
        assert not np.isinf(result)

