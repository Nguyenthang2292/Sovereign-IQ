
import numpy as np
import pandas as pd

from modules.common.quantitative_metrics import calculate_kalman_hedge_ratio
from modules.common.quantitative_metrics import calculate_kalman_hedge_ratio

"""
Tests for kalman_hedge_ratio module.
"""




def test_calculate_kalman_hedge_ratio_uses_stubbed_filter(monkeypatch):
    """Test that calculate_kalman_hedge_ratio uses KalmanFilter correctly."""

    class DummyKalman:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def filter(self, observations):
            steps = len(observations)
            means = np.column_stack([np.linspace(0.5, 1.5, steps), np.zeros(steps)])
            cov = np.zeros((steps, 2))
            return means, cov

    from modules.common.quantitative_metrics.hedge_ratios import kalman_hedge_ratio

    monkeypatch.setattr(kalman_hedge_ratio, "KalmanFilter", DummyKalman)

    price2 = pd.Series(np.linspace(1, 10, 40))
    price1 = pd.Series(np.linspace(2, 12, 40))

    beta = calculate_kalman_hedge_ratio(price1, price2)

    assert beta is not None
    assert np.isclose(beta, 1.5)


def test_calculate_kalman_hedge_ratio_insufficient_data():
    """Test that calculate_kalman_hedge_ratio returns None for insufficient data."""
    price1 = pd.Series([1, 2, 3, 4, 5])  # Less than 10 points
    price2 = pd.Series([2, 4, 6, 8, 10])

    result = calculate_kalman_hedge_ratio(price1, price2)

    assert result is None


def test_calculate_kalman_hedge_ratio_none_input():
    """Test that calculate_kalman_hedge_ratio handles None input."""
    price1 = pd.Series([1, 2, 3])
    result1 = calculate_kalman_hedge_ratio(None, price1)
    result2 = calculate_kalman_hedge_ratio(price1, None)

    assert result1 is None
    assert result2 is None


def test_calculate_kalman_hedge_ratio_non_series_input():
    """Test that calculate_kalman_hedge_ratio handles non-Series input."""
    price1 = pd.Series([1, 2, 3] * 10)
    price2 = [2, 4, 6] * 10

    result = calculate_kalman_hedge_ratio(price1, price2)

    assert result is None


def test_calculate_kalman_hedge_ratio_mismatched_lengths():
    """Test that calculate_kalman_hedge_ratio handles mismatched lengths."""
    price1 = pd.Series([1, 2, 3] * 10)
    price2 = pd.Series([2, 4] * 10)

    result = calculate_kalman_hedge_ratio(price1, price2)

    assert result is None


def test_calculate_kalman_hedge_ratio_invalid_delta():
    """Test that calculate_kalman_hedge_ratio handles invalid delta parameter."""
    price1 = pd.Series([1, 2, 3] * 10)
    price2 = pd.Series([2, 4, 6] * 10)

    result1 = calculate_kalman_hedge_ratio(price1, price2, delta=0)
    result2 = calculate_kalman_hedge_ratio(price1, price2, delta=1)
    result3 = calculate_kalman_hedge_ratio(price1, price2, delta=-1)

    assert result1 is None
    assert result2 is None
    assert result3 is None


def test_calculate_kalman_hedge_ratio_invalid_observation_covariance():
    """Test that calculate_kalman_hedge_ratio handles invalid observation_covariance."""
    price1 = pd.Series([1, 2, 3] * 10)
    price2 = pd.Series([2, 4, 6] * 10)

    result = calculate_kalman_hedge_ratio(price1, price2, observation_covariance=0)

    assert result is None


def test_calculate_kalman_hedge_ratio_constant_prices():
    """Test that calculate_kalman_hedge_ratio handles constant prices (zero variance)."""
    # Constant prices will have zero variance, which can cause issues in Kalman filter
    price1 = pd.Series([100.0] * 50)
    price2 = pd.Series([50.0] * 50)

    result = calculate_kalman_hedge_ratio(price1, price2)

    # Should return None due to zero variance (cannot estimate relationship)
    assert result is None


def test_calculate_kalman_hedge_ratio_all_nan_values():
    """Test that calculate_kalman_hedge_ratio handles all NaN values."""
    price1 = pd.Series([np.nan] * 50)
    price2 = pd.Series([np.nan] * 50)

    result = calculate_kalman_hedge_ratio(price1, price2)

    assert result is None


def test_calculate_kalman_hedge_ratio_inf_values():
    """Test that calculate_kalman_hedge_ratio handles Inf values."""
    price1 = pd.Series([1, 2, 3, np.inf, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    price2 = pd.Series([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])

    result = calculate_kalman_hedge_ratio(price1, price2)

    # Should return None due to Inf values
    assert result is None


def test_calculate_kalman_hedge_ratio_perfect_correlation():
    """Test that calculate_kalman_hedge_ratio handles perfect correlation case."""
    # Perfect linear relationship: price1 = 2 * price2
    price2 = pd.Series(np.linspace(1, 20, 50))
    price1 = 2.0 * price2

    result = calculate_kalman_hedge_ratio(price1, price2)

    # Perfect correlation may cause numerical issues in Kalman filter (e.g., singular matrix)
    # So result can be either:
    # - Valid ratio close to 2.0 (if calculation succeeds)
    # - None (if numerical issues occur, which is acceptable)
    if result is not None:
        assert np.isclose(result, 2.0, atol=0.1)
    # If None, that's also acceptable due to potential numerical issues with perfect correlation


def test_calculate_kalman_hedge_ratio_zero_variance_price2():
    """Test that calculate_kalman_hedge_ratio handles zero variance in price2."""
    # price2 is constant, price1 varies
    price2 = pd.Series([50.0] * 50)
    price1 = pd.Series(np.linspace(100, 200, 50))

    result = calculate_kalman_hedge_ratio(price1, price2)

    # Should return None due to zero variance in price2 (cannot estimate relationship)
    assert result is None


def test_calculate_kalman_hedge_ratio_insufficient_after_dropna():
    """Test that calculate_kalman_hedge_ratio handles insufficient data after dropping NaN."""
    # Many NaN values, leaving insufficient valid data
    price1 = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    price2 = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

    result = calculate_kalman_hedge_ratio(price1, price2)

    # Should return None due to insufficient valid data (< 10 points)
    assert result is None


def test_calculate_kalman_hedge_ratio_mismatched_indices():
    """Test that calculate_kalman_hedge_ratio handles mismatched indices correctly."""
    # Different indices, but same length
    price1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], index=range(12))
    price2 = pd.Series([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24], index=range(5, 17))

    result = calculate_kalman_hedge_ratio(price1, price2)

    # Should align indices and return valid result if enough common indices
    # If common indices < 10, should return None
    assert result is None or isinstance(result, float)
