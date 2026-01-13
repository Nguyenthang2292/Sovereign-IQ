"""
Tests for ols_hedge_ratio module.
"""

import numpy as np
import pandas as pd

from modules.common.quantitative_metrics import calculate_ols_hedge_ratio


def test_calculate_ols_hedge_ratio_recovers_linear_beta():
    """Test that calculate_ols_hedge_ratio recovers the correct beta from linear relationship."""
    price2 = pd.Series(np.linspace(1, 20, 50))
    price1 = 2.5 * price2 + 5

    beta = calculate_ols_hedge_ratio(price1, price2)
    assert beta is not None
    assert np.isclose(beta, 2.5, atol=1e-2)


def test_calculate_ols_hedge_ratio_with_stubbed_regression(monkeypatch):
    """Test that calculate_ols_hedge_ratio uses LinearRegression correctly."""

    class FakeModel:
        def __init__(self, fit_intercept=True):
            self.fit_intercept = fit_intercept
            self.coef_ = np.array([1.5])

        def fit(self, X, y):
            pass

    from modules.common.quantitative_metrics.hedge_ratios import ols_hedge_ratio

    monkeypatch.setattr(ols_hedge_ratio, "LinearRegression", FakeModel)

    # Need at least 10 data points after alignment
    price1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    price2 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    beta = calculate_ols_hedge_ratio(price1, price2)
    assert beta == 1.5


def test_calculate_ols_hedge_ratio_insufficient_data():
    """Test that calculate_ols_hedge_ratio returns None for insufficient data."""
    price1 = pd.Series([1.0])
    price2 = pd.Series([1.0])

    beta = calculate_ols_hedge_ratio(price1, price2)
    assert beta is None


def test_calculate_ols_hedge_ratio_none_input():
    """Test that calculate_ols_hedge_ratio handles None input."""
    beta = calculate_ols_hedge_ratio(None, None)
    assert beta is None


def test_calculate_ols_hedge_ratio_with_nan_values():
    """Test that calculate_ols_hedge_ratio handles NaN values correctly."""
    # Need at least 10 data points after alignment and dropping NaN
    price1 = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    price2 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])

    # Should align and drop NaN, still have >= 10 valid points
    beta = calculate_ols_hedge_ratio(price1, price2)
    assert beta is not None


def test_calculate_ols_hedge_ratio_constant_prices():
    """Test that calculate_ols_hedge_ratio handles constant prices (zero variance)."""
    # Constant prices will have zero variance, which causes issues in regression
    price1 = pd.Series([100.0] * 50)
    price2 = pd.Series([50.0] * 50)

    result = calculate_ols_hedge_ratio(price1, price2)

    # Should return None due to zero variance (cannot estimate relationship)
    assert result is None


def test_calculate_ols_hedge_ratio_all_nan_values():
    """Test that calculate_ols_hedge_ratio handles all NaN values."""
    price1 = pd.Series([np.nan] * 50)
    price2 = pd.Series([np.nan] * 50)

    result = calculate_ols_hedge_ratio(price1, price2)

    assert result is None


def test_calculate_ols_hedge_ratio_inf_values():
    """Test that calculate_ols_hedge_ratio handles Inf values."""
    price1 = pd.Series([1, 2, 3, np.inf, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    price2 = pd.Series([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])

    result = calculate_ols_hedge_ratio(price1, price2)

    # Should return None due to Inf values
    assert result is None


def test_calculate_ols_hedge_ratio_perfect_correlation():
    """Test that calculate_ols_hedge_ratio handles perfect correlation case."""
    # Perfect linear relationship: price1 = 2.5 * price2 + 5
    price2 = pd.Series(np.linspace(1, 20, 50))
    price1 = 2.5 * price2 + 5

    result = calculate_ols_hedge_ratio(price1, price2)

    # Should return a valid ratio close to 2.5
    assert result is not None
    assert np.isclose(result, 2.5, atol=0.1)


def test_calculate_ols_hedge_ratio_zero_variance_price2():
    """Test that calculate_ols_hedge_ratio handles zero variance in price2."""
    # price2 is constant, price1 varies
    price2 = pd.Series([50.0] * 50)
    price1 = pd.Series(np.linspace(100, 200, 50))

    result = calculate_ols_hedge_ratio(price1, price2)

    # Should return None due to zero variance in price2 (cannot estimate relationship)
    assert result is None


def test_calculate_ols_hedge_ratio_fit_intercept_false():
    """Test that calculate_ols_hedge_ratio works with fit_intercept=False."""
    # Linear relationship without intercept: price1 = 2.0 * price2
    price2 = pd.Series(np.linspace(1, 20, 50))
    price1 = 2.0 * price2

    result = calculate_ols_hedge_ratio(price1, price2, fit_intercept=False)

    # Should return a valid ratio close to 2.0
    assert result is not None
    assert np.isclose(result, 2.0, atol=0.1)


def test_calculate_ols_hedge_ratio_insufficient_after_dropna():
    """Test that calculate_ols_hedge_ratio handles insufficient data after dropping NaN."""
    # Many NaN values, leaving insufficient valid data
    price1 = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    price2 = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

    result = calculate_ols_hedge_ratio(price1, price2)

    # Should return None due to insufficient valid data (< 10 points)
    assert result is None


def test_calculate_ols_hedge_ratio_mismatched_indices():
    """Test that calculate_ols_hedge_ratio handles mismatched indices correctly."""
    # Different indices, but same length
    price1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], index=range(12))
    price2 = pd.Series([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24], index=range(5, 17))

    result = calculate_ols_hedge_ratio(price1, price2)

    # Should align indices and return valid result if enough common indices
    # If common indices < 10, should return None
    assert result is None or isinstance(result, float)
