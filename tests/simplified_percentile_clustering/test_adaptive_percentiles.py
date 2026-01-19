"""Tests for adaptive percentile calculation with volatility adjustment.

This module tests the volatility_adjustment feature that dynamically adjusts
percentiles based on market volatility conditions.
"""

import numpy as np
import pandas as pd
import pytest

from modules.simplified_percentile_clustering.core.centers import ClusterCenters, compute_centers


def _sample_values(length=200):
    """Generate sample values."""
    np.random.seed(42)
    return pd.Series(100.0 + np.random.randn(length) * 10.0)


def test_volatility_adjustment_basic():
    """Test basic volatility adjustment functionality."""
    values = _sample_values(200)

    centers_static = compute_centers(values, lookback=50, p_low=5, p_high=95, k=2, volatility_adjustment=False)
    centers_adaptive = compute_centers(values, lookback=50, p_low=5, p_high=95, k=2, volatility_adjustment=True)

    # Both should produce valid results (after lookback period)
    assert isinstance(centers_static, pd.DataFrame)
    assert isinstance(centers_adaptive, pd.DataFrame)
    assert len(centers_static) == len(centers_adaptive) == len(values)
    # Check valid values after lookback
    valid_static = centers_static.iloc[50:].dropna()
    valid_adaptive = centers_adaptive.iloc[50:].dropna()
    assert len(valid_static) > 0, "Should have valid static centers after lookback"
    assert len(valid_adaptive) > 0, "Should have valid adaptive centers after lookback"


def test_volatility_adjustment_high_volatility():
    """Test volatility adjustment with high volatility data."""
    np.random.seed(42)
    # Create data with high volatility
    high_vol_data = pd.Series(100.0 + np.random.randn(200) * 30.0)  # High std

    centers_static = compute_centers(high_vol_data, lookback=50, p_low=5, p_high=95, k=2, volatility_adjustment=False)
    centers_adaptive = compute_centers(high_vol_data, lookback=50, p_low=5, p_high=95, k=2, volatility_adjustment=True)

    # Adaptive should handle high volatility differently
    valid_adaptive = centers_adaptive.iloc[50:].dropna()
    assert len(valid_adaptive) > 0, "Should have valid adaptive centers after lookback"
    # Centers should still be ordered
    if len(valid_adaptive) > 0:
        assert (valid_adaptive["k0"] < valid_adaptive["k1"]).all()


def test_volatility_adjustment_low_volatility():
    """Test volatility adjustment with low volatility data."""
    np.random.seed(42)
    # Create data with low volatility
    low_vol_data = pd.Series(100.0 + np.random.randn(200) * 2.0)  # Low std

    centers_static = compute_centers(low_vol_data, lookback=50, p_low=5, p_high=95, k=2, volatility_adjustment=False)
    centers_adaptive = compute_centers(low_vol_data, lookback=50, p_low=5, p_high=95, k=2, volatility_adjustment=True)

    # Both should work (after lookback)
    valid_static = centers_static.iloc[50:].dropna()
    valid_adaptive = centers_adaptive.iloc[50:].dropna()
    assert len(valid_static) > 0
    assert len(valid_adaptive) > 0


def test_volatility_adjustment_changing_volatility():
    """Test volatility adjustment with changing volatility over time."""
    np.random.seed(42)
    # Create data with changing volatility
    low_vol_part = 100.0 + np.random.randn(100) * 2.0
    high_vol_part = 100.0 + np.random.randn(100) * 20.0
    changing_vol_data = pd.Series(np.concatenate([low_vol_part, high_vol_part]))

    centers_adaptive = compute_centers(changing_vol_data, lookback=50, p_low=5, p_high=95, k=2, volatility_adjustment=True)

    # Should adapt to changing volatility (after lookback)
    valid_adaptive = centers_adaptive.iloc[50:].dropna()
    assert len(valid_adaptive) > 0, "Should have valid adaptive centers after lookback"
    if len(valid_adaptive) > 0:
        assert (valid_adaptive["k0"] < valid_adaptive["k1"]).all()


def test_cluster_centers_volatility_adjustment():
    """Test ClusterCenters class with volatility adjustment."""
    calculator = ClusterCenters(lookback=50, p_low=5.0, p_high=95.0, k=2, volatility_adjustment=True)

    # Add values with varying volatility
    for i in range(100):
        # Simulate changing volatility
        vol = 5.0 if i < 50 else 15.0
        value = 100.0 + np.random.randn() * vol
        centers = calculator.update(value)

        if i >= 10:  # After enough data
            assert len(centers) == 2
            assert centers[0] < centers[1]
            assert not any(np.isnan(c) for c in centers)
