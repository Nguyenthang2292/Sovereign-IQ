"""Tests for nonlinear interpolation modes.

This module tests the interpolation_mode parameter that controls how
real_clust values are interpolated between cluster centers.
"""

import numpy as np
import pandas as pd
import pytest

from modules.simplified_percentile_clustering.core.clustering import (
    ClusteringConfig,
    SimplifiedPercentileClustering,
)
from modules.simplified_percentile_clustering.core.features import FeatureConfig


def _sample_ohlcv_data(length=200):
    """Generate sample OHLCV data."""
    np.random.seed(42)
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(length) * 0.5)

    high = prices + np.abs(np.random.randn(length) * 0.3)
    low = prices - np.abs(np.random.randn(length) * 0.3)
    close = prices

    return pd.Series(high), pd.Series(low), pd.Series(close)


def test_linear_interpolation():
    """Test linear interpolation mode (default)."""
    high, low, close = _sample_ohlcv_data(200)

    config = ClusteringConfig(
        k=2,
        lookback=100,
        interpolation_mode="linear",
        feature_config=FeatureConfig(use_rsi=True),
    )

    clustering = SimplifiedPercentileClustering(config)
    result = clustering.compute(high, low, close)

    assert isinstance(result.real_clust, pd.Series)
    assert len(result.real_clust) == len(close)
    # real_clust should be between cluster values
    valid_real = result.real_clust.dropna()
    if len(valid_real) > 0:
        assert valid_real.min() >= 0
        assert valid_real.max() <= 1  # For k=2


def test_sigmoid_interpolation():
    """Test sigmoid interpolation mode."""
    high, low, close = _sample_ohlcv_data(200)

    config = ClusteringConfig(
        k=2,
        lookback=100,
        interpolation_mode="sigmoid",
        feature_config=FeatureConfig(use_rsi=True),
    )

    clustering = SimplifiedPercentileClustering(config)
    result = clustering.compute(high, low, close)

    assert isinstance(result.real_clust, pd.Series)
    assert len(result.real_clust) == len(close)
    # Sigmoid should produce smooth transitions
    valid_real = result.real_clust.dropna()
    if len(valid_real) > 0:
        assert valid_real.min() >= 0
        assert valid_real.max() <= 1


def test_exponential_interpolation():
    """Test exponential interpolation mode."""
    high, low, close = _sample_ohlcv_data(200)

    config = ClusteringConfig(
        k=2,
        lookback=100,
        interpolation_mode="exponential",
        feature_config=FeatureConfig(use_rsi=True),
    )

    clustering = SimplifiedPercentileClustering(config)
    result = clustering.compute(high, low, close)

    assert isinstance(result.real_clust, pd.Series)
    assert len(result.real_clust) == len(close)
    # Exponential should produce sticky behavior (stays closer to current cluster)
    valid_real = result.real_clust.dropna()
    if len(valid_real) > 0:
        assert valid_real.min() >= 0
        assert valid_real.max() <= 1
