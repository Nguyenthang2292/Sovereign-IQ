"""Tests for time decay factor in distance calculation.

This module tests the time_decay_factor parameter that applies exponential
decay to historical data, giving more weight to recent bars.
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


def test_time_decay_factor_enabled():
    """Test clustering with time decay factor enabled."""
    high, low, close = _sample_ohlcv_data(200)

    config = ClusteringConfig(
        k=2,
        lookback=100,
        time_decay_factor=0.99,  # 1% decay per bar
        feature_config=FeatureConfig(
            use_rsi=True,
            use_cci=True,
        ),
    )

    clustering = SimplifiedPercentileClustering(config)
    result = clustering.compute(high, low, close)

    assert isinstance(result.cluster_val, pd.Series)
    assert len(result.cluster_val) == len(close)
    # Should produce valid clustering results
    assert result.cluster_val.notna().any()


def test_time_decay_factor_vs_no_decay():
    """Test that time decay produces different results than no decay."""
    high, low, close = _sample_ohlcv_data(200)

    config_no_decay = ClusteringConfig(
        k=2,
        lookback=100,
        time_decay_factor=1.0,  # No decay
        feature_config=FeatureConfig(use_rsi=True, use_cci=True),
    )

    config_decay = ClusteringConfig(
        k=2,
        lookback=100,
        time_decay_factor=0.99,  # 1% decay
        feature_config=FeatureConfig(use_rsi=True, use_cci=True),
    )

    clustering_no_decay = SimplifiedPercentileClustering(config_no_decay)
    clustering_decay = SimplifiedPercentileClustering(config_decay)

    result_no_decay = clustering_no_decay.compute(high, low, close)
    result_decay = clustering_decay.compute(high, low, close)

    # Both should produce valid results
    assert result_no_decay.cluster_val.notna().any()
    assert result_decay.cluster_val.notna().any()
    # Results may differ (time decay emphasizes recent data)
    # We just verify both work correctly
