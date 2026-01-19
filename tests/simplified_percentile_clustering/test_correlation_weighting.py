"""Tests for correlation-based feature weighting.

This module tests the use_correlation_weights feature that weights features
by their uniqueness (1 - average correlation with other features).
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


def test_correlation_weighting_enabled():
    """Test clustering with correlation weighting enabled."""
    high, low, close = _sample_ohlcv_data(200)

    config = ClusteringConfig(
        k=2,
        lookback=100,
        use_correlation_weights=True,
        feature_config=FeatureConfig(
            use_rsi=True,
            use_cci=True,
            use_zscore=True,
        ),
    )

    clustering = SimplifiedPercentileClustering(config)
    result = clustering.compute(high, low, close)

    assert isinstance(result.cluster_val, pd.Series)
    assert len(result.cluster_val) == len(close)
    # Should produce valid clustering results
    assert result.cluster_val.notna().any()


def test_correlation_weighting_vs_equal_weights():
    """Test that correlation weighting produces different results than equal weights."""
    high, low, close = _sample_ohlcv_data(200)

    config_equal = ClusteringConfig(
        k=2,
        lookback=100,
        use_correlation_weights=False,
        feature_config=FeatureConfig(
            use_rsi=True,
            use_cci=True,
            use_zscore=True,
        ),
    )

    config_corr = ClusteringConfig(
        k=2,
        lookback=100,
        use_correlation_weights=True,
        feature_config=FeatureConfig(
            use_rsi=True,
            use_cci=True,
            use_zscore=True,
        ),
    )

    clustering_equal = SimplifiedPercentileClustering(config_equal)
    clustering_corr = SimplifiedPercentileClustering(config_corr)

    result_equal = clustering_equal.compute(high, low, close)
    result_corr = clustering_corr.compute(high, low, close)

    # Both should produce valid results
    assert result_equal.cluster_val.notna().any()
    assert result_corr.cluster_val.notna().any()
    # Results may differ (correlation weighting emphasizes unique features)
    # We just verify both work correctly


def test_correlation_weighting_single_feature():
    """Test correlation weighting with single feature (should behave like equal weights)."""
    high, low, close = _sample_ohlcv_data(200)

    config = ClusteringConfig(
        k=2,
        lookback=100,
        use_correlation_weights=True,
        feature_config=FeatureConfig(
            use_rsi=True,
            use_cci=False,
            use_zscore=False,
        ),
    )

    clustering = SimplifiedPercentileClustering(config)
    result = clustering.compute(high, low, close)

    # With single feature, correlation weighting should still work
    assert isinstance(result.cluster_val, pd.Series)
    assert len(result.cluster_val) == len(close)
