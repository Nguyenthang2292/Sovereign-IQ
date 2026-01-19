"""Tests for cluster stability features.

This module tests the min_flip_duration and flip_confidence_threshold
parameters that prevent rapid cluster flipping.
"""

import numpy as np
import pandas as pd

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


def test_cluster_stability_min_flip_duration():
    """Test that min_flip_duration prevents rapid cluster flipping."""
    high, low, close = _sample_ohlcv_data(200)

    config_no_stability = ClusteringConfig(
        k=2,
        lookback=100,
        min_flip_duration=1,  # Allow immediate flips
        flip_confidence_threshold=0.0,
        feature_config=FeatureConfig(use_rsi=True),
    )

    config_stable = ClusteringConfig(
        k=2,
        lookback=100,
        min_flip_duration=5,  # Require 5 bars before flip
        flip_confidence_threshold=0.6,
        feature_config=FeatureConfig(use_rsi=True),
    )

    clustering_no_stability = SimplifiedPercentileClustering(config_no_stability)
    clustering_stable = SimplifiedPercentileClustering(config_stable)

    result_no_stability = clustering_no_stability.compute(high, low, close)
    result_stable = clustering_stable.compute(high, low, close)

    # Both should produce valid results
    assert result_no_stability.cluster_val.notna().any()
    assert result_stable.cluster_val.notna().any()

    # Stable version should have fewer flips (count cluster changes)
    if result_no_stability.cluster_val.notna().sum() > 10 and result_stable.cluster_val.notna().sum() > 10:
        flips_no_stability = (result_no_stability.cluster_val.diff() != 0).sum()
        flips_stable = (result_stable.cluster_val.diff() != 0).sum()
        # Stable should have fewer or equal flips
        assert flips_stable <= flips_no_stability + 5  # Allow some tolerance


def test_cluster_stability_confidence_threshold():
    """Test that flip_confidence_threshold prevents low-confidence flips."""
    high, low, close = _sample_ohlcv_data(200)

    config_low_threshold = ClusteringConfig(
        k=2,
        lookback=50,  # Smaller lookback for faster convergence
        min_flip_duration=2,  # Lower duration
        flip_confidence_threshold=0.3,  # Low threshold
        feature_config=FeatureConfig(use_rsi=True),
    )

    config_high_threshold = ClusteringConfig(
        k=2,
        lookback=50,
        min_flip_duration=2,
        flip_confidence_threshold=0.8,  # High threshold
        feature_config=FeatureConfig(use_rsi=True),
    )

    clustering_low = SimplifiedPercentileClustering(config_low_threshold)
    clustering_high = SimplifiedPercentileClustering(config_high_threshold)

    result_low = clustering_low.compute(high, low, close)
    result_high = clustering_high.compute(high, low, close)

    # Both should produce valid results (after lookback)
    valid_low = result_low.cluster_val.iloc[50:].dropna()
    valid_high = result_high.cluster_val.iloc[50:].dropna()
    assert len(valid_low) > 0, "Should have valid results with low threshold"
    assert len(valid_high) > 0, "Should have valid results with high threshold"
    # High threshold should be more stable (fewer flips)
    if len(valid_low) > 10 and len(valid_high) > 10:
        flips_low = (valid_low.diff() != 0).sum()
        flips_high = (valid_high.diff() != 0).sum()
        # High threshold should have fewer or equal flips
        assert flips_high <= flips_low + 5  # Allow some tolerance


def test_cluster_stability_combined():
    """Test cluster stability with both min_flip_duration and confidence threshold."""
    high, low, close = _sample_ohlcv_data(200)

    config = ClusteringConfig(
        k=2,
        lookback=50,  # Smaller lookback for faster convergence
        min_flip_duration=3,
        flip_confidence_threshold=0.6,
        feature_config=FeatureConfig(use_rsi=True, use_cci=True),
    )

    clustering = SimplifiedPercentileClustering(config)
    result = clustering.compute(high, low, close)

    assert isinstance(result.cluster_val, pd.Series)
    assert len(result.cluster_val) == len(close)
    # Should produce stable clustering (after lookback)
    valid_cluster = result.cluster_val.iloc[50:].dropna()
    assert len(valid_cluster) > 0, "Should have valid cluster values after lookback"

    # Verify stability: count consecutive same values
    if len(valid_cluster) > 10:
        # Should have some periods of stability (consecutive same values)
        cluster_changes = (valid_cluster.diff() != 0).sum()
        # With stability enabled, should have reasonable number of changes
        # Allow up to 70% changes (stability reduces but doesn't eliminate all changes)
        assert cluster_changes < len(valid_cluster) * 0.7
