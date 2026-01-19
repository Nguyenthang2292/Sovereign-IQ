"""Tests for multi-timeframe clustering analysis.

This module tests the compute_multi_timeframe_clustering function that
analyzes clustering across multiple timeframes for stronger signals.
"""

import numpy as np
import pandas as pd
import pytest

from modules.simplified_percentile_clustering.core.clustering import (
    ClusteringConfig,
    compute_multi_timeframe_clustering,
)
from modules.simplified_percentile_clustering.core.features import FeatureConfig


def _sample_ohlcv_data(length=500):
    """Generate sample OHLCV data with datetime index."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=length, freq="15min")
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(length) * 0.5)

    high = prices + np.abs(np.random.randn(length) * 0.3)
    low = prices - np.abs(np.random.randn(length) * 0.3)
    close = prices

    return pd.Series(high, index=dates), pd.Series(low, index=dates), pd.Series(close, index=dates)


def test_multi_timeframe_basic():
    """Test basic multi-timeframe clustering."""
    high, low, close = _sample_ohlcv_data(500)

    config = ClusteringConfig(
        k=2,
        lookback=50,
        feature_config=FeatureConfig(use_rsi=True),
    )

    result = compute_multi_timeframe_clustering(
        high=high,
        low=low,
        close=close,
        timeframes=["1h", "4h"],
        config=config,
    )

    assert "cluster_val" in result
    assert "mtf_agreement" in result
    assert "aligned_cluster" in result
    assert "timeframe_results" in result
    assert isinstance(result["cluster_val"], pd.Series)
    assert len(result["cluster_val"]) == len(close)


def test_multi_timeframe_three_timeframes():
    """Test multi-timeframe with three timeframes."""
    high, low, close = _sample_ohlcv_data(1000)

    config = ClusteringConfig(k=2, lookback=50, feature_config=FeatureConfig(use_rsi=True))

    result = compute_multi_timeframe_clustering(
        high=high,
        low=low,
        close=close,
        timeframes=["1h", "4h", "1d"],
        config=config,
    )

    assert "cluster_val" in result
    assert "mtf_agreement" in result
    assert isinstance(result["mtf_agreement"], float)
    assert 0.0 <= result["mtf_agreement"] <= 1.0
    assert len(result["timeframe_results"]) >= 1


def test_multi_timeframe_agreement_calculation():
    """Test that mtf_agreement is calculated correctly."""
    high, low, close = _sample_ohlcv_data(500)

    config = ClusteringConfig(k=2, lookback=50, feature_config=FeatureConfig(use_rsi=True))

    result = compute_multi_timeframe_clustering(
        high=high,
        low=low,
        close=close,
        timeframes=["1h", "4h"],
        config=config,
    )

    # Agreement should be between 0 and 1
    assert 0.0 <= result["mtf_agreement"] <= 1.0
    # If we have results for both timeframes, agreement should be calculable
    if len(result["timeframe_results"]) >= 2:
        assert result["mtf_agreement"] >= 0.0


def test_multi_timeframe_aligned_cluster():
    """Test aligned_cluster calculation when timeframes agree."""
    high, low, close = _sample_ohlcv_data(500)

    config = ClusteringConfig(k=2, lookback=50, feature_config=FeatureConfig(use_rsi=True))

    result = compute_multi_timeframe_clustering(
        high=high,
        low=low,
        close=close,
        timeframes=["1h", "4h"],
        require_alignment=True,
        config=config,
    )

    assert isinstance(result["aligned_cluster"], pd.Series)
    assert len(result["aligned_cluster"]) == len(close)
    # Aligned cluster should be NaN where timeframes don't agree
    # or valid cluster value where they do agree


def test_multi_timeframe_invalid_timeframe():
    """Test error handling for invalid timeframe strings."""
    high, low, close = _sample_ohlcv_data(500)

    config = ClusteringConfig(k=2, lookback=50, feature_config=FeatureConfig(use_rsi=True))

    with pytest.raises(ValueError, match="Invalid timeframe"):
        compute_multi_timeframe_clustering(
            high=high,
            low=low,
            close=close,
            timeframes=["invalid_tf"],
            config=config,
        )
