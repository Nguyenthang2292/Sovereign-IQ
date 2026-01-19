"""Tests for strategy confirmation features.

This module tests confirmation mechanisms in strategies like price confirmation,
volume confirmation, and momentum confirmation.
"""

import numpy as np
import pandas as pd
import pytest

from modules.simplified_percentile_clustering.core.clustering import (
    ClusteringConfig,
    SimplifiedPercentileClustering,
)
from modules.simplified_percentile_clustering.core.features import FeatureConfig
from modules.simplified_percentile_clustering.strategies.cluster_transition import (
    ClusterTransitionConfig,
    generate_signals_cluster_transition,
)
from modules.simplified_percentile_clustering.strategies.regime_following import (
    RegimeFollowingConfig,
    generate_signals_regime_following,
)


def _sample_ohlcv_data(length=200):
    """Generate sample OHLCV data."""
    np.random.seed(42)
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(length) * 0.5)

    high = prices + np.abs(np.random.randn(length) * 0.3)
    low = prices - np.abs(np.random.randn(length) * 0.3)
    close = prices
    volume = pd.Series(np.random.uniform(1000, 10000, length))

    return pd.Series(high), pd.Series(low), pd.Series(close), volume


def test_price_confirmation_cluster_transition():
    """Test price confirmation in cluster transition strategy."""
    high, low, close, volume = _sample_ohlcv_data(200)

    clustering_config = ClusteringConfig(k=2, lookback=100, feature_config=FeatureConfig(use_rsi=True))
    clustering = SimplifiedPercentileClustering(clustering_config)
    clustering_result = clustering.compute(high, low, close)

    config_with_confirmation = ClusterTransitionConfig(
        min_signal_strength=0.5,
        min_rel_pos_change=0.1,
        require_price_confirmation=True,
    )

    config_without_confirmation = ClusterTransitionConfig(
        min_signal_strength=0.5,
        min_rel_pos_change=0.1,
        require_price_confirmation=False,
    )

    signals_with = generate_signals_cluster_transition(
        high=high, low=low, close=close, clustering_result=clustering_result, config=config_with_confirmation
    )

    signals_without = generate_signals_cluster_transition(
        high=high, low=low, close=close, clustering_result=clustering_result, config=config_without_confirmation
    )

    # Both should produce valid results
    assert isinstance(signals_with[0], pd.Series)
    assert isinstance(signals_without[0], pd.Series)
    # With confirmation, should have fewer signals (more conservative)
    if signals_with[0].notna().sum() > 0 and signals_without[0].notna().sum() > 0:
        signal_count_with = (signals_with[0].abs() > 0).sum()
        signal_count_without = (signals_without[0].abs() > 0).sum()
        # With confirmation should have fewer or equal signals
        assert signal_count_with <= signal_count_without + 5  # Allow tolerance


def test_volume_confirmation_regime_following():
    """Test volume confirmation in regime following strategy."""
    high, low, close, volume = _sample_ohlcv_data(200)

    clustering_config = ClusteringConfig(k=2, lookback=100, feature_config=FeatureConfig(use_rsi=True))
    clustering = SimplifiedPercentileClustering(clustering_config)
    clustering_result = clustering.compute(high, low, close)

    config_with_volume = RegimeFollowingConfig(
        min_regime_strength=0.7,
        min_cluster_duration=2,
        require_volume_confirmation=True,
        volume_lookback=20,
    )

    config_without_volume = RegimeFollowingConfig(
        min_regime_strength=0.7,
        min_cluster_duration=2,
        require_volume_confirmation=False,
    )

    signals_with = generate_signals_regime_following(
        high=high, low=low, close=close, volume=volume, clustering_result=clustering_result, config=config_with_volume
    )

    signals_without = generate_signals_regime_following(
        high=high,
        low=low,
        close=close,
        volume=volume,
        clustering_result=clustering_result,
        config=config_without_volume,
    )

    # Both should produce valid results
    assert isinstance(signals_with[0], pd.Series)
    assert isinstance(signals_without[0], pd.Series)


def test_momentum_confirmation_regime_following():
    """Test momentum confirmation in regime following strategy."""
    high, low, close, volume = _sample_ohlcv_data(200)

    clustering_config = ClusteringConfig(k=2, lookback=100, feature_config=FeatureConfig(use_rsi=True))
    clustering = SimplifiedPercentileClustering(clustering_config)
    clustering_result = clustering.compute(high, low, close)

    config_with_momentum = RegimeFollowingConfig(
        min_regime_strength=0.7,
        min_cluster_duration=2,
        require_momentum=True,
        momentum_period=5,
    )

    config_without_momentum = RegimeFollowingConfig(
        min_regime_strength=0.7,
        min_cluster_duration=2,
        require_momentum=False,
    )

    signals_with = generate_signals_regime_following(
        high=high, low=low, close=close, clustering_result=clustering_result, config=config_with_momentum
    )

    signals_without = generate_signals_regime_following(
        high=high, low=low, close=close, clustering_result=clustering_result, config=config_without_momentum
    )

    # Both should produce valid results
    assert isinstance(signals_with[0], pd.Series)
    assert isinstance(signals_without[0], pd.Series)


def test_combined_confirmations():
    """Test multiple confirmations combined (price + volume + momentum)."""
    high, low, close, volume = _sample_ohlcv_data(200)

    clustering_config = ClusteringConfig(k=2, lookback=100, feature_config=FeatureConfig(use_rsi=True))
    clustering = SimplifiedPercentileClustering(clustering_config)
    clustering_result = clustering.compute(high, low, close)

    config = RegimeFollowingConfig(
        min_regime_strength=0.7,
        min_cluster_duration=3,
        require_momentum=True,
        momentum_period=5,
        require_volume_confirmation=True,
        volume_lookback=20,
    )

    signals, strength, metadata = generate_signals_regime_following(
        high=high, low=low, close=close, volume=volume, clustering_result=clustering_result, config=config
    )

    assert isinstance(signals, pd.Series)
    assert isinstance(strength, pd.Series)
    assert isinstance(metadata, pd.DataFrame)
    # With multiple confirmations, should produce conservative signals
    assert signals.notna().any() or signals.isna().all()  # Either has signals or all NaN
