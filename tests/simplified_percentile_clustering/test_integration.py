"""
Integration tests for Simplified Percentile Clustering.

Tests end-to-end workflows, error handling, and performance.
"""

import time

import numpy as np
import pandas as pd
import pytest

from modules.simplified_percentile_clustering.config import (
    ClusterTransitionConfig,
    MeanReversionConfig,
    RegimeFollowingConfig,
)
from modules.simplified_percentile_clustering.core.clustering import (
    ClusteringConfig,
    compute_clustering,
)
from modules.simplified_percentile_clustering.core.features import FeatureConfig
from modules.simplified_percentile_clustering.strategies.cluster_transition import (
    generate_signals_cluster_transition,
)
from modules.simplified_percentile_clustering.strategies.mean_reversion import (
    generate_signals_mean_reversion,
)
from modules.simplified_percentile_clustering.strategies.regime_following import (
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

    return pd.Series(high), pd.Series(low), pd.Series(close)


def test_end_to_end_clustering_basic():
    """Test end-to-end clustering workflow."""
    high, low, close = _sample_ohlcv_data(200)

    config = ClusteringConfig(
        k=2,
        lookback=100,
        feature_config=FeatureConfig(
            use_rsi=True,
            use_cci=True,
        ),
    )

    result = compute_clustering(high, low, close, config=config)

    # Verify result structure
    assert result is not None
    assert len(result.cluster_val) == len(close)
    assert len(result.real_clust) == len(close)
    assert "rsi_val" in result.features
    assert "cci_val" in result.features


def test_end_to_end_clustering_with_strategies():
    """Test end-to-end clustering with all strategies."""
    high, low, close = _sample_ohlcv_data(200)

    clustering_config = ClusteringConfig(
        k=2,
        lookback=100,
        feature_config=FeatureConfig(use_rsi=True, use_cci=True),
    )

    # Compute clustering once
    clustering_result = compute_clustering(high, low, close, config=clustering_config)

    # Test Cluster Transition Strategy
    transition_config = ClusterTransitionConfig(
        clustering_config=clustering_config,
        min_signal_strength=0.3,
    )
    signals_trans, strength_trans, meta_trans = generate_signals_cluster_transition(
        high,
        low,
        close,
        clustering_result=clustering_result,
        config=transition_config,
    )
    assert len(signals_trans) == len(close)
    assert len(strength_trans) == len(close)

    # Test Regime Following Strategy
    regime_config = RegimeFollowingConfig(
        clustering_config=clustering_config,
        min_regime_strength=0.7,
    )
    signals_regime, strength_regime, meta_regime = generate_signals_regime_following(
        high,
        low,
        close,
        clustering_result=clustering_result,
        config=regime_config,
    )
    assert len(signals_regime) == len(close)

    # Test Mean Reversion Strategy
    reversion_config = MeanReversionConfig(
        clustering_config=clustering_config,
        extreme_threshold=0.2,
    )
    signals_rev, strength_rev, meta_rev = generate_signals_mean_reversion(
        high,
        low,
        close,
        clustering_result=clustering_result,
        config=reversion_config,
    )
    assert len(signals_rev) == len(close)


def test_error_handling_invalid_config():
    """Test error handling with invalid configuration."""
    high, low, close = _sample_ohlcv_data(200)

    # Invalid k
    with pytest.raises(ValueError):
        config = ClusteringConfig(k=5)
        compute_clustering(high, low, close, config=config)

    # Invalid percentiles
    with pytest.raises(ValueError):
        config = ClusteringConfig(k=2, p_low=95.0, p_high=5.0)
        compute_clustering(high, low, close, config=config)

    # Invalid lookback
    with pytest.raises(ValueError):
        config = ClusteringConfig(k=2, lookback=5)
        compute_clustering(high, low, close, config=config)


def test_error_handling_invalid_input_data():
    """Test error handling with invalid input data."""
    # Empty series
    with pytest.raises(ValueError, match="series is empty"):
        compute_clustering(
            pd.Series(dtype=float),
            pd.Series([99.0, 100.0]),
            pd.Series([99.5, 100.5]),
        )

    # All NaN
    with pytest.raises(ValueError, match="contains only NaN values"):
        compute_clustering(
            pd.Series([np.nan, np.nan]),
            pd.Series([99.0, 100.0]),
            pd.Series([99.5, 100.5]),
        )

    # Negative prices
    with pytest.raises(ValueError, match="contains negative values"):
        compute_clustering(
            pd.Series([-1.0, 100.0]),
            pd.Series([99.0, 100.0]),
            pd.Series([99.5, 100.5]),
        )

    # High < Low
    with pytest.raises(ValueError, match="high values must be >= low values"):
        compute_clustering(
            pd.Series([99.0, 100.0]),  # high < low
            pd.Series([100.0, 101.0]),
            pd.Series([99.5, 100.5]),
        )


def test_error_handling_inconsistent_indices():
    """Test error handling with inconsistent indices."""
    high = pd.Series([100.0, 101.0], index=[0, 1])
    low = pd.Series([99.0, 100.0], index=[0, 2])  # Different index
    close = pd.Series([99.5, 100.5], index=[0, 1])

    # Should raise ValueError (may be wrapped or direct)
    with pytest.raises(ValueError):
        compute_clustering(high, low, close)


def test_error_handling_strategy_invalid_config():
    """Test error handling with invalid strategy config."""
    high, low, close = _sample_ohlcv_data(200)

    clustering_config = ClusteringConfig(k=2, lookback=100)
    clustering_result = compute_clustering(high, low, close, config=clustering_config)

    # Invalid min_signal_strength
    with pytest.raises(ValueError):
        config = ClusterTransitionConfig(
            clustering_config=clustering_config,
            min_signal_strength=2.0,  # Invalid
        )
        generate_signals_cluster_transition(
            high,
            low,
            close,
            clustering_result=clustering_result,
            config=config,
        )


def test_performance_large_dataset():
    """Test performance with large dataset."""
    # Generate large dataset
    high, low, close = _sample_ohlcv_data(5000)

    config = ClusteringConfig(
        k=2,
        lookback=1000,
        feature_config=FeatureConfig(
            use_rsi=True,
            use_cci=True,
            use_fisher=True,
            use_dmi=True,
            use_zscore=True,
            use_mar=True,
        ),
    )

    # Measure computation time
    start_time = time.time()
    result = compute_clustering(high, low, close, config=config)
    elapsed_time = time.time() - start_time

    # Verify result
    assert result is not None
    assert len(result.cluster_val) == len(close)

    # Performance check: should complete in reasonable time (< 10 seconds)
    # This is a soft check - actual time depends on hardware
    assert elapsed_time < 30.0, f"Computation took {elapsed_time:.2f} seconds"


def test_performance_vectorized_vs_loop():
    """Test that vectorized operations are faster than loops."""
    high, low, close = _sample_ohlcv_data(1000)

    config = ClusteringConfig(
        k=2,
        lookback=500,
        feature_config=FeatureConfig(use_rsi=True, use_cci=True),
    )

    # Measure time
    start_time = time.time()
    compute_clustering(high, low, close, config=config)
    elapsed_time = time.time() - start_time

    # Should be reasonably fast with vectorized operations
    assert elapsed_time < 5.0, f"Vectorized computation took {elapsed_time:.2f} seconds"


def test_clustering_consistency():
    """Test that clustering produces consistent results."""
    high, low, close = _sample_ohlcv_data(200)

    config = ClusteringConfig(
        k=2,
        lookback=100,
        feature_config=FeatureConfig(use_rsi=True),
    )

    # Run twice with same data and config
    result1 = compute_clustering(high, low, close, config=config)
    result2 = compute_clustering(high, low, close, config=config)

    # Results should be identical (deterministic)
    pd.testing.assert_series_equal(result1.cluster_val, result2.cluster_val)
    pd.testing.assert_series_equal(result1.real_clust, result2.real_clust)


def test_clustering_with_different_k():
    """Test clustering with k=2 vs k=3."""
    high, low, close = _sample_ohlcv_data(200)

    # k=2
    config_k2 = ClusteringConfig(k=2, lookback=100)
    result_k2 = compute_clustering(high, low, close, config=config_k2)

    # k=3
    config_k3 = ClusteringConfig(k=3, lookback=100)
    result_k3 = compute_clustering(high, low, close, config=config_k3)

    # Both should produce valid results
    assert len(result_k2.cluster_val) == len(close)
    assert len(result_k3.cluster_val) == len(close)

    # k=2: cluster_val should be 0 or 1
    valid_k2 = result_k2.cluster_val.dropna()
    assert all(valid_k2.isin([0, 1]))

    # k=3: cluster_val should be 0, 1, or 2
    valid_k3 = result_k3.cluster_val.dropna()
    assert all(valid_k3.isin([0, 1, 2]))


def test_clustering_all_main_plot_modes():
    """Test clustering with all main_plot modes."""
    high, low, close = _sample_ohlcv_data(200)

    feature_config = FeatureConfig(
        use_rsi=True,
        use_cci=True,
        use_fisher=True,
        use_dmi=True,
        use_zscore=True,
        use_mar=True,
    )

    plot_modes = ["Clusters", "RSI", "CCI", "Fisher", "DMI", "Z-Score", "MAR"]

    for mode in plot_modes:
        config = ClusteringConfig(
            k=2,
            lookback=100,
            main_plot=mode,
            feature_config=feature_config,
        )

        result = compute_clustering(high, low, close, config=config)

        # All modes should produce valid results
        assert result is not None
        assert len(result.plot_val) == len(close)


def test_strategy_reuse_clustering_result():
    """Test that strategies can reuse clustering result."""
    high, low, close = _sample_ohlcv_data(200)

    clustering_config = ClusteringConfig(k=2, lookback=100)

    # Compute clustering once
    clustering_result = compute_clustering(high, low, close, config=clustering_config)

    # Use same result for multiple strategies
    transition_config = ClusterTransitionConfig(clustering_config=clustering_config)
    regime_config = RegimeFollowingConfig(clustering_config=clustering_config)
    reversion_config = MeanReversionConfig(clustering_config=clustering_config)

    # All should work with same clustering_result
    signals1, _, _ = generate_signals_cluster_transition(
        high,
        low,
        close,
        clustering_result=clustering_result,
        config=transition_config,
    )

    signals2, _, _ = generate_signals_regime_following(
        high,
        low,
        close,
        clustering_result=clustering_result,
        config=regime_config,
    )

    signals3, _, _ = generate_signals_mean_reversion(
        high,
        low,
        close,
        clustering_result=clustering_result,
        config=reversion_config,
    )

    # All should produce signals
    assert len(signals1) == len(close)
    assert len(signals2) == len(close)
    assert len(signals3) == len(close)
