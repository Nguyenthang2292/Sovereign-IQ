"""
Tests for validation utilities.
"""

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
)
from modules.simplified_percentile_clustering.core.features import FeatureConfig
from modules.simplified_percentile_clustering.utils.validation import (
    validate_clustering_config,
    validate_feature_config,
    validate_input_data,
)


def test_validate_clustering_config_valid():
    """Test validation with valid ClusteringConfig."""
    config = ClusteringConfig(
        k=2,
        lookback=1000,
        p_low=5.0,
        p_high=95.0,
        main_plot="Clusters",
    )
    # Should not raise
    validate_clustering_config(config)


def test_validate_clustering_config_invalid_k():
    """Test validation with invalid k."""
    with pytest.raises(ValueError, match="k must be 2 or 3"):
        ClusteringConfig(k=5, lookback=1000)


def test_validate_clustering_config_invalid_percentiles():
    """Test validation with invalid percentiles."""
    # p_low >= p_high
    with pytest.raises(ValueError, match="Percentiles must satisfy"):
        ClusteringConfig(k=2, p_low=95.0, p_high=5.0)

    # p_low <= 0
    with pytest.raises(ValueError, match="Percentiles must satisfy"):
        ClusteringConfig(k=2, p_low=0.0, p_high=95.0)

    # p_high >= 100
    with pytest.raises(ValueError, match="Percentiles must satisfy"):
        ClusteringConfig(k=2, p_low=5.0, p_high=100.0)


def test_validate_clustering_config_invalid_lookback():
    """Test validation with invalid lookback."""
    with pytest.raises(ValueError, match="lookback must be at least 10"):
        ClusteringConfig(k=2, lookback=5)


def test_validate_clustering_config_invalid_main_plot():
    """Test validation with invalid main_plot."""
    with pytest.raises(ValueError, match="main_plot must be one of"):
        ClusteringConfig(k=2, main_plot="Invalid")


def test_validate_clustering_config_with_feature_config():
    """Test validation with nested FeatureConfig."""
    feature_config = FeatureConfig(use_rsi=True)
    config = ClusteringConfig(k=2, feature_config=feature_config)
    # Should not raise if feature_config is valid
    validate_clustering_config(config)


def test_validate_feature_config_valid():
    """Test validation with valid FeatureConfig."""
    config = FeatureConfig(
        use_rsi=True,
        rsi_len=14,
        use_cci=True,
        cci_len=20,
    )
    # Should not raise
    validate_feature_config(config)


def test_validate_feature_config_invalid_length():
    """Test validation with invalid feature length."""
    # Length < 1
    with pytest.raises(ValueError, match="rsi_len must be at least 1"):
        FeatureConfig(rsi_len=0)

    # Length > 1000
    with pytest.raises(ValueError, match="rsi_len must be at most 1000"):
        FeatureConfig(rsi_len=2000)


def test_validate_feature_config_invalid_mar_type():
    """Test validation with invalid mar_type."""
    with pytest.raises(ValueError, match="mar_type must be 'SMA' or 'EMA'"):
        FeatureConfig(use_mar=True, mar_type="Invalid")


def test_validate_feature_config_no_features_enabled():
    """Test validation when no features are enabled."""
    with pytest.raises(ValueError, match="At least one feature must be enabled"):
        FeatureConfig(
            use_rsi=False,
            use_cci=False,
            use_fisher=False,
            use_dmi=False,
            use_zscore=False,
            use_mar=False,
        )


def test_validate_strategy_config_cluster_transition():
    """Test validation for ClusterTransitionConfig."""
    # Valid config shouldn't raise
    ClusterTransitionConfig(
        min_signal_strength=0.3,
        min_rel_pos_change=0.1,
    )

    # Invalid min_signal_strength
    with pytest.raises(ValueError, match="min_signal_strength must be in"):
        ClusterTransitionConfig(min_signal_strength=1.5)

    # Invalid min_rel_pos_change
    with pytest.raises(ValueError, match="min_rel_pos_change must be in"):
        ClusterTransitionConfig(min_rel_pos_change=-0.1)


def test_validate_strategy_config_regime_following():
    """Test validation for RegimeFollowingConfig."""
    # Valid config shouldn't raise
    RegimeFollowingConfig(
        min_regime_strength=0.7,
        min_cluster_duration=2,
        momentum_period=5,
    )

    # Invalid min_regime_strength
    with pytest.raises(ValueError, match="min_regime_strength must be in"):
        RegimeFollowingConfig(min_regime_strength=2.0)

    # Invalid min_cluster_duration
    with pytest.raises(ValueError, match="min_cluster_duration must be at least 1"):
        RegimeFollowingConfig(min_cluster_duration=0)

    # Invalid momentum_period
    with pytest.raises(ValueError, match="momentum_period must be at least 1"):
        RegimeFollowingConfig(momentum_period=0)


def test_validate_strategy_config_mean_reversion():
    """Test validation for MeanReversionConfig."""
    # Valid config shouldn't raise
    MeanReversionConfig(
        extreme_threshold=0.2,
        min_extreme_duration=3,
        reversal_lookback=3,
    )

    # Invalid extreme_threshold
    with pytest.raises(ValueError, match="extreme_threshold must be in"):
        MeanReversionConfig(extreme_threshold=1.5)

    # Invalid min_extreme_duration
    with pytest.raises(ValueError, match="min_extreme_duration must be at least 1"):
        MeanReversionConfig(min_extreme_duration=0)


def test_validate_input_data_valid():
    """Test validation with valid input data."""
    high = pd.Series([100.0, 101.0, 102.0])
    low = pd.Series([99.0, 100.0, 101.0])
    close = pd.Series([99.5, 100.5, 101.5])

    # Should not raise
    validate_input_data(high=high, low=low, close=close, require_all=True)


def test_validate_input_data_empty_series():
    """Test validation with empty series."""
    high = pd.Series(dtype=float)
    low = pd.Series([99.0, 100.0])
    close = pd.Series([99.5, 100.5])

    with pytest.raises(ValueError, match="high series is empty"):
        validate_input_data(high=high, low=low, close=close)


def test_validate_input_data_all_nan():
    """Test validation with all NaN values."""
    high = pd.Series([np.nan, np.nan, np.nan])
    low = pd.Series([99.0, 100.0, 101.0])
    close = pd.Series([99.5, 100.5, 101.5])

    with pytest.raises(ValueError, match="high series contains only NaN values"):
        validate_input_data(high=high, low=low, close=close)


def test_validate_input_data_negative_prices():
    """Test validation with negative prices."""
    high = pd.Series([100.0, -1.0, 102.0])
    low = pd.Series([99.0, 100.0, 101.0])
    close = pd.Series([99.5, 100.5, 101.5])

    with pytest.raises(ValueError, match="high series contains negative values"):
        validate_input_data(high=high, low=low, close=close)


def test_validate_input_data_high_less_than_low():
    """Test validation when high < low."""
    high = pd.Series([100.0, 99.0, 102.0])  # high[1] < low[1]
    low = pd.Series([99.0, 100.0, 101.0])
    close = pd.Series([99.5, 100.5, 101.5])

    with pytest.raises(ValueError, match="high values must be >= low values"):
        validate_input_data(high=high, low=low, close=close)


def test_validate_input_data_inconsistent_indices():
    """Test validation with inconsistent indices."""
    high = pd.Series([100.0, 101.0], index=[0, 1])
    low = pd.Series([99.0, 100.0], index=[0, 2])  # Different index
    close = pd.Series([99.5, 100.5], index=[0, 1])

    # Should catch index mismatch before trying to compare values
    with pytest.raises(ValueError):
        validate_input_data(high=high, low=low, close=close)


def test_validate_input_data_wrong_type():
    """Test validation with wrong data type."""
    high = [100.0, 101.0, 102.0]  # List instead of Series
    low = pd.Series([99.0, 100.0, 101.0])
    close = pd.Series([99.5, 100.5, 101.5])

    with pytest.raises(TypeError, match="high must be a pandas Series"):
        validate_input_data(high=high, low=low, close=close)


def test_validate_input_data_require_all_false():
    """Test validation with require_all=False."""
    high = pd.Series([100.0, 101.0, 102.0])
    # Missing low and close
    validate_input_data(high=high, require_all=False)  # Should not raise


def test_validate_input_data_require_all_true():
    """Test validation with require_all=True but missing data."""
    high = pd.Series([100.0, 101.0, 102.0])
    # Missing low and close
    with pytest.raises(ValueError, match="high, low, and close must all be provided"):
        validate_input_data(high=high, require_all=True)


def test_config_post_init_validation():
    """Test that configs validate automatically via __post_init__."""
    # ClusteringConfig
    with pytest.raises(ValueError):
        ClusteringConfig(k=5)

    # FeatureConfig
    with pytest.raises(ValueError):
        FeatureConfig(use_rsi=False, use_cci=False, use_fisher=False, use_dmi=False, use_zscore=False, use_mar=False)

    # ClusterTransitionConfig
    with pytest.raises(ValueError):
        ClusterTransitionConfig(min_signal_strength=2.0)
