"""
Validation utilities for Simplified Percentile Clustering.

Provides validation functions for configurations and input data.
"""

from typing import Optional, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from modules.simplified_percentile_clustering.core.clustering import (
        ClusteringConfig,
    )
    from modules.simplified_percentile_clustering.core.features import (
        FeatureConfig,
    )
    from modules.simplified_percentile_clustering.strategies.cluster_transition import (
        ClusterTransitionConfig,
    )
    from modules.simplified_percentile_clustering.strategies.regime_following import (
        RegimeFollowingConfig,
    )
    from modules.simplified_percentile_clustering.strategies.mean_reversion import (
        MeanReversionConfig,
    )


def validate_clustering_config(config: "ClusteringConfig") -> None:
    """
    Validate ClusteringConfig parameters.
    
    Args:
        config: ClusteringConfig instance to validate
        
    Raises:
        ValueError: If any parameter is invalid
    """
    if config.k not in [2, 3]:
        raise ValueError(f"k must be 2 or 3, got {config.k}")
    
    if not (0 < config.p_low < config.p_high < 100):
        raise ValueError(
            f"Percentiles must satisfy 0 < p_low ({config.p_low}) < p_high ({config.p_high}) < 100"
        )
    
    if config.lookback < 10:
        raise ValueError(f"lookback must be at least 10, got {config.lookback}")
    
    valid_main_plots = ["Clusters", "RSI", "CCI", "Fisher", "DMI", "Z-Score", "MAR"]
    if config.main_plot not in valid_main_plots:
        raise ValueError(
            f"main_plot must be one of {valid_main_plots}, got {config.main_plot}"
        )
    
    if config.feature_config is not None:
        validate_feature_config(config.feature_config)


def validate_feature_config(config: "FeatureConfig") -> None:
    """
    Validate FeatureConfig parameters.
    
    Args:
        config: FeatureConfig instance to validate
        
    Raises:
        ValueError: If any parameter is invalid
    """
    # Validate lengths
    for name, length in [
        ("rsi_len", config.rsi_len),
        ("cci_len", config.cci_len),
        ("fisher_len", config.fisher_len),
        ("dmi_len", config.dmi_len),
        ("zscore_len", config.zscore_len),
        ("mar_len", config.mar_len),
    ]:
        if length < 1:
            raise ValueError(f"{name} must be at least 1, got {length}")
        if length > 1000:
            raise ValueError(f"{name} must be at most 1000, got {length}")
    
    # Validate MAR type
    if config.mar_type not in ["SMA", "EMA"]:
        raise ValueError(f"mar_type must be 'SMA' or 'EMA', got {config.mar_type}")
    
    # Check if at least one feature is enabled
    if not any([
        config.use_rsi,
        config.use_cci,
        config.use_fisher,
        config.use_dmi,
        config.use_zscore,
        config.use_mar,
    ]):
        raise ValueError("At least one feature must be enabled")


def validate_strategy_config(config) -> None:
    """Validate strategy configuration."""
    # Import here to avoid circular import
    from modules.simplified_percentile_clustering.config.cluster_transition_config import (
        ClusterTransitionConfig,
    )
    from modules.simplified_percentile_clustering.config.regime_following_config import (
        RegimeFollowingConfig,
    )
    from modules.simplified_percentile_clustering.config.mean_reversion_config import (
        MeanReversionConfig,
    )
    """
    Validate strategy configuration.
    
    Args:
        config: Strategy config instance (ClusterTransitionConfig, RegimeFollowingConfig, or MeanReversionConfig)
        
    Raises:
        ValueError: If any parameter is invalid
    """
    if isinstance(config, ClusterTransitionConfig):
        if not (0.0 <= config.min_signal_strength <= 1.0):
            raise ValueError(
                f"min_signal_strength must be in [0.0, 1.0], got {config.min_signal_strength}"
            )
        if not (0.0 <= config.min_rel_pos_change <= 1.0):
            raise ValueError(
                f"min_rel_pos_change must be in [0.0, 1.0], got {config.min_rel_pos_change}"
            )
        if config.clustering_config is not None:
            validate_clustering_config(config.clustering_config)
    
    elif isinstance(config, RegimeFollowingConfig):
        if not (0.0 <= config.min_regime_strength <= 1.0):
            raise ValueError(
                f"min_regime_strength must be in [0.0, 1.0], got {config.min_regime_strength}"
            )
        if config.min_cluster_duration < 1:
            raise ValueError(
                f"min_cluster_duration must be at least 1, got {config.min_cluster_duration}"
            )
        if config.momentum_period < 1:
            raise ValueError(
                f"momentum_period must be at least 1, got {config.momentum_period}"
            )
        if not (0.0 <= config.bullish_real_clust_threshold <= 1.0):
            raise ValueError(
                f"bullish_real_clust_threshold must be in [0.0, 1.0], got {config.bullish_real_clust_threshold}"
            )
        if not (0.0 <= config.bearish_real_clust_threshold <= 1.0):
            raise ValueError(
                f"bearish_real_clust_threshold must be in [0.0, 1.0], got {config.bearish_real_clust_threshold}"
            )
        if config.clustering_config is not None:
            validate_clustering_config(config.clustering_config)
    
    elif isinstance(config, MeanReversionConfig):
        if not (0.0 <= config.extreme_threshold <= 1.0):
            raise ValueError(
                f"extreme_threshold must be in [0.0, 1.0], got {config.extreme_threshold}"
            )
        if config.min_extreme_duration < 1:
            raise ValueError(
                f"min_extreme_duration must be at least 1, got {config.min_extreme_duration}"
            )
        if config.reversal_lookback < 1:
            raise ValueError(
                f"reversal_lookback must be at least 1, got {config.reversal_lookback}"
            )
        if not (0.0 <= config.min_signal_strength <= 1.0):
            raise ValueError(
                f"min_signal_strength must be in [0.0, 1.0], got {config.min_signal_strength}"
            )
        if config.clustering_config is not None:
            validate_clustering_config(config.clustering_config)


def validate_input_data(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    require_all: bool = True,
) -> None:
    """
    Validate input OHLCV data.
    
    Args:
        high: High price series (optional)
        low: Low price series (optional)
        close: Close price series (optional)
        require_all: If True, all series must be provided and valid
        
    Raises:
        ValueError: If data is invalid
        TypeError: If data types are incorrect
    """
    series_list = []
    if high is not None:
        series_list.append(("high", high))
    if low is not None:
        series_list.append(("low", low))
    if close is not None:
        series_list.append(("close", close))
    
    if require_all and len(series_list) < 3:
        raise ValueError("high, low, and close must all be provided when require_all=True")
    
    if len(series_list) == 0:
        raise ValueError("At least one series must be provided")
    
    # Check types
    for name, series in series_list:
        if not isinstance(series, pd.Series):
            raise TypeError(f"{name} must be a pandas Series, got {type(series)}")
        
        if len(series) == 0:
            raise ValueError(f"{name} series is empty")
        
        # Check for all NaN
        if series.isna().all():
            raise ValueError(f"{name} series contains only NaN values")
        
        # Check for negative values (prices should be positive)
        if name in ["high", "low", "close"]:
            if (series < 0).any():
                raise ValueError(f"{name} series contains negative values")
        
    # Check index consistency if multiple series provided (must be done before value comparisons)
    if len(series_list) > 1:
        indices = [s.index for _, s in series_list]
        try:
            if not all(idx.equals(indices[0]) for idx in indices):
                raise ValueError("All series must have the same index")
        except (ValueError, AttributeError) as e:
            # If comparison fails due to different indices, raise our error
            if "Can only compare identically-labeled" in str(e) or "All series must have the same index" in str(e):
                raise ValueError("All series must have the same index") from e
            raise
    
    # Check for high >= low if both provided (after index check)
    if high is not None and low is not None:
        if len(high) == len(low):
            try:
                invalid = (high < low) & high.notna() & low.notna()
                if invalid.any():
                    raise ValueError(
                        f"high values must be >= low values. Found {invalid.sum()} invalid rows"
                    )
            except ValueError as e:
                # If error is about index mismatch, re-raise as our error
                if "Can only compare identically-labeled" in str(e):
                    raise ValueError("All series must have the same index") from e
                raise

