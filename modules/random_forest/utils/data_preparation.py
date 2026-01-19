"""Data preparation utilities for Random Forest training.

This module provides utilities for preparing training data, including feature
engineering and target variable creation.
"""

from typing import Optional, Tuple, cast

import pandas as pd

from config import MIN_TRAINING_SAMPLES
from config.model_features import MODEL_FEATURES
from config.random_forest import RANDOM_FOREST_TARGET_HORIZON
from modules.common.core.indicator_engine import IndicatorConfig, IndicatorEngine, IndicatorProfile
from modules.common.ui.logging import (
    log_error,
    log_progress,
    log_warn,
)


def prepare_training_data(df: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Prepare feature data and target variable for model training.

    Args:
        df (pd.DataFrame): Input OHLCV data.

    Returns:
        Optional[Tuple[pd.DataFrame, pd.Series]]: Tuple (features, target) or None if error.
    """
    log_progress("Calculating features for training data...")
    # Use IndicatorEngine to generate features
    engine = IndicatorEngine(IndicatorConfig.for_profile(IndicatorProfile.CORE))
    df_with_features = engine.compute_features(df)
    if df_with_features.empty:
        log_error("Feature calculation resulted in an empty DataFrame.")
        return None

    # Apply Advanced Feature Engineering
    from modules.random_forest.utils.features import add_advanced_features, get_enhanced_feature_names

    df_with_features = add_advanced_features(df_with_features)

    log_progress("Creating target variable 'target' with advanced labeling...")
    # Use advanced labeling strategy (volatility-adjusted, trend-based, multi-horizon)
    from config.random_forest import (
        RANDOM_FOREST_MULTI_HORIZON_ENABLED,
        RANDOM_FOREST_TREND_BASED_LABELING_ENABLED,
        RANDOM_FOREST_USE_VOLATILITY_ADJUSTED_THRESHOLDS,
    )
    from modules.random_forest.utils.advanced_labeling import create_advanced_target

    target_series, multi_horizon_targets = create_advanced_target(
        df_with_features,
        use_volatility_adjusted=RANDOM_FOREST_USE_VOLATILITY_ADJUSTED_THRESHOLDS,
        use_trend_based=RANDOM_FOREST_TREND_BASED_LABELING_ENABLED,
        use_multi_horizon=RANDOM_FOREST_MULTI_HORIZON_ENABLED,
    )

    df_with_features["target"] = pd.to_numeric(target_series, errors="coerce").astype(int)

    # Add multi-horizon targets if enabled
    if multi_horizon_targets:
        for horizon_name, horizon_target in multi_horizon_targets.items():
            df_with_features[horizon_name] = pd.to_numeric(horizon_target, errors="coerce").astype(int)

    # Drop rows with NaN targets immediately to prevent contamination
    # The last RANDOM_FOREST_TARGET_HORIZON rows will have NaN targets (no future data available)
    # Also drop NaN in multi-horizon targets if enabled
    target_columns = ["target"]
    if multi_horizon_targets:
        target_columns.extend(multi_horizon_targets.keys())

    initial_rows = len(df_with_features)
    df_with_features = df_with_features.dropna(subset=target_columns).copy()
    dropped_rows = initial_rows - len(df_with_features)
    if dropped_rows > 0:
        log_progress(
            f"Dropped {dropped_rows} rows with NaN targets "
            f"(last {RANDOM_FOREST_TARGET_HORIZON} periods have no future data)"
        )

    # Filter features: Include both base MODEL_FEATURES and new enhanced features
    # Better approach: preserve order from MODEL_FEATURES
    available_features = []
    for feat in MODEL_FEATURES:
        if feat in df_with_features.columns:
            available_features.append(feat)

    # Then add enhanced features that aren't in MODEL_FEATURES
    enhanced_features = get_enhanced_feature_names(df_with_features.columns.tolist())
    for feat in enhanced_features:
        if feat not in available_features:
            available_features.append(feat)

    # No need to sort - preserve insertion order to ensure consistency between training/inference

    if not available_features:
        log_error("None of the required MODEL_FEATURES or enhanced features are present in the DataFrame.")
        return None

    # Final dropna for any remaining NaN values in features
    df_with_features.dropna(subset=available_features, inplace=True)

    if len(df_with_features) < MIN_TRAINING_SAMPLES:
        log_warn(
            f"Insufficient training samples after feature creation: {len(df_with_features)} < {MIN_TRAINING_SAMPLES}"
        )
        return None

    # Use only available features
    features = df_with_features[available_features]
    target = df_with_features["target"]
    log_progress(f"Training data prepared. Features shape: {features.shape}, target shape: {target.shape}")
    return cast(Tuple[pd.DataFrame, pd.Series], (features, target))
