"""Data preparation utilities for Random Forest training.

This module provides utilities for preparing training data, including feature
engineering and target variable creation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, cast

from config import (
    BUY_THRESHOLD,
    MIN_TRAINING_SAMPLES,
    SELL_THRESHOLD,
)
from config.random_forest import RANDOM_FOREST_FEATURES
from modules.common.core.indicator_engine import IndicatorEngine, IndicatorConfig, IndicatorProfile
from modules.common.ui.logging import (
    log_info,
    log_error,
    log_warn,
    log_progress,
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
    
    log_progress("Creating target variable 'target'...")
    future_return = (
        df_with_features["close"].shift(-5) / df_with_features["close"] - 1
    )
    df_with_features['target'] = np.select(
        [future_return > BUY_THRESHOLD, future_return < SELL_THRESHOLD],
        [1, -1],
        default=0
    )
    
    # Ensure target is numeric and convert to int
    df_with_features['target'] = pd.to_numeric(df_with_features['target'], errors='coerce').astype(int)
    
    # Filter RANDOM_FOREST_FEATURES to only include features that actually exist in the DataFrame
    # RANDOM_FOREST_FEATURES only includes features computed by IndicatorProfile.CORE
    available_features = [f for f in RANDOM_FOREST_FEATURES if f in df_with_features.columns]
    if not available_features:
        log_error("None of the required RANDOM_FOREST_FEATURES are present in the DataFrame.")
        return None
    
    df_with_features.dropna(subset=['target'] + available_features, inplace=True)
    
    if len(df_with_features) < MIN_TRAINING_SAMPLES:
        log_warn(
            "Insufficient training samples after feature creation: "
            f"{len(df_with_features)} < {MIN_TRAINING_SAMPLES}"
        )
        return None
    
    # Use only available features
    features = df_with_features[available_features]
    target = df_with_features['target']
    log_progress(
        f"Training data prepared. Features shape: {features.shape}, "
        f"target shape: {target.shape}"
    )
    return cast(Tuple[pd.DataFrame, pd.Series], (features, target))

