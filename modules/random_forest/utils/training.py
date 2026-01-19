import gc
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

"""Training utilities for Random Forest models.

This module provides utilities for training Random Forest models, including
SMOTE for class balancing and model creation with class weights.
"""


try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.ensemble import BalancedRandomForestClassifier

    BALANCED_RF_AVAILABLE = True
except ImportError:
    BALANCED_RF_AVAILABLE = False

from config import (
    LARGE_DATASET_THRESHOLD_FOR_SMOTE,
    MIN_MEMORY_GB,
    MODEL_RANDOM_STATE,
    RANDOM_FOREST_SAMPLING_STRATEGY,
)
from modules.common.ui.logging import (
    log_error,
    log_progress,
    log_warn,
)

SMOTE_RANDOM_STATE = MODEL_RANDOM_STATE


def apply_sampling(features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series, bool]:
    """Balance training data using chosen sampling strategy.

    Args:
        features (pd.DataFrame): Input features.
        target (pd.Series): Target labels.

    Returns:
        Tuple[pd.DataFrame, pd.Series, bool]: Balanced data and flag indicating if sampling was applied.
    """
    strategy = RANDOM_FOREST_SAMPLING_STRATEGY

    if strategy == "NONE" or strategy == "BALANCED_RF":
        log_progress(f"Skipping resampling step (Strategy: {strategy}).")
        return features, target, False

    log_progress(f"Applying {strategy} for class balancing...")
    try:
        # Check available memory if psutil is available
        if PSUTIL_AVAILABLE:
            available_gb = psutil.virtual_memory().available / (1024**3)
            if available_gb < MIN_MEMORY_GB:
                log_warn(f"Low memory ({available_gb:.2f}GB), skipping {strategy}.")
                return features, target, False

        kwargs = {"random_state": SMOTE_RANDOM_STATE, "sampling_strategy": "auto"}

        if strategy in ["SMOTE", "BorderlineSMOTE"]:
            if len(features) > LARGE_DATASET_THRESHOLD_FOR_SMOTE:
                log_progress(f"Dataset is large, using reduced k_neighbors for {strategy}.")
                kwargs["k_neighbors"] = 3

        sampler = None
        if strategy == "SMOTE":
            sampler = SMOTE(**kwargs)
        elif strategy == "ADASYN":
            sampler = ADASYN(**kwargs)
        elif strategy == "BorderlineSMOTE":
            sampler = BorderlineSMOTE(**kwargs)
        else:
            log_warn(f"Unknown sampling strategy '{strategy}'. Falling back to SMOTE.")
            sampler = SMOTE(**kwargs)

        try:
            features_resampled, target_resampled = sampler.fit_resample(features, target)
        except (ValueError, RuntimeError, MemoryError) as e:
            log_warn(f"{strategy} application failed: {e}. Proceed with original data.")
            return features, target, False

        log_progress(f"{strategy} applied successfully. New data shape: {features_resampled.shape}")
        gc.collect()

        if not isinstance(features_resampled, pd.DataFrame):
            features_resampled = pd.DataFrame(features_resampled, columns=features.columns)
        if not isinstance(target_resampled, pd.Series):
            target_resampled = pd.Series(target_resampled, name="target")

        # Ensure target remains numeric
        target_resampled = pd.to_numeric(target_resampled, errors="coerce").astype(int)

        return features_resampled, target_resampled, True
    except Exception as e:
        log_error(f"Unexpected error during {strategy} process: {e}. Training will continue with original data.")
        return features, target, False


def create_model_and_weights(
    target_resampled: pd.Series, custom_params: Optional[Dict[str, Any]] = None, sampling_applied: bool = False
) -> RandomForestClassifier:
    """Create Random Forest model and compute class weights.

    Args:
        target_resampled (pd.Series): Resampled target variable.
        custom_params (Optional[Dict[str, Any]]): Custom hyperparameters to override defaults.
        sampling_applied (bool): Whether a resampling strategy was applied to balance the data.

    Returns:
        RandomForestClassifier: Model (Standard or Balanced).
    """
    strategy = RANDOM_FOREST_SAMPLING_STRATEGY

    use_balanced_rf = strategy == "BALANCED_RF"
    if use_balanced_rf and not BALANCED_RF_AVAILABLE:
        log_warn("BalancedRandomForestClassifier not available. Falling back to standard RandomForestClassifier.")
        use_balanced_rf = False

    class_weight_value = None
    if use_balanced_rf:
        log_progress("Using BalancedRandomForestClassifier (handles balancing internally).")
    elif sampling_applied:
        # Use strategy from config for logging consistency
        log_progress(f"{RANDOM_FOREST_SAMPLING_STRATEGY} was applied - disabling class weights.")
    else:
        # Compute class weights for imbalanced data
        class_weights = compute_class_weight("balanced", classes=np.unique(target_resampled), y=target_resampled)
        weight_dict = {int(cls): weight for cls, weight in zip(np.unique(target_resampled), class_weights)}
        class_weight_value = weight_dict
        log_progress("Computing class weights for imbalanced data (No resampling applied).")

    # Default parameters
    default_params = {
        "n_estimators": 100,
        "class_weight": class_weight_value,
        "random_state": MODEL_RANDOM_STATE,
        "n_jobs": -1,
        "min_samples_leaf": 5,
    }

    # Merge custom params if provided (custom params override defaults)
    if custom_params:
        model_params = {**default_params, **custom_params}
        # Ensure class_weight is set correctly based on strategy/sampling status
        model_params["class_weight"] = class_weight_value
    else:
        model_params = default_params

    if use_balanced_rf:
        # Remove class_weight if using BalancedRandomForest (it handles it via sampling)
        model_params.pop("class_weight", None)
        return BalancedRandomForestClassifier(**model_params)

    return RandomForestClassifier(**model_params)
