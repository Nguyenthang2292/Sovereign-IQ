
from typing import Any, Dict, Optional, Tuple
import gc

import numpy as np
import pandas as pd
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

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from config import (
    LARGE_DATASET_THRESHOLD_FOR_SMOTE,
    MIN_MEMORY_GB,
    MODEL_RANDOM_STATE,
)
from modules.common.ui.logging import (
    log_error,
    log_progress,
    log_warn,
)

SMOTE_RANDOM_STATE = MODEL_RANDOM_STATE


def apply_smote(features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Balance training data using SMOTE.

    Args:
        features (pd.DataFrame): Input features.
        target (pd.Series): Target labels.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Balanced data or original data if error.
    """
    log_progress("Applying SMOTE for class balancing...")
    try:
        # Check available memory if psutil is available
        if PSUTIL_AVAILABLE:
            available_gb = psutil.virtual_memory().available / (1024**3)
            if available_gb < MIN_MEMORY_GB:
                log_warn(f"Low memory ({available_gb:.2f}GB), skipping SMOTE to avoid crashing.")
                return features, target

        smote_kwargs = {"random_state": SMOTE_RANDOM_STATE, "sampling_strategy": "auto"}
        if len(features) > LARGE_DATASET_THRESHOLD_FOR_SMOTE:
            log_progress("Dataset is large, using reduced k_neighbors for SMOTE.")
            smote_kwargs["k_neighbors"] = 3

        smote = SMOTE(**smote_kwargs)
        result = smote.fit_resample(features, target)

        if isinstance(result, tuple) and len(result) == 2:
            features_resampled, target_resampled = result
        else:
            log_error("SMOTE did not return a tuple of (features, target). Skipping resampling.")
            return features, target

        log_progress(f"SMOTE applied. New data shape: {features_resampled.shape}")
        gc.collect()

        if not isinstance(features_resampled, pd.DataFrame):
            features_resampled = pd.DataFrame(features_resampled, columns=features.columns)
        if not isinstance(target_resampled, pd.Series):
            target_resampled = pd.Series(target_resampled, name="target")

        # Ensure target remains numeric
        target_resampled = pd.to_numeric(target_resampled, errors="coerce").astype(int)

        return features_resampled, target_resampled
    except (ValueError, RuntimeError, MemoryError) as e:
        log_error(f"SMOTE failed: {e}. Training will continue with original data.")
        return features, target


def create_model_and_weights(
    target_resampled: pd.Series, custom_params: Optional[Dict[str, Any]] = None
) -> RandomForestClassifier:
    """Create Random Forest model and compute class weights.

    Args:
        target_resampled (pd.Series): Resampled target variable.
        custom_params (Optional[Dict[str, Any]]): Custom hyperparameters to override defaults.

    Returns:
        RandomForestClassifier: Model with computed class weights.
    """
    class_weights = compute_class_weight("balanced", classes=np.unique(target_resampled), y=target_resampled)
    weight_dict = {int(cls): weight for cls, weight in zip(np.unique(target_resampled), class_weights)}

    # Default parameters
    default_params = {
        "n_estimators": 100,
        "class_weight": weight_dict,
        "random_state": MODEL_RANDOM_STATE,
        "n_jobs": -1,
        "min_samples_leaf": 5,
    }

    # Merge custom params if provided (custom params override defaults)
    if custom_params:
        # Ensure class_weight is always from computed weights (don't override)
        model_params = {**default_params, **custom_params}
        model_params["class_weight"] = weight_dict  # Always use computed weights
    else:
        model_params = default_params

    model = RandomForestClassifier(**model_params)
    return model
