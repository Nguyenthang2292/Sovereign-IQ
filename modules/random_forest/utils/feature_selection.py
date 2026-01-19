"""Feature Selection for Random Forest models.

This module provides feature selection methods:
1. SelectKBest with mutual information
2. Random Forest feature importance with threshold
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from config.random_forest import (
    RANDOM_FOREST_FEATURE_IMPORTANCE_THRESHOLD,
    RANDOM_FOREST_FEATURE_SELECTION_K,
    RANDOM_FOREST_FEATURE_SELECTION_METHOD,
)
from modules.common.ui.logging import log_info, log_progress, log_warn


def select_features_mutual_info(
    features: pd.DataFrame, target: pd.Series, k: int = RANDOM_FOREST_FEATURE_SELECTION_K
) -> Tuple[pd.DataFrame, List[str], SelectKBest]:
    """
    Select top-K features using mutual information.

    Args:
        features: Feature DataFrame
        target: Target Series
        k: Number of top features to select

    Returns:
        Tuple of (selected_features, selected_feature_names, selector)
    """
    log_progress(f"Selecting top {k} features using mutual information...")

    # Ensure k doesn't exceed available features
    k = min(k, len(features.columns))

    if k >= len(features.columns):
        log_warn(f"k ({k}) >= number of features ({len(features.columns)}). Using all features.")
        return features, list(features.columns), None

    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    features_selected = selector.fit_transform(features, target)

    # Get selected feature names
    selected_mask = selector.get_support()
    selected_feature_names = [name for name, selected in zip(features.columns, selected_mask) if selected]

    # Convert back to DataFrame with original column names
    features_selected_df = pd.DataFrame(features_selected, columns=selected_feature_names, index=features.index)

    log_info(f"Selected {len(selected_feature_names)} features from {len(features.columns)} total features")
    log_info(f"Selected features: {', '.join(selected_feature_names[:10])}{'...' if len(selected_feature_names) > 10 else ''}")

    return features_selected_df, selected_feature_names, selector


def select_features_rf_importance(
    features: pd.DataFrame,
    target: pd.Series,
    model: Optional[RandomForestClassifier] = None,
    threshold: float = RANDOM_FOREST_FEATURE_IMPORTANCE_THRESHOLD,
) -> Tuple[pd.DataFrame, List[str], Optional[RandomForestClassifier]]:
    """
    Select features based on Random Forest feature importance.

    Args:
        features: Feature DataFrame
        target: Target Series
        model: Pre-trained RandomForest model (if None, trains a new one)
        threshold: Minimum feature importance threshold

    Returns:
        Tuple of (selected_features, selected_feature_names, trained_model)
    """
    log_progress(f"Selecting features using Random Forest importance (threshold={threshold})...")

    # Train model if not provided
    if model is None:
        log_progress("Training Random Forest model for feature importance calculation...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(features, target)

    # Get feature importances
    feature_importances = model.feature_importances_
    feature_names = features.columns.tolist()

    # Select features above threshold
    importance_mask = feature_importances > threshold
    selected_feature_names = [name for name, selected in zip(feature_names, importance_mask) if selected]

    if len(selected_feature_names) == 0:
        log_warn(f"No features above threshold {threshold}. Using top 10 features by importance.")
        # Fallback: select top 10 features
        top_indices = np.argsort(feature_importances)[-10:]
        selected_feature_names = [feature_names[i] for i in top_indices]
        importance_mask = np.zeros(len(feature_names), dtype=bool)
        importance_mask[top_indices] = True

    # Create selected features DataFrame
    features_selected = features.loc[:, selected_feature_names].copy()

    log_info(
        f"Selected {len(selected_feature_names)} features from {len(features.columns)} total features "
        f"(threshold={threshold})"
    )
    log_info(f"Selected features: {', '.join(selected_feature_names[:10])}{'...' if len(selected_feature_names) > 10 else ''}")

    # Log top feature importances
    top_features = sorted(
        zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True
    )[:5]
    log_info(f"Top 5 feature importances: {', '.join([f'{name}={imp:.4f}' for name, imp in top_features])}")

    return features_selected, selected_feature_names, model


def select_features(
    features: pd.DataFrame,
    target: pd.Series,
    method: str = RANDOM_FOREST_FEATURE_SELECTION_METHOD,
    model: Optional[RandomForestClassifier] = None,
) -> Tuple[pd.DataFrame, List[str], Optional[object]]:
    """
    Select features using specified method.

    Args:
        features: Feature DataFrame
        target: Target Series
        method: Selection method ("mutual_info" or "rf_importance")
        model: Pre-trained RandomForest model (only used for rf_importance method)

    Returns:
        Tuple of (selected_features, selected_feature_names, selector_or_model)
    """
    if method == "mutual_info":
        return select_features_mutual_info(features, target)
    elif method == "rf_importance":
        return select_features_rf_importance(features, target, model=model)
    else:
        log_warn(f"Unknown feature selection method: {method}. Using all features.")
        return features, list(features.columns), None


__all__ = [
    "select_features",
    "select_features_mutual_info",
    "select_features_rf_importance",
]
