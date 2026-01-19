"""Utility functions for Random Forest module.

This module provides utility functions for data preparation, training,
feature selection, and walk-forward optimization.
"""

from modules.random_forest.utils.calibration import calibrate_model, evaluate_calibration
from modules.random_forest.utils.data_preparation import prepare_training_data
from modules.random_forest.utils.feature_selection import (
    select_features,
    select_features_mutual_info,
    select_features_rf_importance,
)
from modules.random_forest.utils.training import apply_sampling, create_model_and_weights
from modules.random_forest.utils.walk_forward import (
    ModelDriftDetector,
    ModelVersionManager,
    WalkForwardValidator,
    should_retrain_model,
)

__all__ = [
    "prepare_training_data",
    "apply_sampling",
    "create_model_and_weights",
    # Feature selection
    "select_features",
    "select_features_mutual_info",
    "select_features_rf_importance",
    # Walk-forward
    "WalkForwardValidator",
    "ModelDriftDetector",
    "ModelVersionManager",
    "should_retrain_model",
    # Calibration
    "calibrate_model",
    "evaluate_calibration",
]
