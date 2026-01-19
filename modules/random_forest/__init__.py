"""Random Forest Machine Learning Module.

This module provides sklearn-based Random Forest classification for trading signals.
Uses scikit-learn's RandomForestClassifier for machine learning-based signal prediction.

⚠️ IMPORTANT: NOT to be confused with modules.decision_matrix.RandomForestCore
- This module (modules.random_forest): sklearn RandomForestClassifier wrapper (ML-based signals)
- modules.decision_matrix.RandomForestCore: Pine Script pattern matching algorithm (pattern-based classification)

For ML-based Random Forest model training and prediction, use this module.
For Pine Script pattern matching algorithm, see modules.decision_matrix.core.random_forest_core instead.
"""

from modules.random_forest.core import (
    apply_confidence_threshold,
    calculate_and_display_metrics,
    calculate_random_forest_vote,
    evaluate_model_with_confidence,
    get_latest_random_forest_signal,
    get_random_forest_signal_for_decision_matrix,
    load_random_forest_model,
    train_and_save_global_rf_model,
    train_random_forest_model,
)
from modules.random_forest.optimization import HyperparameterTuner, StudyManager

__all__ = [
    "load_random_forest_model",
    "train_random_forest_model",
    "get_latest_random_forest_signal",
    "train_and_save_global_rf_model",
    "evaluate_model_with_confidence",
    "apply_confidence_threshold",
    "calculate_and_display_metrics",
    "calculate_random_forest_vote",
    "get_random_forest_signal_for_decision_matrix",
    "HyperparameterTuner",
    "StudyManager",
]
