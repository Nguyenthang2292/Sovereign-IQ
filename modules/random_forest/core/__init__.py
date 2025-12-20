"""Core Random Forest functionality.

This module provides core functionality for Random Forest model training,
signal generation, and evaluation.
"""

from modules.random_forest.core.model import (
    load_random_forest_model,
    train_random_forest_model,
    train_and_save_global_rf_model,
)
from modules.random_forest.core.signals import (
    get_latest_random_forest_signal,
)
from modules.random_forest.core.evaluation import (
    evaluate_model_with_confidence,
    apply_confidence_threshold,
    calculate_and_display_metrics,
)
from modules.random_forest.core.decision_matrix_integration import (
    calculate_random_forest_vote,
    get_random_forest_signal_for_decision_matrix,
)

__all__ = [
    "load_random_forest_model",
    "train_random_forest_model",
    "train_and_save_global_rf_model",
    "get_latest_random_forest_signal",
    "evaluate_model_with_confidence",
    "apply_confidence_threshold",
    "calculate_and_display_metrics",
    "calculate_random_forest_vote",
    "get_random_forest_signal_for_decision_matrix",
]

