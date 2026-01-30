"""XGBoost prediction component."""

from modules.xgboost.core.labeling import apply_directional_labels
from modules.xgboost.core.model import (
    ClassDiversityError,
    predict_next_move,
    train_and_predict,
)
from modules.xgboost.core.optimization import HyperparameterTuner, StudyManager
from modules.xgboost.utils.display import print_classification_report
from modules.xgboost.utils.utils import get_prediction_window

__all__ = [
    "HyperparameterTuner",
    "StudyManager",
    "ClassDiversityError",
    "predict_next_move",
    "train_and_predict",
    "apply_directional_labels",
    "print_classification_report",
    "get_prediction_window",
]
