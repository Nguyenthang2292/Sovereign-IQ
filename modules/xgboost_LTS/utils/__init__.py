"""Utility functions for XGBoost module."""

# Re-export common utilities for backward compatibility
from modules.common.utils import color_text, format_price, timeframe_to_minutes
from modules.xgboost_LTS.utils.display import print_classification_report
from modules.xgboost_LTS.utils.utils import get_prediction_window
from modules.xgboost_LTS.utils.gpu_utils import detect_cuda_available, get_gpu_info

__all__ = [
    "print_classification_report",
    "get_prediction_window",
    "detect_cuda_available",
    "get_gpu_info",
    "color_text",
    "format_price",
    "timeframe_to_minutes",
]
