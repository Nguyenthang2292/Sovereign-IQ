"""Utility functions for XGBoost module."""

# Re-export common utilities for backward compatibility
from modules.common.utils import color_text, format_price, timeframe_to_minutes
from modules.xgboost.utils.display import print_classification_report
from modules.xgboost.utils.utils import get_prediction_window

__all__ = [
    "print_classification_report",
    "get_prediction_window",
    "color_text",
    "format_price",
    "timeframe_to_minutes",
]
