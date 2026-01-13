"""
Decision Matrix Classification Algorithm Module.

Inspired by Random Forest voting system, this module provides a voting-based
classification system for combining signals from multiple indicators.
"""

from modules.decision_matrix.classifier import DecisionMatrixClassifier
from modules.random_forest.core.decision_matrix_integration import (
    calculate_random_forest_vote,
    get_random_forest_signal_for_decision_matrix,
)

__all__ = [
    "DecisionMatrixClassifier",
    "calculate_random_forest_vote",
    "get_random_forest_signal_for_decision_matrix",
]
