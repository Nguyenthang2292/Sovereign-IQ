"""
Decision Matrix Classification Algorithm Module.

Inspired by Random Forest voting system, this module provides a voting-based
classification system for combining signals from multiple indicators.
"""

from modules.decision_matrix.core.classifier import DecisionMatrixClassifier
from modules.decision_matrix.core.random_forest_core import RandomForestCore
from modules.decision_matrix.utils.pattern_matcher import PatternMatcher
from modules.decision_matrix.utils.shuffle import ShuffleMechanism
from modules.decision_matrix.utils.threshold import ThresholdCalculator
from modules.decision_matrix.utils.training_data import TrainingDataStorage
from modules.decision_matrix.config.config import (
    FeatureType,
    RandomForestConfig,
    TargetType,
    MAX_WEIGHT_CAP_N2,
    MAX_WEIGHT_CAP_N3_PLUS,
    MAX_CAP_ITERATIONS,
)

__all__ = [
    "DecisionMatrixClassifier",
    "TrainingDataStorage",
    "ShuffleMechanism",
    "ThresholdCalculator",
    "PatternMatcher",
    "RandomForestCore",
    "RandomForestConfig",
    "FeatureType",
    "TargetType",
    "MAX_WEIGHT_CAP_N2",
    "MAX_WEIGHT_CAP_N3_PLUS",
    "MAX_CAP_ITERATIONS",
]
