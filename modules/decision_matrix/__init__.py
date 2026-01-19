"""
Decision Matrix Classification Algorithm Module.

Inspired by Random Forest voting system, this module provides a voting-based
classification system for combining signals from multiple indicators.

⚠️ IMPORTANT: RandomForestCore in this module is NOT sklearn's RandomForestClassifier.
- modules.decision_matrix.RandomForestCore: Pine Script pattern matching algorithm
- modules.random_forest: sklearn RandomForestClassifier wrapper (ML-based signals)

These are two completely independent implementations with similar names.
"""

from modules.decision_matrix.config.config import (
    MAX_CAP_ITERATIONS,
    MAX_WEIGHT_CAP_N2,
    MAX_WEIGHT_CAP_N3_PLUS,
    FeatureType,
    RandomForestConfig,
    TargetType,
)
from modules.decision_matrix.core.classifier import DecisionMatrixClassifier
from modules.decision_matrix.core.random_forest_core import RandomForestCore
from modules.decision_matrix.utils.pattern_matcher import PatternMatcher
from modules.decision_matrix.utils.shuffle import ShuffleMechanism
from modules.decision_matrix.utils.threshold import ThresholdCalculator
from modules.decision_matrix.utils.training_data import TrainingDataStorage

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
