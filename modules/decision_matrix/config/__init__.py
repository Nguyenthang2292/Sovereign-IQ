"""Configuration for Decision Matrix module."""

from modules.decision_matrix.config.config import (
    MAX_CAP_ITERATIONS,
    MAX_WEIGHT_CAP_N2,
    MAX_WEIGHT_CAP_N3_PLUS,
    FeatureType,
    RandomForestConfig,
    TargetType,
)

__all__ = [
    "RandomForestConfig",
    "FeatureType",
    "TargetType",
    "MAX_WEIGHT_CAP_N2",
    "MAX_WEIGHT_CAP_N3_PLUS",
    "MAX_CAP_ITERATIONS",
]
