"""
Shared configurations used across multiple modules.

This package contains configuration that is shared between modules:
- evaluation: Evaluation metrics and thresholds
- model_features: Feature definitions for ML models
- forex_pairs: Currency pair definitions
"""

from .evaluation import *  # noqa: F403, F401
from .forex_pairs import *  # noqa: F403, F401
from .model_features import *  # noqa: F403, F401
