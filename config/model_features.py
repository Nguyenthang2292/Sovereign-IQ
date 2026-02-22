"""Backward-compatible model feature exports.

This module preserves legacy imports like:
    from config.model_features import MODEL_FEATURES

Source of truth remains in config.shared.model_features.
"""

from .shared.model_features import *  # noqa: F403, F401
