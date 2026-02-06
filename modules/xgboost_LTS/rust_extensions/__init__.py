"""
XGBoost Rust Extensions

High-performance Rust implementations for critical XGBoost operations.
"""

from .xgboost_rust import (  # type: ignore[import-not-found]
    add_advanced_features_rust,
    add_price_derived_features_rust,
    apply_directional_labels_rust,
    calculate_volatility_multiplier_rust,
    pct_change_rust,
    rolling_mean_rust,
    rolling_quantile_rust,
    rolling_std_rust,
)

__all__ = [
    "calculate_volatility_multiplier_rust",
    "apply_directional_labels_rust",
    "rolling_quantile_rust",
    "rolling_mean_rust",
    "add_price_derived_features_rust",
    "rolling_std_rust",
    "pct_change_rust",
    "add_advanced_features_rust",
]
