"""XGBoost Rust Extensions.

High-performance Rust implementations for critical XGBoost operations.
"""

from importlib import import_module

_xgboost_rust = import_module("modules.xgboost_LTS.rust_extensions.xgboost_rust")

add_advanced_features_rust = _xgboost_rust.add_advanced_features_rust
add_price_derived_features_rust = _xgboost_rust.add_price_derived_features_rust
apply_directional_labels_rust = _xgboost_rust.apply_directional_labels_rust
calculate_volatility_multiplier_rust = _xgboost_rust.calculate_volatility_multiplier_rust
pct_change_rust = _xgboost_rust.pct_change_rust
rolling_mean_rust = _xgboost_rust.rolling_mean_rust
rolling_quantile_rust = _xgboost_rust.rolling_quantile_rust
rolling_std_rust = _xgboost_rust.rolling_std_rust

calculate_all_features_rust = getattr(_xgboost_rust, "calculate_all_features_rust", None)

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

if calculate_all_features_rust is not None:
    __all__.append("calculate_all_features_rust")
