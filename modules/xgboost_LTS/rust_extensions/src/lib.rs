//! XGBoost Rust Extensions
//!
//! High-performance implementations of labeling and feature engineering operations.

use pyo3::prelude::*;

pub mod features;
pub mod labeling;
mod utils;

/// Python module initialization

#[pymodule]
fn xgboost_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register labeling functions
    m.add_function(wrap_pyfunction!(
        labeling::calculate_volatility_multiplier_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        labeling::apply_directional_labels_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(labeling::rolling_quantile_rust, m)?)?;
    m.add_function(wrap_pyfunction!(labeling::rolling_mean_rust, m)?)?;

    // Register feature functions
    m.add_function(wrap_pyfunction!(
        features::add_price_derived_features_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(features::rolling_std_rust, m)?)?;
    m.add_function(wrap_pyfunction!(features::rolling_skew_rust, m)?)?;
    m.add_function(wrap_pyfunction!(features::pct_change_rust, m)?)?;
    m.add_function(wrap_pyfunction!(features::add_advanced_features_rust, m)?)?;
    m.add_function(wrap_pyfunction!(features::calculate_all_features_rust, m)?)?;

    Ok(())
}
