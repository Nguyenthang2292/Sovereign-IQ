use pyo3::prelude::*;

// Adaptive Trend modules
pub mod batch_processing_cpu;
pub mod equity;
pub mod incremental_atc;
pub mod kama;
pub mod liquidity_metrics;
pub mod ma_calculations;
pub mod signal_persistence;
pub mod utils;

// XGBoost modules (prefixed with xgb_)
pub mod xgb_features;
pub mod xgb_labeling;
pub mod xgb_utils;

/// A Python module implemented in Rust.
#[pymodule]
fn sovereign_prime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Adaptive Trend functions
    m.add_function(wrap_pyfunction!(equity::calculate_equity_rust, m)?)?;
    m.add_function(wrap_pyfunction!(kama::calculate_kama_rust, m)?)?;
    m.add_function(wrap_pyfunction!(
        signal_persistence::process_signal_persistence_rust,
        m
    )?)?;

    // MA calculations
    m.add_function(wrap_pyfunction!(ma_calculations::calculate_ema_rust, m)?)?;
    m.add_function(wrap_pyfunction!(ma_calculations::calculate_wma_rust, m)?)?;
    m.add_function(wrap_pyfunction!(ma_calculations::calculate_dema_rust, m)?)?;
    m.add_function(wrap_pyfunction!(ma_calculations::calculate_lsma_rust, m)?)?;
    m.add_function(wrap_pyfunction!(ma_calculations::calculate_hma_rust, m)?)?;

    // Batch CPU processing
    m.add_function(wrap_pyfunction!(
        batch_processing_cpu::compute_atc_signals_batch_cpu,
        m
    )?)?;

    // Liquidity metrics
    m.add_function(wrap_pyfunction!(
        liquidity_metrics::compute_liquidity_metrics_batch,
        m
    )?)?;

    // Incremental ATC
    m.add_function(wrap_pyfunction!(
        incremental_atc::update_incremental_atc_rust,
        m
    )?)?;

    // XGBoost labeling functions
    m.add_function(wrap_pyfunction!(
        xgb_labeling::calculate_volatility_multiplier_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        xgb_labeling::apply_directional_labels_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(xgb_labeling::rolling_quantile_rust, m)?)?;
    m.add_function(wrap_pyfunction!(xgb_labeling::rolling_mean_rust, m)?)?;

    // XGBoost feature functions
    m.add_function(wrap_pyfunction!(
        xgb_features::add_price_derived_features_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(xgb_features::rolling_std_rust, m)?)?;
    m.add_function(wrap_pyfunction!(xgb_features::rolling_skew_rust, m)?)?;
    m.add_function(wrap_pyfunction!(xgb_features::pct_change_rust, m)?)?;
    m.add_function(wrap_pyfunction!(
        xgb_features::add_advanced_features_rust,
        m
    )?)?;

    Ok(())
}
