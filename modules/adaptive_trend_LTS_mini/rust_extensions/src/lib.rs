use pyo3::prelude::*;

pub mod batch_processing_cpu;
pub mod equity;
pub mod incremental_atc;
pub mod kama;
pub mod liquidity_metrics;
pub mod ma_calculations;
pub mod signal_persistence;
pub mod utils;

#[pymodule]
fn atc_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
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

    // Batch CPU processing (Rayon - CPU-only)
    m.add_function(wrap_pyfunction!(
        batch_processing_cpu::compute_atc_signals_batch_cpu,
        m
    )?)?;

    // Liquidity metrics for Stage 0 sampling (Rayon parallel)
    m.add_function(wrap_pyfunction!(
        liquidity_metrics::compute_liquidity_metrics_batch,
        m
    )?)?;

    // Incremental ATC updates (Rust backend for single-bar updates)
    m.add_function(wrap_pyfunction!(
        incremental_atc::update_incremental_atc_rust,
        m
    )?)?;

    Ok(())
}
