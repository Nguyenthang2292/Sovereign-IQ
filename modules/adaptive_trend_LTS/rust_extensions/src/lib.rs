use pyo3::prelude::*;

pub mod equity;
pub mod kama;
pub mod signal_persistence;
pub mod ma_calculations;
pub mod equity_cuda;
pub mod ma_cuda;
pub mod signal_cuda;
pub mod batch_processing;
pub mod utils;
pub mod batch_processing_cpu;

#[pymodule]
fn atc_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(equity::calculate_equity_rust, m)?)?;
    m.add_function(wrap_pyfunction!(kama::calculate_kama_rust, m)?)?;
    m.add_function(wrap_pyfunction!(signal_persistence::process_signal_persistence_rust, m)?)?;

    // MA calculations
    m.add_function(wrap_pyfunction!(ma_calculations::calculate_ema_rust, m)?)?;
    m.add_function(wrap_pyfunction!(ma_calculations::calculate_wma_rust, m)?)?;
    m.add_function(wrap_pyfunction!(ma_calculations::calculate_dema_rust, m)?)?;
    m.add_function(wrap_pyfunction!(ma_calculations::calculate_lsma_rust, m)?)?;
    m.add_function(wrap_pyfunction!(ma_calculations::calculate_hma_rust, m)?)?;
    
    // CUDA functions
    m.add_function(wrap_pyfunction!(equity_cuda::calculate_equity_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(ma_cuda::calculate_ema_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(ma_cuda::calculate_kama_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(ma_cuda::calculate_wma_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(ma_cuda::calculate_hma_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(signal_cuda::calculate_average_signal_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(signal_cuda::classify_trend_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(signal_cuda::calculate_and_classify_cuda, m)?)?;
    
    // Batch CUDA processing (True Batch - processes all symbols in one kernel)
    m.add_function(wrap_pyfunction!(batch_processing::compute_atc_signals_batch, m)?)?;
    
    // Batch CPU processing (Rayon)
    m.add_function(wrap_pyfunction!(batch_processing_cpu::compute_atc_signals_batch_cpu, m)?)?;
    
    Ok(())
}
