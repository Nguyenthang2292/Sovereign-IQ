use pyo3::prelude::*;

pub mod equity;
pub mod kama;
pub mod signal_persistence;
pub mod ma_calculations;

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
    
    Ok(())
}
