// Add this function to batch_processing.rs after compute_atc_signals_batch

#[pyfunction]
pub fn compute_atc_signals_batch_debug(
    py: Python,
    symbols_data: &PyDict,
    ema_len: usize,
    hull_len: usize,
    wma_len: usize,
    dema_len: usize,
    lsma_len: usize,
    kama_len: usize,
    robustness: &str,
    La: f64,
    De: f64,
    long_threshold: f64,
    short_threshold: f64,
) -> PyResult<PyObject> {
    // Same as compute_atc_signals_batch but export intermediate values

    let cache = get_batch_cuda_cache()?;
    let batch = prepare_batch_data(symbols_data)?;
    let total_bars = batch.total_bars;

    // ... (copy all the computation code from compute_atc_signals_batch)
    // But before returning, export f1 and f2

    // Export Layer 1 signals (f1)
    let mut l1_host = vec![0.0; total_bars * 6];
    cache
        .ctx
        .default_stream()
        .memcpy_dtoh(&f1, &mut l1_host)
        .unwrap();

    // Export Layer 2 equities (f2)
    let mut l2_host = vec![0.0; total_bars * 6];
    cache
        .ctx
        .default_stream()
        .memcpy_dtoh(&f2, &mut l2_host)
        .unwrap();

    // Create result dict
    let result = PyDict::new(py);

    // Add Layer 1 signals by MA type
    let ma_types = ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"];
    let l1_dict = PyDict::new(py);
    let l2_dict = PyDict::new(py);

    for (i, ma_type) in ma_types.iter().enumerate() {
        let start_idx = i * total_bars;
        let end_idx = (i + 1) * total_bars;

        l1_dict.set_item(
            *ma_type,
            PyArray1::from_vec(py, l1_host[start_idx..end_idx].to_vec()),
        )?;

        l2_dict.set_item(
            *ma_type,
            PyArray1::from_vec(py, l2_host[start_idx..end_idx].to_vec()),
        )?;
    }

    result.set_item("layer1_signals", l1_dict)?;
    result.set_item("layer2_equities", l2_dict)?;

    // Also add final result
    let mut host = vec![0.0; total_bars];
    cache
        .ctx
        .default_stream()
        .memcpy_dtoh(&avg, &mut host)
        .unwrap();

    let final_dict = PyDict::new(py);
    for (i, (key, _)) in symbols_data.iter().enumerate() {
        let sym: String = key.extract()?;
        let start = batch.offsets[i] as usize;
        let len = batch.lengths[i] as usize;
        final_dict.set_item(
            sym,
            PyArray1::from_vec(py, host[start..start + len].to_vec()),
        )?;
    }

    result.set_item("final_signals", final_dict)?;

    Ok(result.into())
}
