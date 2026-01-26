use crate::equity::calculate_equity_internal;
use crate::kama::calculate_kama_internal;
use crate::ma_calculations::*;
use crate::signal_persistence::process_signal_persistence_internal;
use crate::utils::{calculate_roc_with_growth, get_diflen};
use ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

/// Helper to shift array by 1 (equivalent to pandas shift(1))
fn shift_array(arr: &Array1<f64>) -> Array1<f64> {
    let mut shifted = Array1::from_elem(arr.len(), f64::NAN);
    if arr.len() > 1 {
        // Copy elements 0..n-1 to 1..n
        for i in 1..arr.len() {
            shifted[i] = arr[i - 1];
        }
    }
    shifted
}

/// Compute ATC signals for a single symbol (CPU implementation)
fn process_single_symbol(
    prices: Vec<f64>,
    ema_len: usize,
    hull_len: usize,
    wma_len: usize,
    dema_len: usize,
    lsma_len: usize,
    kama_len: usize,
    ema_w: f64,
    hma_w: f64,
    wma_w: f64,
    dema_w: f64,
    lsma_w: f64,
    kama_w: f64,
    robustness: &str,
    la: f64,
    de: f64,
    cutout: usize,
    long_threshold: f64,
    short_threshold: f64,
) -> Vec<f64> {
    let prices_arr = Array1::from_vec(prices);
    let n = prices_arr.len();

    // 1. Calculate ROC with growth
    let r = calculate_roc_with_growth(prices_arr.view(), la);

    // MA Configurations
    let ma_configs = [
        ("EMA", ema_len, ema_w),
        ("HMA", hull_len, hma_w),
        ("WMA", wma_len, wma_w),
        ("DEMA", dema_len, dema_w),
        ("LSMA", lsma_len, lsma_w),
        ("KAMA", kama_len, kama_w),
    ];

    let mut layer1_signals = Vec::with_capacity(6);
    let mut layer2_equities = Vec::with_capacity(6);

    // Process each MA type
    for (ma_type, base_len, _) in ma_configs.iter() {
        let lengths = get_diflen(*base_len, robustness);

        let mut component_signals = Vec::with_capacity(9);
        let mut component_equities = Vec::with_capacity(9);

        for &len in &lengths {
            // Calculate MA
            let ma = match *ma_type {
                "EMA" => calculate_ema_internal(prices_arr.view(), len),
                "HMA" => calculate_hma_internal(prices_arr.view(), len),
                "WMA" => calculate_wma_internal(prices_arr.view(), len),
                "DEMA" => calculate_dema_internal(prices_arr.view(), len),
                "LSMA" => calculate_lsma_internal(prices_arr.view(), len),
                "KAMA" => calculate_kama_internal(prices_arr.view(), len),
                _ => Array1::from_elem(n, f64::NAN), // Should not happen
            };

            // Generate Signal (Cross Over/Under)
            // Signal logic: price > ma => 1, price < ma => -1, else 0 (or persist?)
            // Actually usually it's persistent based on cross.
            // Let's reproduce `generate_signal_from_ma` logic:
            // up = (close > ma) & (close[1] <= ma[1])
            // down = (close < ma) & (close[1] >= ma[1])
            // But strict cross usually check previous.

            // Using vector operations for cross detection
            // We need persistent signal based on crossovers
            let mut up = Array1::<bool>::from_elem(n, false);
            let mut down = Array1::<bool>::from_elem(n, false);

            for i in 1..n {
                let p_curr = prices_arr[i];
                let m_curr = ma[i];
                let p_prev = prices_arr[i - 1];
                let m_prev = ma[i - 1];

                if !p_curr.is_nan() && !m_curr.is_nan() && !p_prev.is_nan() && !m_prev.is_nan() {
                    // Crossover: Price crosses MA upwards
                    if p_curr > m_curr && p_prev <= m_prev {
                        up[i] = true;
                    }
                    // Crossunder: Price crosses MA downwards
                    if p_curr < m_curr && p_prev >= m_prev {
                        down[i] = true;
                    }
                }
            }

            // Persistence
            // Output is i8: 1, -1, 0. We need f64 for calculations later
            let sig_i8 = process_signal_persistence_internal(up.view(), down.view());
            let sig_f64 = sig_i8.mapv(|x| x as f64);

            // Equity
            let sig_prev = shift_array(&sig_f64);
            let eq = calculate_equity_internal(r.view(), sig_prev.view(), 1.0, 1.0 - de, cutout);

            component_signals.push(sig_f64);
            component_equities.push(eq);
        }

        // Calculate Weighted Signal for this MA type
        // Signal = Sum(s_i * w_i) / Sum(w_i)
        // Check `weighted_signal.py`:
        // weights should be normalized? No, it's just weighted avg.
        // Actually usually strictly: sum(s * e) / sum(e)
        // But need to handle negative equity? Usually equity > 0.

        let mut weighted_sig_sum = Array1::<f64>::zeros(n);
        let mut weight_sum = Array1::<f64>::zeros(n);

        for i in 0..9 {
            let s = &component_signals[i];
            let e = &component_equities[i];

            for j in 0..n {
                if !s[j].is_nan() && !e[j].is_nan() {
                    weighted_sig_sum[j] += s[j] * e[j];
                    weight_sum[j] += e[j].abs(); // Use abs just in case, though eq should be > 0
                }
            }
        }

        let mut l1_sig = Array1::<f64>::from_elem(n, f64::NAN);
        for j in 0..n {
            if weight_sum[j] > 1e-9 {
                l1_sig[j] = weighted_sig_sum[j] / weight_sum[j];
            } else {
                l1_sig[j] = 0.0;
            }
        }

        // Calculate Layer 2 Equity for this MA type
        // Use L1 signal (shifted)
        let l1_prev = shift_array(&l1_sig);
        let l2_eq = calculate_equity_internal(r.view(), l1_prev.view(), 1.0, 1.0 - de, cutout);

        layer1_signals.push(l1_sig);
        layer2_equities.push(l2_eq);
    }

    // Final Average Signal
    // Average_Signal = Sum(L1_sig_i * L2_eq_i * weight_i) / Sum(L2_eq_i * weight_i)
    let weights = [ema_w, hma_w, wma_w, dema_w, lsma_w, kama_w];

    let mut final_sig_num = Array1::<f64>::zeros(n);
    let mut final_sig_den = Array1::<f64>::zeros(n);

    for i in 0..6 {
        let s = &layer1_signals[i];
        let e = &layer2_equities[i];
        let w = weights[i];

        for j in 0..n {
            if !s[j].is_nan() && !e[j].is_nan() {
                final_sig_num[j] += s[j] * e[j] * w;
                final_sig_den[j] += e[j] * w;
            }
        }
    }

    let mut average_signal = Array1::<f64>::from_elem(n, f64::NAN);
    for j in cutout..n {
        if final_sig_den[j] > 1e-9 {
            let raw = final_sig_num[j] / final_sig_den[j];
            // Classify
            if raw > long_threshold {
                average_signal[j] = 1.0;
            } else if raw < short_threshold {
                average_signal[j] = -1.0;
            } else {
                average_signal[j] = 0.0;
            }
        } else {
            // Default neutral if no data
            average_signal[j] = 0.0;
        }
    }

    // Fill cutout with 0.0 or NaN? Usually 0.0 for signal safe
    for j in 0..cutout {
        average_signal[j] = 0.0;
    }

    average_signal.to_vec()
}

#[pyfunction]
#[pyo3(signature = (symbols_data, ema_len=28, hull_len=28, wma_len=28, dema_len=28, lsma_len=28, kama_len=28, ema_w=1.0, hma_w=1.0, wma_w=1.0, dema_w=1.0, lsma_w=1.0, kama_w=1.0, robustness="Medium", la=0.02, de=0.03, cutout=0, long_threshold=0.1, short_threshold=-0.1, _strategy_mode=false))]
pub fn compute_atc_signals_batch_cpu<'py>(
    py: Python<'py>,
    symbols_data: &Bound<'py, PyDict>,
    ema_len: usize,
    hull_len: usize,
    wma_len: usize,
    dema_len: usize,
    lsma_len: usize,
    kama_len: usize,
    ema_w: f64,
    hma_w: f64,
    wma_w: f64,
    dema_w: f64,
    lsma_w: f64,
    kama_w: f64,
    robustness: &str,
    la: f64,
    de: f64,
    cutout: usize,
    long_threshold: f64,
    short_threshold: f64,
    _strategy_mode: bool,
) -> PyResult<Bound<'py, PyDict>> {
    // 1. Extract data from Python Dictionary
    let mut symbols_vec = Vec::new();
    for (key, value) in symbols_data.iter() {
        let sym: String = key.extract()?;
        let prices: PyReadonlyArray1<f64> = value.extract()?;
        // Convert to standard Vec to release Python GIL during parallel processing
        let prices_vec = prices.as_slice()?.to_vec();
        symbols_vec.push((sym, prices_vec));
    }

    // 2. Process in parallel using Rayon
    // We release the GIL to allow true parallelism
    let results: Vec<(String, Vec<f64>)> = py.allow_threads(|| {
        symbols_vec
            .into_par_iter()
            .map(|(sym, prices)| {
                let res = process_single_symbol(
                    prices,
                    ema_len,
                    hull_len,
                    wma_len,
                    dema_len,
                    lsma_len,
                    kama_len,
                    ema_w,
                    hma_w,
                    wma_w,
                    dema_w,
                    lsma_w,
                    kama_w,
                    robustness,
                    la,
                    de,
                    cutout,
                    long_threshold,
                    short_threshold,
                );
                (sym, res)
            })
            .collect()
    });

    // 3. Convert back to Python Dictionary
    let res_dict = PyDict::new(py);
    for (sym, sig_vec) in results {
        let py_array = PyArray1::from_vec(py, sig_vec);
        res_dict.set_item(sym, py_array)?;
    }

    Ok(res_dict)
}
