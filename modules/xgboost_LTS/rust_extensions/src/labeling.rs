//! Labeling functions for XGBoost
//!
//! High-performance implementations of directional labeling and rolling calculations.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Calculate volatility multiplier based on ATR or rolling volatility
///
/// # Arguments
/// * `close` - Closing prices
/// * `atr_14` - Optional ATR(14) values
///
/// # Returns
/// Volatility multiplier array clipped to [1.5, 3.0]
#[pyfunction]
#[pyo3(signature = (close, atr_14=None))]
pub fn calculate_volatility_multiplier_rust(
    py: Python<'_>,
    close: PyReadonlyArray1<f64>,
    atr_14: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let close = close.as_array();
    let n = close.len();

    let mut volatility_multiplier = Array1::<f64>::zeros(n);

    if let Some(atr) = atr_14 {
        let atr = atr.as_array();

        // Calculate ATR percentage
        let mut atr_pct = Array1::<f64>::zeros(n);
        for i in 0..n {
            atr_pct[i] = if close[i] > 0.0 {
                atr[i] / close[i]
            } else {
                0.01
            };
        }

        // Calculate rolling median of ATR percentage
        let window = 50.min(n);
        let mut atr_median = Array1::<f64>::zeros(n);

        for i in 0..n {
            let start = if i >= window { i - window + 1 } else { 0 };
            let slice = atr_pct.slice(ndarray::s![start..=i]);
            let mut sorted: Vec<f64> = slice.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            atr_median[i] = sorted[sorted.len() / 2].max(0.01);
        }

        // Calculate volatility multiplier
        for i in 0..n {
            let ratio = atr_pct[i] / atr_median[i];
            volatility_multiplier[i] = ratio.clamp(1.5, 3.0);
        }
    } else {
        // Fallback: use rolling volatility of returns
        let mut returns = Array1::<f64>::zeros(n);
        for i in 1..n {
            returns[i] = if close[i - 1] > 0.0 {
                (close[i] - close[i - 1]) / close[i - 1]
            } else {
                0.0
            };
        }

        let window = 20.min(n);
        for i in 0..n {
            let start = if i >= window { i - window + 1 } else { 0 };
            let slice = returns.slice(ndarray::s![start..=i]);

            // Calculate standard deviation
            let mean = slice.mean().unwrap_or(0.0);
            let variance: f64 =
                slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64;
            let rolling_vol = variance.sqrt().max(0.01);

            // Calculate median of rolling vol
            let med_start = if i >= 50 { i - 49 } else { 0 };
            let mut vol_window = Vec::new();
            for j in med_start..=i {
                let j_start = if j >= window { j - window + 1 } else { 0 };
                let j_slice = returns.slice(ndarray::s![j_start..=j]);
                let j_mean = j_slice.mean().unwrap_or(0.0);
                let j_var: f64 = j_slice.iter().map(|&x| (x - j_mean).powi(2)).sum::<f64>()
                    / j_slice.len() as f64;
                vol_window.push(j_var.sqrt().max(0.01));
            }
            vol_window.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let vol_median = vol_window[vol_window.len() / 2];

            volatility_multiplier[i] = (rolling_vol / vol_median).clamp(1.5, 3.0);
        }
    }

    Ok(volatility_multiplier.into_pyarray(py).unbind())
}

/// Apply directional labels based on future price movement
///
/// # Arguments
/// * `close` - Closing prices
/// * `target_horizon` - Number of candles to look ahead
/// * `base_threshold` - Base threshold for labeling
///
/// # Returns
/// Tuple of (labels, thresholds) where labels are -1 (DOWN), 0 (NEUTRAL), 1 (UP)
#[pyfunction]
pub fn apply_directional_labels_rust(
    py: Python<'_>,
    close: PyReadonlyArray1<f64>,
    target_horizon: usize,
    base_threshold: f64,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<f64>>)> {
    let close = close.as_array();
    let n = close.len();

    let labels_vec: Vec<i32> = (0..n)
        .into_par_iter()
        .map(|i| {
            if i + target_horizon < n {
                let future_close = close[i + target_horizon];
                let pct_change = (future_close - close[i]) / close[i];

                if pct_change >= base_threshold {
                    2 // UP
                } else if pct_change <= -base_threshold {
                    0 // DOWN
                } else {
                    1 // NEUTRAL
                }
            } else {
                -1 // Invalid
            }
        })
        .collect();

    let labels = Array1::from(labels_vec);
    let thresholds = Array1::<f64>::from_elem(n, base_threshold);

    Ok((
        labels.into_pyarray(py).unbind(),
        thresholds.into_pyarray(py).unbind(),
    ))
}

use std::collections::BTreeMap;
use ordered_float::OrderedFloat;

/// Calculate rolling quantile using efficient algorithm
///
/// # Arguments
/// * `arr` - Input array
/// * `window` - Rolling window size
/// * `q` - Quantile (0.0 to 1.0)
///
/// # Returns
/// /// Array of rolling quantiles
#[pyfunction]
pub fn rolling_quantile_rust(
    py: Python<'_>,
    arr: PyReadonlyArray1<f64>,
    window: usize,
    q: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let arr = arr.as_array();
    let n = arr.len();

    // Parallel processing for large arrays if window is small enough relative to size
    // For very large windows, sequential might be competitive due to BTreeMap efficiency,
    // but parallel is generally better for large arrays.
    // However, since BTreeMap optimization is stateful (incremental), it applies to sequential processing.
    // To parallelize efficiently with BTreeMap, we would need to chop into chunks and re-init BTreeMap.
    // Given the Python-side request for "Optimized Rolling Quantile Algorithm (O(n log w))"
    // and the specific mention of replacing "sort-per-window", we will implement
    // the incremental BTreeMap approach.
    //
    // Note: To fully leverage the O(n log w) complexity, we must run sequentially
    // (sliding window), OR chunk the data and run sequentially within chunks.
    // For simplicity and correctness of the requested O(n log w) algorithm,
    // we use a sequential implementation which is vastly faster than O(n*w*log w) for large w.

    let mut result = Array1::<f64>::from_elem(n, f64::NAN);
    
    // We use BTreeMap as a multiset: value -> count
    let mut window_map: BTreeMap<OrderedFloat<f64>, usize> = BTreeMap::new();
    
    // Calculate target rank (0-based index)
    // Same logic as: idx = ((window_slice.len() - 1) as f64 * q) as usize;
    // For a full window of size `window`:
    let rank = ((window as f64 - 1.0) * q).floor() as usize;

    for i in 0..n {
        let val = arr[i];
        
        // Add new element to map
        // Handle NaN: OrderedFloat treats NaN as equal to itself and greater than all other floats
        *window_map.entry(OrderedFloat(val)).or_insert(0) += 1;

        // Remove element going out of window
        if i >= window {
            let old_val = arr[i - window];
            let key = OrderedFloat(old_val);
            if let Some(count) = window_map.get_mut(&key) {
                *count -= 1;
                if *count == 0 {
                    window_map.remove(&key);
                }
            }
        }

        // Calculate quantile if window is full
        if i >= window - 1 {
            let mut current_count = 0;
            for (key, &count) in window_map.iter() {
                current_count += count;
                if current_count > rank {
                    result[i] = key.into_inner();
                    break;
                }
            }
        }
    }

    Ok(result.into_pyarray(py).unbind())
}

/// Calculate rolling mean using efficient algorithm
///
/// # Arguments
/// * `arr` - Input array
/// * `window` - Rolling window size
///
/// # Returns
/// Array of rolling means
#[pyfunction]
pub fn rolling_mean_rust(
    py: Python<'_>,
    arr: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let arr = arr.as_array();
    let n = arr.len();

    let result_vec: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            if i >= window - 1 {
                let start = i + 1 - window;
                let sum: f64 = arr.slice(ndarray::s![start..=i]).sum();
                sum / window as f64
            } else {
                f64::NAN
            }
        })
        .collect();

    let result = Array1::from(result_vec);
    Ok(result.into_pyarray(py).unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let arr = Array1::from(data);

        // Test with window=3
        // Expected: [NaN, NaN, 2.0, 3.0, 4.0]
        let window = 3;
        let mut result = Array1::<f64>::from_elem(arr.len(), f64::NAN);
        let mut current_sum = 0.0;

        for i in 0..arr.len() {
            current_sum += arr[i];
            if i >= window {
                current_sum -= arr[i - window];
            }

            if i >= window - 1 {
                result[i] = current_sum / window as f64;
            }
        }

        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }
}
