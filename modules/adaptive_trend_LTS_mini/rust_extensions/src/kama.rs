use ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Calculate KAMA values aligned with source `modules/adaptive_trend`.
pub fn calculate_kama_internal(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let n = prices_arr.len();
    let mut kama = Array1::<f64>::from_elem(n, f64::NAN);

    if n < 1 {
        return kama;
    }

    let fast = 0.666;
    let slow = 0.064;

    for i in 0..n {
        if i == 0 {
            kama[i] = prices_arr[i];
            continue;
        }

        if i < length {
            kama[i] = kama[i - 1];
            continue;
        }

        let mut noise = 0.0;
        for j in (i - length + 1)..=i {
            noise += (prices_arr[j] - prices_arr[j - 1]).abs();
        }

        let signal = (prices_arr[i] - prices_arr[i - length]).abs();
        let ratio = if noise == 0.0 { 0.0 } else { signal / noise };

        let smooth = (ratio * (fast - slow) + slow).powi(2);

        let prev_kama = if kama[i - 1].is_nan() { prices_arr[i] } else { kama[i - 1] };

        kama[i] = prev_kama + (smooth * (prices_arr[i] - prev_kama));
    }
    kama
}

/// Calculate KAMA (Kaufman Adaptive Moving Average) values specifically for ATC.
///
/// # Arguments
///
/// * `prices` - Array of price values
/// * `length` - Efficiency ratio length
///
/// # Returns
///
/// PyArray1<f64> containing calculated KAMA values
#[pyfunction]
pub fn calculate_kama_rust<'py>(
    _py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    length: usize,
) -> Bound<'py, PyArray1<f64>> {
    let prices_arr = prices.as_array();
    let kama = calculate_kama_internal(prices_arr, length);
    PyArray1::from_array(_py, &kama)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kama_logic() {
        let prices = array![10.0, 11.0, 12.0, 11.0, 10.0];
        let kama = calculate_kama_internal(prices.view(), 2);
        assert!(!kama[4].is_nan());
    }

    #[test]
    fn test_kama_simd_large_array() {
        // Test SIMD optimizations with large array
        let n = 5000;
        let prices = Array1::from_iter((0..n).map(|i| 100.0 + (i as f64) * 0.1));
        let kama = calculate_kama_internal(prices.view(), 20);

        // Verify all values are calculated
        assert!(!kama[n - 1].is_nan());
        assert!(kama[0] == prices[0]);

        // Verify KAMA values are reasonable
        assert!(kama[n - 1] > 0.0);
    }

    #[test]
    fn test_kama_large_window_stability() {
        let n = 2000;
        let length = 50;
        let prices = Array1::from_iter((0..n).map(|i| 100.0 + (i as f64) * 0.1));
        let kama = calculate_kama_internal(prices.view(), length);

        // Verify large-window calculation remains valid
        assert!(!kama[n - 1].is_nan());
        assert!(kama[0] == prices[0]);

        // Correctness check only
    }

    #[test]
    fn test_kama_edge_cases() {
        // Test with small array (should not use parallel)
        let prices = array![10.0, 11.0, 12.0];
        let kama = calculate_kama_internal(prices.view(), 2);
        assert!(!kama[2].is_nan());

        // Test with empty array
        let prices_empty = Array1::<f64>::from_vec(vec![]);
        let kama_empty = calculate_kama_internal(prices_empty.view(), 2);
        assert_eq!(kama_empty.len(), 0);
    }
}
