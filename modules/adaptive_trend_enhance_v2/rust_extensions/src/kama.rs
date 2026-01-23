use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use ndarray::{Array1, ArrayView1};

pub fn calculate_kama_internal(
    prices_arr: ArrayView1<f64>,
    length: usize,
) -> Array1<f64> {
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
            if j == 0 {
                continue;
            }
            noise += (prices_arr[j] - prices_arr[j - 1]).abs();
        }
        
        let signal = (prices_arr[i] - prices_arr[i - length]).abs();
        let ratio = if noise == 0.0 { 0.0 } else { signal / noise };
        
        let smooth = (ratio * (fast - slow) + slow).powi(2);
        
        let prev_kama = if kama[i - 1].is_nan() {
            prices_arr[i]
        } else {
            kama[i - 1]
        };
        
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
}
