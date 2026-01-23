use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use ndarray::{Array1, ArrayView1};


/// Calculate EMA (Exponential Moving Average) internally
///
/// # Arguments
///
/// * `prices_arr` - Array view of price values
/// * `length` - Period for EMA calculation
///
/// # Returns
///
/// Array1<f64> containing calculated EMA values
#[allow(dead_code)]
pub fn calculate_ema_internal(
    prices_arr: ArrayView1<f64>,
    length: usize,
) -> Array1<f64> {
    let n = prices_arr.len();
    let mut ema = Array1::<f64>::from_elem(n, f64::NAN);

    if n < 1 {
        return ema;
    }

    let alpha = 2.0 / (length as f64 + 1.0);
    ema[0] = prices_arr[0];

    for i in 1..n {
        ema[i] = alpha * prices_arr[i] + (1.0 - alpha) * ema[i - 1];
    }

    ema
}

/// Calculate EMA (Exponential Moving Average) for Python
///
/// # Arguments
///
/// * `prices` - Array of price values
/// * `length` - Period for EMA calculation
///
/// # Returns
///
/// PyArray1<f64> containing calculated EMA values
///
/// # Example
///
/// ```python
/// import numpy as np
/// from atc_rust import calculate_ema_rust
///
/// prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0])
/// ema = calculate_ema_rust(prices, 3)
/// ```
#[pyfunction]
pub fn calculate_ema_rust<'py>(
    _py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    length: usize,
) -> Bound<'py, PyArray1<f64>> {
    let prices_arr = prices.as_array();
    let ema = calculate_ema_internal(prices_arr, length);
    PyArray1::from_array(_py, &ema)
}

/// Calculate WMA (Weighted Moving Average) internally
///
/// # Arguments
///
/// * `prices_arr` - Array view of price values
/// * `length` - Period for WMA calculation
///
/// # Returns
///
/// Array1<f64> containing calculated WMA values
#[allow(dead_code)]
pub fn calculate_wma_internal(
    prices_arr: ArrayView1<f64>,
    length: usize,
) -> Array1<f64> {
    let n = prices_arr.len();
    let mut wma = Array1::<f64>::from_elem(n, f64::NAN);

    if n < length {
        return wma;
    }

    let denominator = (length * (length + 1)) as f64 / 2.0;

    for i in (length - 1)..n {
        let mut weighted_sum = 0.0;
        for j in 0..length {
            let weight = (length - j) as f64;
            weighted_sum += prices_arr[i - j] * weight;
        }
        wma[i] = weighted_sum / denominator;
    }

    wma
}

/// Calculate WMA (Weighted Moving Average) for Python
///
/// # Arguments
///
/// * `prices` - Array of price values
/// * `length` - Period for WMA calculation
///
/// # Returns
///
/// PyArray1<f64> containing calculated WMA values
///
/// # Example
///
/// ```python
/// import numpy as np
/// from atc_rust import calculate_wma_rust
///
/// prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0])
/// wma = calculate_wma_rust(prices, 3)
/// ```
#[pyfunction]
pub fn calculate_wma_rust<'py>(
    _py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    length: usize,
) -> Bound<'py, PyArray1<f64>> {
    let prices_arr = prices.as_array();
    let wma = calculate_wma_internal(prices_arr, length);
    PyArray1::from_array(_py, &wma)
}

/// Calculate DEMA (Double Exponential Moving Average) internally
///
/// # Arguments
///
/// * `prices_arr` - Array view of price values
/// * `length` - Period for DEMA calculation
///
/// # Returns
///
/// Array1<f64> containing calculated DEMA values
#[allow(dead_code)]
pub fn calculate_dema_internal(
    prices_arr: ArrayView1<f64>,
    length: usize,
) -> Array1<f64> {
    let ema1 = calculate_ema_internal(prices_arr, length);
    let ema2 = calculate_ema_internal(ema1.view(), length);

    // DEMA = 2 * EMA1 - EMA2
    2.0 * &ema1 - &ema2
}

/// Calculate DEMA (Double Exponential Moving Average) for Python
///
/// # Arguments
///
/// * `prices` - Array of price values
/// * `length` - Period for DEMA calculation
///
/// # Returns
///
/// PyArray1<f64> containing calculated DEMA values
///
/// # Example
///
/// ```python
/// import numpy as np
/// from atc_rust import calculate_dema_rust
///
/// prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0])
/// dema = calculate_dema_rust(prices, 3)
/// ```
#[pyfunction]
pub fn calculate_dema_rust<'py>(
    _py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    length: usize,
) -> Bound<'py, PyArray1<f64>> {
    let prices_arr = prices.as_array();
    let dema = calculate_dema_internal(prices_arr, length);
    PyArray1::from_array(_py, &dema)
}

/// Calculate LSMA (Least Squares Moving Average) internally
///
/// # Arguments
///
/// * `prices_arr` - Array view of price values
/// * `length` - Period for LSMA calculation
///
/// # Returns
///
/// Array1<f64> containing calculated LSMA values
#[allow(dead_code)]
pub fn calculate_lsma_internal(
    prices_arr: ArrayView1<f64>,
    length: usize,
) -> Array1<f64> {
    let n = prices_arr.len();
    let mut lsma = Array1::<f64>::from_elem(n, f64::NAN);

    if n < length {
        return lsma;
    }

    let x_mean = (length as f64 - 1.0) / 2.0;
    let mut x_sq_sum = 0.0;
    for i in 0..length {
        let x = i as f64;
        x_sq_sum += (x - x_mean).powi(2);
    }

    for i in (length - 1)..n {
        let mut y_sum = 0.0;
        for j in 0..length {
            y_sum += prices_arr[i - length + 1 + j];
        }
        let y_mean = y_sum / length as f64;

        let mut xy_sum = 0.0;
        for j in 0..length {
            let x = j as f64;
            let y = prices_arr[i - length + 1 + j];
            xy_sum += (x - x_mean) * (y - y_mean);
        }

        let b = xy_sum / x_sq_sum;
        let a = y_mean - b * x_mean;
        lsma[i] = a + b * (length as f64 - 1.0);
    }

    lsma
}

/// Calculate LSMA (Least Squares Moving Average) for Python
///
/// # Arguments
///
/// * `prices` - Array of price values
/// * `length` - Period for LSMA calculation
///
/// # Returns
///
/// PyArray1<f64> containing calculated LSMA values
///
/// # Example
///
/// ```python
/// import numpy as np
/// from atc_rust import calculate_lsma_rust
///
/// prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0])
/// lsma = calculate_lsma_rust(prices, 3)
/// ```
#[pyfunction]
pub fn calculate_lsma_rust<'py>(
    _py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    length: usize,
) -> Bound<'py, PyArray1<f64>> {
    let prices_arr = prices.as_array();
    let lsma = calculate_lsma_internal(prices_arr, length);
    PyArray1::from_array(_py, &lsma)
}

/// Calculate SMA (Simple Moving Average) internally - helper for HMA
///
/// # Arguments
///
/// * `prices_arr` - Array view of price values
/// * `length` - Period for SMA calculation
///
/// # Returns
///
/// Array1<f64> containing calculated SMA values
pub fn calculate_sma_internal(
    prices_arr: ArrayView1<f64>,
    length: usize,
) -> Array1<f64> {
    let n = prices_arr.len();
    let mut sma = Array1::<f64>::from_elem(n, f64::NAN);

    if n < length {
        return sma;
    }

    for i in (length - 1)..n {
        let mut sum = 0.0;
        for j in 0..length {
            sum += prices_arr[i - j];
        }
        sma[i] = sum / length as f64;
    }

    sma
}

/// Calculate HMA (Hull Moving Average) internally
///
/// HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
///
/// # Arguments
///
/// * `prices_arr` - Array view of price values
/// * `length` - Period for HMA calculation
///
/// # Returns
///
/// Array1<f64> containing calculated HMA values
#[allow(dead_code)]
pub fn calculate_hma_internal(
    prices_arr: ArrayView1<f64>,
    length: usize,
) -> Array1<f64> {
    let n = prices_arr.len();

    if n < length {
        return Array1::<f64>::from_elem(n, f64::NAN);
    }

    let half_length = length / 2;
    let sqrt_length = (length as f64).sqrt() as usize;

    // Calculate WMA(n/2)
    let wma_half = calculate_wma_internal(prices_arr, half_length);

    // Calculate WMA(n)
    let wma_full = calculate_wma_internal(prices_arr, length);

    // Calculate 2 * WMA(n/2) - WMA(n)
    let diff = 2.0 * &wma_half - &wma_full;

    // Calculate WMA(diff, sqrt(n))
    calculate_wma_internal(diff.view(), sqrt_length)
}

/// Calculate HMA (Hull Moving Average) for Python
///
/// # Arguments
///
/// * `prices` - Array of price values
/// * `length` - Period for HMA calculation
///
/// # Returns
///
/// PyArray1<f64> containing calculated HMA values
///
/// # Example
///
/// ```python
/// import numpy as np
/// from atc_rust import calculate_hma_rust
///
/// prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0])
/// hma = calculate_hma_rust(prices, 5)
/// ```
#[pyfunction]
pub fn calculate_hma_rust<'py>(
    _py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
    length: usize,
) -> Bound<'py, PyArray1<f64>> {
    let prices_arr = prices.as_array();
    let hma = calculate_hma_internal(prices_arr, length);
    PyArray1::from_array(_py, &hma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ema_basic() {
        let prices = array![10.0, 11.0, 12.0, 11.0, 10.0];
        let ema = calculate_ema_internal(prices.view(), 3);
        assert!(!ema[4].is_nan());
        assert!(ema[0] == 10.0);
    }

    #[test]
    fn test_wma_basic() {
        let prices = array![10.0, 11.0, 12.0, 11.0, 10.0];
        let wma = calculate_wma_internal(prices.view(), 3);
        assert!(!wma[4].is_nan());
        assert!(wma[0].is_nan());
    }

    #[test]
    fn test_dema_basic() {
        let prices = array![10.0, 11.0, 12.0, 11.0, 10.0];
        let dema = calculate_dema_internal(prices.view(), 2);
        assert!(!dema[4].is_nan());
    }

    #[test]
    fn test_lsma_basic() {
        let prices = array![10.0, 11.0, 12.0, 11.0, 10.0];
        let lsma = calculate_lsma_internal(prices.view(), 3);
        assert!(!lsma[4].is_nan());
        assert!(lsma[0].is_nan());
    }

    #[test]
    fn test_hma_basic() {
        let prices = array![10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0];
        let hma = calculate_hma_internal(prices.view(), 5);
        // HMA requires more data points
        assert!(hma.len() == 7);
    }

    #[test]
    fn test_sma_basic() {
        let prices = array![10.0, 11.0, 12.0, 11.0, 10.0];
        let sma = calculate_sma_internal(prices.view(), 3);
        assert!(!sma[4].is_nan());
        assert!((sma[2] - 11.0).abs() < 1e-10);
    }
}
