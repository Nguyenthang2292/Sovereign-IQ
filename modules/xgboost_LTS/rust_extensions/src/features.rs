//! Feature engineering functions
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// Calculate price derived features: returns, log_volume, ranges
///
/// # Arguments
/// * `open` - Open prices
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `volume` - Volume
///
/// # Returns
/// Dictionary of feature arrays
#[pyfunction]
#[pyo3(signature = (open, high, low, close, volume))]
pub fn add_price_derived_features_rust<'py>(
    py: Python<'py>,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let open = open.as_array();
    let high = high.as_array();
    let low = low.as_array();
    let close = close.as_array();
    let volume = volume.as_array();
    let n = close.len();

    // Verify all arrays have same length
    if open.len() != n || high.len() != n || low.len() != n || volume.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All input arrays must have the same length",
        ));
    }

    let mut returns_1 = Array1::<f64>::from_elem(n, f64::NAN);
    let mut returns_5 = Array1::<f64>::from_elem(n, f64::NAN);
    let mut log_volume = Array1::<f64>::zeros(n);
    let mut high_low_range = Array1::<f64>::zeros(n);
    let mut close_open_diff = Array1::<f64>::zeros(n);

    // Get mutable slices for parallel writing
    // We assume contiguous memory for arrays which is true for Array1 unless sliced
    let s_r1 = returns_1.as_slice_mut().expect("returns_1 not contiguous");
    let s_r5 = returns_5.as_slice_mut().expect("returns_5 not contiguous");
    let s_lv = log_volume
        .as_slice_mut()
        .expect("log_volume not contiguous");
    let s_hlr = high_low_range
        .as_slice_mut()
        .expect("high_low_range not contiguous");
    let s_cod = close_open_diff
        .as_slice_mut()
        .expect("close_open_diff not contiguous");

    (0..n)
        .into_par_iter()
        .zip(s_r1.par_iter_mut())
        .zip(s_r5.par_iter_mut())
        .zip(s_lv.par_iter_mut())
        .zip(s_hlr.par_iter_mut())
        .zip(s_cod.par_iter_mut())
        .for_each(|(((((i, r1), r5), lv), hlr), cod)| {
            // returns_1
            if i >= 1 && close[i - 1] != 0.0 {
                *r1 = (close[i] - close[i - 1]) / close[i - 1];
            }

            // returns_5
            if i >= 5 && close[i - 5] != 0.0 {
                *r5 = (close[i] - close[i - 5]) / close[i - 5];
            }

            // log_volume
            *lv = (volume[i] + 1.0).ln();

            // high_low_range & close_open_diff
            if close[i] != 0.0 {
                *hlr = (high[i] - low[i]) / close[i];
                *cod = (close[i] - open[i]) / close[i];
            }
        });

    let results = pyo3::types::PyDict::new(py);
    results.set_item("returns_1", returns_1.into_pyarray_bound(py))?;
    results.set_item("returns_5", returns_5.into_pyarray_bound(py))?;
    results.set_item("log_volume", log_volume.into_pyarray_bound(py))?;
    results.set_item("high_low_range", high_low_range.into_pyarray_bound(py))?;
    results.set_item("close_open_diff", close_open_diff.into_pyarray_bound(py))?;

    Ok(results)
}

/// Calculate rolling standard deviation
#[pyfunction]
pub fn rolling_std_rust<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr = arr.as_array();
    let n = arr.len();

    let result_vec: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            if i >= window - 1 {
                let start = i + 1 - window;
                let slice = arr.slice(ndarray::s![start..=i]);
                // Calculate sample standard deviation (ddof=1)
                slice.std(1.0)
            } else {
                f64::NAN
            }
        })
        .collect();

    let result = Array1::from(result_vec);
    Ok(result.into_pyarray_bound(py))
}

/// Calculate rolling skewness
#[pyfunction]
pub fn rolling_skew_rust<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<f64>,
    window: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr = arr.as_array();
    let n = arr.len();

    let result_vec: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            if i >= window - 1 {
                let start = i + 1 - window;
                let slice = arr.slice(ndarray::s![start..=i]);

                // Calculate mean
                let mean = slice.mean().unwrap_or(0.0);

                // Calculate m2 and m3
                let mut m2 = 0.0;
                let mut m3 = 0.0;
                for &x in slice {
                    let diff = x - mean;
                    m2 += diff * diff;
                    m3 += diff * diff * diff;
                }

                if m2 == 0.0 || window < 3 {
                    0.0
                } else {
                    let variance = m2 / (window - 1) as f64;
                    let std_dev = variance.sqrt();
                    let n_f = window as f64;
                    let skew = (n_f * m3) / ((n_f - 1.0) * (n_f - 2.0) * std_dev.powi(3));
                    skew
                }
            } else {
                f64::NAN
            }
        })
        .collect();

    let result = Array1::from(result_vec);
    Ok(result.into_pyarray_bound(py))
}

/// Calculate percentage change
#[pyfunction]
pub fn pct_change_rust<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<f64>,
    period: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr = arr.as_array();
    let n = arr.len();

    let result_vec: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            if i >= period {
                let prev = arr[i - period];
                if prev != 0.0 {
                    (arr[i] - prev) / prev
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            }
        })
        .collect();

    let result = Array1::from(result_vec);
    Ok(result.into_pyarray_bound(py))
}

/// Calculate advanced features in batch
#[pyfunction]
#[pyo3(signature = (close, volume, returns_1, atr_14=None, rsi_14=None, sma_20=None, sma_50=None, sma_200=None))]
pub fn add_advanced_features_rust<'py>(
    py: Python<'py>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
    returns_1: PyReadonlyArray1<f64>,
    atr_14: Option<PyReadonlyArray1<f64>>,
    rsi_14: Option<PyReadonlyArray1<f64>>,
    sma_20: Option<PyReadonlyArray1<f64>>,
    sma_50: Option<PyReadonlyArray1<f64>>,
    sma_200: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let close = close.as_array();
    let volume = volume.as_array();
    let returns_1 = returns_1.as_array();
    let n = close.len();

    let results = pyo3::types::PyDict::new(py);

    // 1. Price Momentum (ROC)
    for period in [3, 5, 10, 20] {
        let mut roc = Array1::<f64>::from_elem(n, f64::NAN);
        let s_roc = roc.as_slice_mut().expect("roc not contiguous");
        s_roc.par_iter_mut().enumerate().for_each(|(i, val)| {
            if i >= period && close[i - period] != 0.0 {
                *val = (close[i] - close[i - period]) / close[i - period];
            }
        });
        results.set_item(format!("roc_{}", period), roc.into_pyarray_bound(py))?;
    }

    // 2. Volatility Ratios (ATR/Close) & atr_ratio lag prep
    let mut atr_ratio_arr = Array1::<f64>::from_elem(n, f64::NAN);
    if let Some(atr) = &atr_14 {
        let atr = atr.as_array();
        let s_atr_ratio = atr_ratio_arr
            .as_slice_mut()
            .expect("atr_ratio not contiguous");
        s_atr_ratio.par_iter_mut().enumerate().for_each(|(i, val)| {
            if close[i] != 0.0 {
                *val = atr[i] / close[i];
            }
        });
        results.set_item("atr_ratio", atr_ratio_arr.clone().into_pyarray_bound(py))?;
    }

    // 3. Relative Strength (Price vs SMA)
    if let Some(sma) = sma_20 {
        let sma = sma.as_array();
        let mut ratio = Array1::<f64>::from_elem(n, f64::NAN);
        let s_ratio = ratio.as_slice_mut().expect("ratio not contiguous");
        s_ratio.par_iter_mut().enumerate().for_each(|(i, val)| {
            if sma[i] != 0.0 {
                *val = close[i] / sma[i];
            }
        });
        results.set_item("price_to_SMA_20", ratio.into_pyarray_bound(py))?;
    }
    if let Some(sma) = sma_50 {
        let sma = sma.as_array();
        let mut ratio = Array1::<f64>::from_elem(n, f64::NAN);
        let s_ratio = ratio.as_slice_mut().expect("ratio not contiguous");
        s_ratio.par_iter_mut().enumerate().for_each(|(i, val)| {
            if sma[i] != 0.0 {
                *val = close[i] / sma[i];
            }
        });
        results.set_item("price_to_SMA_50", ratio.into_pyarray_bound(py))?;
    }
    if let Some(sma) = sma_200 {
        let sma = sma.as_array();
        let mut ratio = Array1::<f64>::from_elem(n, f64::NAN);
        let s_ratio = ratio.as_slice_mut().expect("ratio not contiguous");
        s_ratio.par_iter_mut().enumerate().for_each(|(i, val)| {
            if sma[i] != 0.0 {
                *val = close[i] / sma[i];
            }
        });
        results.set_item("price_to_SMA_200", ratio.into_pyarray_bound(py))?;
    }

    // 4. Rolling Statistics on Returns
    for window in [10, 20] {
        let mut roll_std = Array1::<f64>::from_elem(n, f64::NAN);
        let mut roll_skew = Array1::<f64>::from_elem(n, f64::NAN);

        let s_std = roll_std.as_slice_mut().expect("roll_std not contiguous");
        let s_skew = roll_skew.as_slice_mut().expect("roll_skew not contiguous");

        s_std
            .par_iter_mut()
            .zip(s_skew.par_iter_mut())
            .enumerate()
            .for_each(|(i, (val_std, val_skew))| {
                if i >= window - 1 {
                    let start = i + 1 - window;
                    let slice = returns_1.slice(ndarray::s![start..=i]);
                    // Calculate mean
                    let mean = slice.mean().unwrap_or(0.0);

                    // Calculate m2 and m3
                    let mut m2 = 0.0;
                    let mut m3 = 0.0;
                    for &x in slice {
                        let diff = x - mean;
                        m2 += diff * diff;
                        m3 += diff * diff * diff;
                    }

                    // Std Dev
                    let variance = m2 / (window - 1) as f64;
                    let std_dev = if variance > 0.0 { variance.sqrt() } else { 0.0 };
                    *val_std = std_dev;

                    // Skew
                    if m2 > 0.0 && window >= 3 {
                        let n_f = window as f64;
                        let skew = (n_f * m3) / ((n_f - 1.0) * (n_f - 2.0) * std_dev.powi(3));
                        *val_skew = skew;
                    } else if window >= 3 {
                        *val_skew = 0.0;
                    }
                }
            });

        results.set_item(
            format!("rolling_std_{}", window),
            roll_std.into_pyarray_bound(py),
        )?;
        results.set_item(
            format!("rolling_skew_{}", window),
            roll_skew.into_pyarray_bound(py),
        )?;
    }

    // 5. Lag Features
    for lag in 1..=3 {
        let mut lag_arr = Array1::<f64>::from_elem(n, f64::NAN);
        let s_lag = lag_arr.as_slice_mut().expect("lag_arr not contiguous");
        s_lag.par_iter_mut().enumerate().for_each(|(i, val)| {
            if i >= lag {
                *val = returns_1[i - lag];
            }
        });
        results.set_item(
            format!("returns_1_lag_{}", lag),
            lag_arr.into_pyarray_bound(py),
        )?;
    }

    if let Some(rsi) = rsi_14 {
        let rsi = rsi.as_array();
        for lag in 1..=3 {
            let mut lag_arr = Array1::<f64>::from_elem(n, f64::NAN);
            let s_lag = lag_arr.as_slice_mut().expect("lag_arr not contiguous");
            s_lag.par_iter_mut().enumerate().for_each(|(i, val)| {
                if i >= lag {
                    *val = rsi[i - lag];
                }
            });
            results.set_item(
                format!("RSI_14_lag_{}", lag),
                lag_arr.into_pyarray_bound(py),
            )?;
        }
    }

    let mut log_volume = Array1::<f64>::zeros(n);
    let s_lv = log_volume
        .as_slice_mut()
        .expect("log_volume not contiguous");
    s_lv.par_iter_mut().enumerate().for_each(|(i, val)| {
        *val = (volume[i] + 1.0).ln();
    });

    for lag in 1..=3 {
        let mut lag_arr = Array1::<f64>::from_elem(n, f64::NAN);
        let s_lag = lag_arr.as_slice_mut().expect("lag_arr not contiguous");
        s_lag.par_iter_mut().enumerate().for_each(|(i, val)| {
            if i >= lag {
                *val = log_volume[i - lag];
            }
        });
        results.set_item(
            format!("log_volume_lag_{}", lag),
            lag_arr.into_pyarray_bound(py),
        )?;
    }

    if atr_14.is_some() {
        for lag in 1..=3 {
            let mut lag_arr = Array1::<f64>::from_elem(n, f64::NAN);
            let s_lag = lag_arr.as_slice_mut().expect("lag_arr not contiguous");
            s_lag.par_iter_mut().enumerate().for_each(|(i, val)| {
                if i >= lag {
                    *val = atr_ratio_arr[i - lag];
                }
            });
            results.set_item(
                format!("atr_ratio_lag_{}", lag),
                lag_arr.into_pyarray_bound(py),
            )?;
        }
    }

    Ok(results)
}
