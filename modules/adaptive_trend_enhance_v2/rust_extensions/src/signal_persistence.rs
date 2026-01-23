use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use ndarray::{Array1, ArrayView1};

pub fn process_signal_persistence_internal(
    up_arr: ArrayView1<bool>,
    down_arr: ArrayView1<bool>,
) -> Array1<i8> {
    let n = up_arr.len();
    let mut out = Array1::<i8>::zeros(n);
    let mut current_sig: i8 = 0;
    
    for i in 0..n {
        if up_arr[i] {
            current_sig = 1;
        } else if down_arr[i] {
            current_sig = -1;
        }
        out[i] = current_sig;
    }
    out
}

/// Apply signal persistence logic (state tracking).
///
/// sig = 0
/// if up: sig = 1
/// elif down: sig = -1
/// else: sig = sig[prev]
///
/// # Arguments
///
/// * `up` - Boolean array of bullish crossover events
/// * `down` - Boolean array of bearish crossunder events
///
/// # Returns
///
/// PyArray1<i8> containing persistent signals (1, -1, or last state)
#[pyfunction]
pub fn process_signal_persistence_rust<'py>(
    _py: Python<'py>,
    up: PyReadonlyArray1<'py, bool>,
    down: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i8>> {
    let up_arr = up.as_array();
    let down_arr = down.as_array();
    let out = process_signal_persistence_internal(up_arr, down_arr);
    PyArray1::from_array(_py, &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_persistence_logic() {
        let up = array![true, false, false, false];
        let down = array![false, false, true, false];
        let out = process_signal_persistence_internal(up.view(), down.view());
        
        assert_eq!(out[0], 1);
        assert_eq!(out[1], 1);
        assert_eq!(out[2], -1);
        assert_eq!(out[3], -1);
    }
}
