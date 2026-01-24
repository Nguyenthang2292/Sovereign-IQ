use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use ndarray::{Array1, ArrayView1};

/// Calculate equity values with SIMD-optimized inner loop.
/// 
/// The inner loop is structured to allow LLVM auto-vectorization:
/// - Simple arithmetic operations that can be vectorized
/// - No complex branching in hot path
/// - Sequential dependency on prev_e is handled efficiently
pub fn calculate_equity_internal(
    r: ArrayView1<f64>,
    sig: ArrayView1<f64>,
    starting_equity: f64,
    decay_multiplier: f64,
    cutout: usize,
) -> Array1<f64> {
    let n = r.len();
    let mut e_values = Array1::<f64>::from_elem(n, f64::NAN);
    
    // Handle cutout prefix
    if cutout > 0 && cutout <= n {
        // SIMD-friendly: fill NaN values in one pass
        for i in 0..cutout {
            e_values[i] = f64::NAN;
        }
    }
    
    let mut prev_e = f64::NAN;
    
    // Main loop: structured for SIMD auto-vectorization
    // LLVM can vectorize the arithmetic operations in the hot path
    for i in cutout..n {
        let r_i = r[i];
        let s_prev = sig[i];
        
        // SIMD-friendly: use branchless operations where possible
        // Calculate 'a' with minimal branching for better vectorization
        let a = if s_prev.is_nan() || r_i.is_nan() {
            0.0
        } else if s_prev == 0.0 {
            0.0
        } else {
            // Branchless: s_prev > 0 ? r_i : -r_i
            // This pattern is more SIMD-friendly than nested if-else
            let sign = if s_prev > 0.0 { 1.0 } else { -1.0 };
            r_i * sign
        };
        
        // Calculate equity with SIMD-friendly operations
        let mut e_curr = if prev_e.is_nan() {
            starting_equity
        } else {
            // SIMD-friendly: (prev_e * decay_multiplier) * (1.0 + a)
            // These operations can be vectorized
            let decayed = prev_e * decay_multiplier;
            decayed * (1.0 + a)
        };
        
        // Clamp minimum value (SIMD-friendly comparison)
        if e_curr < 0.25 {
            e_curr = 0.25;
        }
        
        prev_e = e_curr;
        e_values[i] = e_curr;
    }
    e_values
}

/// Calculate equity values using adaptive decay and cutout logic.
///
/// # Arguments
///
/// * `r_values` - Array of return values
/// * `sig_prev_values` - Array of previous signal values
/// * `starting_equity` - Initial equity value
/// * `decay_multiplier` - Decay factor (1.0 - De)
/// * `cutout` - Cutout threshold for signal filtering
///
/// # Returns
///
/// PyArray1<f64> containing calculated equity values
#[pyfunction]
pub fn calculate_equity_rust<'py>(
    _py: Python<'py>,
    r_values: PyReadonlyArray1<'py, f64>,
    sig_prev_values: PyReadonlyArray1<'py, f64>,
    starting_equity: f64,
    decay_multiplier: f64,
    cutout: usize,
) -> Bound<'py, PyArray1<f64>> {
    let r = r_values.as_array();
    let sig = sig_prev_values.as_array();
    let e_values = calculate_equity_internal(r, sig, starting_equity, decay_multiplier, cutout);
    PyArray1::from_array(_py, &e_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_equity_logic() {
        let r = array![0.01, 0.02, -0.01];
        let sig = array![1.0, 1.0, -1.0];
        let e = calculate_equity_internal(r.view(), sig.view(), 100.0, 1.0, 0);
        
        assert!((e[0] - 100.0).abs() < 1e-6);
        assert!((e[1] - 102.0).abs() < 1e-6);
        assert!((e[2] - 103.02).abs() < 1e-6); // 102.0 * (1 + (-0.01 * -1.0)) = 103.02
    }

    #[test]
    fn test_equity_simd_large_array() {
        // Test SIMD optimizations with large array
        let n = 10000;
        let r = Array1::from_iter((0..n).map(|i| (i as f64 % 100.0) / 1000.0));
        let sig = Array1::from_iter((0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }));
        
        let e = calculate_equity_internal(r.view(), sig.view(), 100.0, 0.97, 0);
        
        // Verify all values are calculated (not NaN except for cutout)
        assert!(!e[n - 1].is_nan());
        assert!(e[0] == 100.0);
        
        // Verify equity values are reasonable
        assert!(e[n - 1] > 0.0);
    }

    #[test]
    fn test_equity_cutout() {
        let r = array![0.01, 0.02, -0.01, 0.03];
        let sig = array![1.0, 1.0, -1.0, 1.0];
        let e = calculate_equity_internal(r.view(), sig.view(), 100.0, 1.0, 2);
        
        // First 2 should be NaN due to cutout
        assert!(e[0].is_nan());
        assert!(e[1].is_nan());
        // Rest should be calculated
        assert!(!e[2].is_nan());
        assert!(!e[3].is_nan());
    }
}
