use ndarray::{Array1, ArrayView1};

/// Calculate exponential growth factor over time.
///
/// Matches Python exp_growth logic:
/// - bar_index 0 treated as 1
/// - bars < cutout use 1.0
/// - bars >= cutout use exp(lambda_val * (bar_index - cutout))
pub fn exp_growth(lambda_val: f64, n: usize, cutout: usize) -> Array1<f64> {
    let mut growth = Array1::<f64>::from_elem(n, 1.0);

    for i in 0..n {
        let bar_index = if i == 0 { 1.0 } else { i as f64 };
        if i >= cutout {
            let exponent = lambda_val * (bar_index - cutout as f64);
            growth[i] = exponent.exp();
        }
    }

    growth
}

/// Calculate equity values with SIMD-optimized inner loop.
pub fn calculate_equity(
    r: ArrayView1<f64>,
    sig: ArrayView1<f64>,
    starting_equity: f64,
    decay_multiplier: f64,
    cutout: usize,
    floor_val: f64,
) -> Array1<f64> {
    let n = r.len();
    let mut e_values = Array1::<f64>::from_elem(n, f64::NAN);

    // Handle cutout prefix
    if cutout > 0 && cutout <= n {
        for i in 0..cutout {
            e_values[i] = f64::NAN;
        }
    }

    let mut prev_e = f64::NAN;

    // Main loop
    for i in cutout..n {
        let r_i = r[i];
        let s_prev = sig[i];

        let a = if s_prev.is_nan() || r_i.is_nan() {
            0.0
        } else if s_prev == 0.0 {
            0.0
        } else {
            let sign = if s_prev > 0.0 { 1.0 } else { -1.0 };
            r_i * sign
        };

        let mut e_curr = if prev_e.is_nan() {
            starting_equity
        } else {
            let decayed = prev_e * decay_multiplier;
            decayed * (1.0 + a)
        };

        // Clamp minimum value
        if e_curr < floor_val {
            e_curr = floor_val;
        }

        prev_e = e_curr;
        e_values[i] = e_curr;
    }
    e_values
}
