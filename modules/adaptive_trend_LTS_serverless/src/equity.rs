use ndarray::{Array1, ArrayView1};

/// Calculate exponential growth factor over time.
///
/// IMPORTANT: This function must stay bit-compatible with legacy Python/Pine logic.
///
/// Legacy/Pine semantics are:
/// - `bar_index_for_growth = 1` when `i == 0`
/// - otherwise `bar_index_for_growth = i`
/// - bars `< cutout` use `1.0`
/// - bars `>= cutout` use `exp(lambda_val * (bar_index_for_growth - cutout))`
///
/// This means when `cutout == 0`, the first two elements (`i=0` and `i=1`) are
/// intentionally equal because both use `bar_index_for_growth = 1`.
///
/// Do not "simplify" this to `bar_index = i` unless algorithm parity requirements
/// are intentionally changed and all downstream strategy baselines are re-validated.
pub fn exp_growth(lambda_val: f64, n: usize, cutout: usize) -> Array1<f64> {
    // Intentional direct allocation: growth is computed once per symbol/timeframe and
    // consumed immediately; keeping this path independent from thread-local signal
    // buffers avoids cross-module coupling in a low-cost O(n) operation.
    let mut growth = Array1::<f64>::from_elem(n, 1.0);

    for i in 0..n {
        // Pine parity contract: first bar is treated as bar_index = 1.
        let bar_index = if i == 0 { 1.0 } else { i as f64 };
        if i >= cutout {
            let exponent = lambda_val * (bar_index - cutout as f64);
            growth[i] = exponent.exp();
        }
    }

    growth
}

/// Calculate equity values for the ATC algorithm.
pub fn calculate_equity(
    r: ArrayView1<f64>,
    sig: ArrayView1<f64>,
    starting_equity: f64,
    decay_multiplier: f64,
    cutout: usize,
    floor_val: f64,
) -> Array1<f64> {
    let n = r.len();
    // Pre-fill with NaN so prefix [0, cutout) is already in the desired state.
    let mut e_values = Array1::<f64>::from_elem(n, f64::NAN);

    let mut prev_e = f64::NAN;

    // Main loop
    for i in cutout..n {
        let r_i = r[i];
        let s_prev = sig[i];

        let a = if s_prev.is_nan() || r_i.is_nan() || s_prev == 0.0 {
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

#[cfg(test)]
mod tests {
    use super::exp_growth;

    #[test]
    fn test_exp_growth_matches_legacy_first_bar_contract() {
        let lambda = 0.00002;
        let growth = exp_growth(lambda, 5, 0);

        let expected_first = lambda.exp();
        assert!(
            (growth[0] - expected_first).abs() < 1e-15,
            "i=0 must map to bar_index=1 per legacy contract"
        );
        assert!(
            (growth[1] - expected_first).abs() < 1e-15,
            "i=1 must equal i=0 when cutout=0"
        );
        assert!(growth[2] > growth[1]);
    }

    #[test]
    fn test_exp_growth_cutout_prefix_remains_one() {
        let lambda = 0.00002;
        let growth = exp_growth(lambda, 5, 2);
        assert_eq!(growth[0], 1.0);
        assert_eq!(growth[1], 1.0);
        // Boundary case: i == cutout yields exponent 0.0, so exp(0) = 1.0.
        assert_eq!(growth[2], 1.0);
        assert!(growth[3] > growth[2]);
    }
}
