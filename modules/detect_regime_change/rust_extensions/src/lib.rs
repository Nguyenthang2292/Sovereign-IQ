use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

fn l2_cost(sum_y: &[f64], sum_y_sq: &[f64], start: usize, end: usize) -> f64 {
    let n = (end - start) as f64;
    let s_y = sum_y[end] - sum_y[start];
    let s_y_sq = sum_y_sq[end] - sum_y_sq[start];
    s_y_sq - (s_y * s_y) / n
}

fn normal_cost(sum_y: &[f64], sum_y_sq: &[f64], start: usize, end: usize) -> f64 {
    let n = (end - start) as f64;
    let sse = l2_cost(sum_y, sum_y_sq, start, end);
    let variance = (sse / n).max(1e-12);
    n * variance.ln()
}

fn segment_cost(
    sum_y: &[f64],
    sum_y_sq: &[f64],
    start: usize,
    end: usize,
    model: &str,
) -> PyResult<f64> {
    match model {
        "l2" => Ok(l2_cost(sum_y, sum_y_sq, start, end)),
        "normal" => Ok(normal_cost(sum_y, sum_y_sq, start, end)),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported Rust PELT model: {model}. Supported: l2, normal"
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (returns, penalty, min_size, model = "l2"))]
fn detect_change_points_pelt_rs(
    returns: Vec<f64>,
    penalty: f64,
    min_size: usize,
    model: &str,
) -> PyResult<Vec<usize>> {
    let n = returns.len();
    if n < min_size * 2 {
        return Ok(vec![]);
    }

    let normalized_model = model.to_ascii_lowercase();

    let mut sum_y = vec![0.0; n + 1];
    let mut sum_y_sq = vec![0.0; n + 1];
    for i in 0..n {
        sum_y[i + 1] = sum_y[i] + returns[i];
        sum_y_sq[i + 1] = sum_y_sq[i] + returns[i] * returns[i];
    }

    let mut f = vec![0.0; n + 1];
    f[0] = -penalty;

    let mut cp = vec![0; n + 1];
    let mut r_set = vec![0];

    for t in min_size..=n {
        let mut min_val = f64::MAX;
        let mut min_idx = 0;

        for &tau in &r_set {
            if t - tau >= min_size {
                let cost = segment_cost(&sum_y, &sum_y_sq, tau, t, &normalized_model)?;
                let val = f[tau] + cost + penalty;
                if val < min_val {
                    min_val = val;
                    min_idx = tau;
                }
            }
        }

        f[t] = min_val;
        cp[t] = min_idx;

        let mut next_r = Vec::new();
        for &tau in &r_set {
            if t - tau >= min_size {
                let cost = segment_cost(&sum_y, &sum_y_sq, tau, t, &normalized_model)?;
                if f[tau] + cost <= f[t] {
                    next_r.push(tau);
                }
            } else {
                next_r.push(tau);
            }
        }
        next_r.push(t);
        r_set = next_r;
    }

    let mut breakpoints = Vec::new();
    let mut current = n;
    while current > 0 {
        breakpoints.push(current);
        current = cp[current];
    }
    breakpoints.reverse();
    
    if breakpoints.first() == Some(&0) {
        breakpoints.remove(0);
    }
    
    Ok(breakpoints)
}

#[pymodule]
fn rust_extensions(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_change_points_pelt_rs, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_change_points() {
        let mut data = Vec::new();
        // Segment 1: mean 0.0
        for _ in 0..100 {
            data.push(0.0);
        }
        // Segment 2: mean 5.0
        for _ in 0..100 {
            data.push(5.0);
        }
        
        let penalty = 10.0;
        let bps = detect_change_points_pelt_rs(data, penalty, 10, "l2").unwrap();
        // Should find a change point around 100, and end at 200
        assert_eq!(bps.len(), 2);
        assert_eq!(bps[0], 100);
        assert_eq!(bps[1], 200);
    }

    #[test]
    fn test_detect_change_points_normal_model() {
        let mut data = Vec::new();
        // Segment 1: low variance around mean 0
        for i in 0..100 {
            data.push(if i % 2 == 0 { 0.05 } else { -0.05 });
        }
        // Segment 2: high variance around mean 0
        for i in 0..100 {
            data.push(if i % 2 == 0 { 1.0 } else { -1.0 });
        }

        let penalty = 10.0;
        let bps = detect_change_points_pelt_rs(data, penalty, 10, "normal").unwrap();
        assert_eq!(bps.len(), 2);
        assert_eq!(bps[0], 100);
        assert_eq!(bps[1], 200);
    }
}
