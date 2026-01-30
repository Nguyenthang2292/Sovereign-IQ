//! Utility functions for XGBoost Rust extensions

use ndarray::ArrayView1;

/// Calculate median of a slice (assumes sorted)
pub fn median(sorted_slice: &[f64]) -> f64 {
    let len = sorted_slice.len();
    if len == 0 {
        return 0.0;
    }

    if len % 2 == 0 {
        (sorted_slice[len / 2 - 1] + sorted_slice[len / 2]) / 2.0
    } else {
        sorted_slice[len / 2]
    }
}

/// Calculate standard deviation
pub fn std_dev(arr: ArrayView1<f64>) -> f64 {
    let mean = arr.mean().unwrap_or(0.0);
    let variance: f64 = arr.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / arr.len() as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median() {
        assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median(&[]), 0.0);
    }
}
