use ndarray::{Array1, ArrayView1};
use rayon::prelude::*;

/// Calculate percentage Rate of Change (ROC) multiplied by exponential growth factor.
///
/// Formula: ((price[i] - price[i-1]) / price[i-1]) * exp(La * i)
///
/// # Arguments
///
/// * `prices_arr` - Array view of price values
/// * `la` - Lambda parameter (growth rate)
///
/// # Returns
///
/// Array1<f64> containing calculated ROC values with growth factor
pub fn calculate_roc_with_growth(prices_arr: ArrayView1<f64>, la: f64) -> Array1<f64> {
    let n = prices_arr.len();
    let mut roc = Array1::<f64>::zeros(n);

    // Parallel processing for large arrays
    const PARALLEL_THRESHOLD: usize = 5000;
    
    if n > PARALLEL_THRESHOLD {
        roc.as_slice_mut().unwrap().par_iter_mut().enumerate().for_each(|(i, val)| {
            if i == 0 {
                *val = f64::NAN;
            } else {
                let prev = prices_arr[i - 1];
                if prev == 0.0 {
                    *val = 0.0;
                } else {
                    let r = (prices_arr[i] - prev) / prev;
                    let growth = (la * i as f64).exp();
                    *val = r * growth;
                }
            }
        });
    } else {
        roc[0] = f64::NAN;
        for i in 1..n {
            let prev = prices_arr[i - 1];
            if prev == 0.0 {
                roc[i] = 0.0;
            } else {
                let r = (prices_arr[i] - prev) / prev;
                let growth = (la * i as f64).exp();
                roc[i] = r * growth;
            }
        }
    }
    
    roc
}

/// Calculate the 9 robustness lengths based on base length and robustness setting.
pub fn get_diflen(length: usize, robustness: &str) -> Vec<usize> {
    let (l1, l2, l3, l4, l_1, l_2, l_3, l_4) = match robustness {
        "Narrow" => (
            length + 1,
            length + 2,
            length + 3,
            length + 4,
            length.saturating_sub(1),
            length.saturating_sub(2),
            length.saturating_sub(3),
            length.saturating_sub(4),
        ),
        "Wide" => (
            length + 1,
            length + 3,
            length + 5,
            length + 7,
            length.saturating_sub(1),
            length.saturating_sub(3),
            length.saturating_sub(5),
            length.saturating_sub(7),
        ),
        _ => ( // Medium or default
            length + 1,
            length + 2,
            length + 4,
            length + 6,
            length.saturating_sub(1),
            length.saturating_sub(2),
            length.saturating_sub(4),
            length.saturating_sub(6),
        ),
    };
    vec![
        length,
        l1,
        l2,
        l3,
        l4,
        l_1.max(1),
        l_2.max(1),
        l_3.max(1),
        l_4.max(1),
    ]
}
