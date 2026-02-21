use ndarray::{Array1, ArrayView1};

#[cfg(feature = "simd")]
use std::simd::{f64x4};
#[cfg(feature = "simd")]
use std::simd::num::SimdFloat;

/// SIMD-optimized EMA calculation (requires `simd` feature and nightly Rust)
///
/// Uses 4-way f64 SIMD vectors for faster computation on arrays.
/// Falls back to scalar implementation when SIMD is not available.
#[cfg(feature = "simd")]
pub fn calculate_ema_simd(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let n = prices_arr.len();
    let mut ema = Array1::<f64>::from_elem(n, f64::NAN);

    // Find first valid index
    let mut start_idx = 0;
    while start_idx < n && prices_arr[start_idx].is_nan() {
        start_idx += 1;
    }

    // Need 'length' items for SMA init
    if n < start_idx + length {
        return ema;
    }

    // SMA Initialization using SIMD
    let mut sum = 0.0;
    let init_end = start_idx + length;

    // Process 4 elements at a time with SIMD
    let chunks = length / 4;
    if chunks > 0 {
        let mut vec_sum = f64x4::splat(0.0);
        for i in 0..chunks {
            let base_idx = start_idx + i * 4;
            if base_idx + 4 <= init_end {
                let vec = f64x4::from_array([
                    prices_arr[base_idx],
                    prices_arr[base_idx + 1],
                    prices_arr[base_idx + 2],
                    prices_arr[base_idx + 3],
                ]);
                vec_sum += vec;
            }
        }
        // Sum the vector elements using horizontal add
        sum = vec_sum.reduce_sum();

        // Process remaining elements
        for i in (chunks * 4)..length {
            sum += prices_arr[start_idx + i];
        }
    } else {
        // If length < 4, use scalar
        for i in 0..length {
            sum += prices_arr[start_idx + i];
        }
    }

    ema[init_end - 1] = sum / length as f64;

    let alpha = 2.0 / (length as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;

    // Recursive calculation (harder to SIMD-ize due to dependency)
    for i in init_end..n {
        ema[i] = alpha * prices_arr[i] + one_minus_alpha * ema[i - 1];
    }

    ema
}

/// SIMD-optimized SMA calculation (requires `simd` feature and nightly Rust)
#[cfg(feature = "simd")]
pub fn calculate_sma_simd(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let n = prices_arr.len();
    let mut sma = Array1::<f64>::from_elem(n, f64::NAN);

    if n < length {
        return sma;
    }

    let length_f64 = length as f64;

    for i in (length - 1)..n {
        let mut sum = 0.0;

        // Process 4 elements at a time with SIMD
        let chunks = length / 4;
        if chunks > 0 {
            let mut vec_sum = f64x4::splat(0.0);
            for j in 0..chunks {
                let base_idx = i - (j * 4 + 3);
                if base_idx + 4 <= i + 1 {
                    let vec = f64x4::from_array([
                        prices_arr[base_idx],
                        prices_arr[base_idx + 1],
                        prices_arr[base_idx + 2],
                        prices_arr[base_idx + 3],
                    ]);
                    vec_sum += vec;
                }
            }
            sum = vec_sum.reduce_sum();

            // Process remaining elements
            for j in (chunks * 4)..length {
                sum += prices_arr[i - j];
            }
        } else {
            // If length < 4, use scalar
            for j in 0..length {
                sum += prices_arr[i - j];
            }
        }

        sma[i] = sum / length_f64;
    }

    sma
}

/// SIMD-optimized weighted sum calculation
#[cfg(feature = "simd")]
pub fn calculate_wma_simd(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let n = prices_arr.len();
    let mut wma = Array1::<f64>::from_elem(n, f64::NAN);

    if n < length {
        return wma;
    }

    let denominator = (length * (length + 1)) as f64 / 2.0;

    for i in (length - 1)..n {
        let mut weighted_sum = 0.0;

        // Process 4 elements at a time with SIMD
        let chunks = length / 4;
        if chunks > 0 {
            let mut vec_sum = f64x4::splat(0.0);
            for j in 0..chunks {
                let base_idx = i - (j * 4 + 3);
                let weight_base = length - (j * 4 + 3);

                if base_idx + 4 <= i + 1 {
                    let prices = f64x4::from_array([
                        prices_arr[base_idx],
                        prices_arr[base_idx + 1],
                        prices_arr[base_idx + 2],
                        prices_arr[base_idx + 3],
                    ]);
                    let weights = f64x4::from_array([
                        weight_base as f64,
                        (weight_base + 1) as f64,
                        (weight_base + 2) as f64,
                        (weight_base + 3) as f64,
                    ]);
                    vec_sum += prices * weights;
                }
            }
            weighted_sum = vec_sum.reduce_sum();

            // Process remaining elements
            for j in (chunks * 4)..length {
                let weight = (length - j) as f64;
                weighted_sum += prices_arr[i - j] * weight;
            }
        } else {
            // If length < 4, use scalar
            for j in 0..length {
                let weight = (length - j) as f64;
                weighted_sum += prices_arr[i - j] * weight;
            }
        }

        wma[i] = weighted_sum / denominator;
    }

    wma
}

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_ema_simd_vs_scalar() {
        use crate::ma_calculations::calculate_ema;

        let prices = Array1::from_vec(vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
        ]);
        let length = 5;

        let ema_scalar = calculate_ema(prices.view(), length);
        let ema_simd = calculate_ema_simd(prices.view(), length);

        // Compare results (should be very close, within floating point precision)
        for i in 0..prices.len() {
            if !ema_scalar[i].is_nan() && !ema_simd[i].is_nan() {
                let diff = (ema_scalar[i] - ema_simd[i]).abs();
                assert!(
                    diff < 1e-10,
                    "EMA mismatch at index {}: scalar={}, simd={}",
                    i,
                    ema_scalar[i],
                    ema_simd[i]
                );
            }
        }
    }

    #[test]
    fn test_sma_simd_vs_scalar() {
        use crate::ma_calculations::calculate_sma;

        let prices = Array1::from_vec(vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
        ]);
        let length = 5;

        let sma_scalar = calculate_sma(prices.view(), length);
        let sma_simd = calculate_sma_simd(prices.view(), length);

        for i in 0..prices.len() {
            if !sma_scalar[i].is_nan() && !sma_simd[i].is_nan() {
                let diff = (sma_scalar[i] - sma_simd[i]).abs();
                assert!(
                    diff < 1e-10,
                    "SMA mismatch at index {}: scalar={}, simd={}",
                    i,
                    sma_scalar[i],
                    sma_simd[i]
                );
            }
        }
    }

    #[test]
    fn test_wma_simd_vs_scalar() {
        use crate::ma_calculations::calculate_wma;

        let prices = Array1::from_vec(vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
        ]);
        let length = 5;

        let wma_scalar = calculate_wma(prices.view(), length);
        let wma_simd = calculate_wma_simd(prices.view(), length);

        for i in 0..prices.len() {
            if !wma_scalar[i].is_nan() && !wma_simd[i].is_nan() {
                let diff = (wma_scalar[i] - wma_simd[i]).abs();
                assert!(
                    diff < 1e-10,
                    "WMA mismatch at index {}: scalar={}, simd={}",
                    i,
                    wma_scalar[i],
                    wma_simd[i]
                );
            }
        }
    }
}
