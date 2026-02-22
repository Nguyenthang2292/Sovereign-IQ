use ndarray::{Array1, ArrayView1};

#[cfg(feature = "simd")]
use std::simd::f64x4;
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

    let mut window_sum = 0.0;
    let chunks = length / 4;
    if chunks > 0 {
        let mut vec_sum = f64x4::splat(0.0);
        for j in 0..chunks {
            let base_idx = j * 4;
            let vec = f64x4::from_array([
                prices_arr[base_idx],
                prices_arr[base_idx + 1],
                prices_arr[base_idx + 2],
                prices_arr[base_idx + 3],
            ]);
            vec_sum += vec;
        }
        window_sum = vec_sum.reduce_sum();

        for j in (chunks * 4)..length {
            window_sum += prices_arr[j];
        }
    } else {
        for j in 0..length {
            window_sum += prices_arr[j];
        }
    }

    sma[length - 1] = window_sum / length_f64;

    for i in length..n {
        window_sum += prices_arr[i] - prices_arr[i - length];
        sma[i] = window_sum / length_f64;
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

/// SIMD-optimized EMA calculation for internal use in DEMA
#[cfg(feature = "simd")]
fn calculate_ema_simple_simd(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let n = prices_arr.len();
    let mut ema = Array1::<f64>::from_elem(n, f64::NAN);

    let mut start_idx = 0;
    while start_idx < n && prices_arr[start_idx].is_nan() {
        start_idx += 1;
    }

    if start_idx >= n {
        return ema;
    }

    ema[start_idx] = prices_arr[start_idx];

    let alpha = 2.0 / (length as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;

    for i in (start_idx + 1)..n {
        ema[i] = alpha * prices_arr[i] + one_minus_alpha * ema[i - 1];
    }
    ema
}

/// SIMD-optimized DEMA (Double Exponential Moving Average)
///
/// DEMA = 2 * EMA1 - EMA2
/// where:
///   EMA1 = EMA(prices, length) with SMA initialization
///   EMA2 = EMA(EMA1, length) with simple initialization
#[cfg(feature = "simd")]
pub fn calculate_dema_simd(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let ema1 = calculate_ema_simd(prices_arr, length);
    let ema2 = calculate_ema_simple_simd(ema1.view(), length);
    2.0 * &ema1 - &ema2
}

/// SIMD-optimized HMA (Hull Moving Average)
///
/// HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
/// where:
///   WMA(n/2) = WMA with half length
///   WMA(n) = WMA with full length
///   Final WMA uses sqrt(n) as length
#[cfg(feature = "simd")]
pub fn calculate_hma_simd(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let n = prices_arr.len();

    let half_len = std::cmp::max(length / 2, 1);
    let sqrt_len = std::cmp::max((length as f64).sqrt() as usize, 1);

    if n < length {
        return Array1::<f64>::from_elem(n, f64::NAN);
    }

    let wma_half = calculate_wma_simd(prices_arr, half_len);
    let wma_full = calculate_wma_simd(prices_arr, length);

    let raw = 2.0 * &wma_half - &wma_full;

    calculate_wma_simd(raw.view(), sqrt_len)
}

/// SIMD-optimized LSMA (Least Squares Moving Average)
///
/// Uses SIMD for the inner loop sum calculations while maintaining
/// the same algorithmic approach as the scalar version.
#[cfg(feature = "simd")]
pub fn calculate_lsma_simd(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    use crate::constants::MIN_LENGTH_MEDIUM;
    let n = prices_arr.len();
    let mut lsma = Array1::<f64>::from_elem(n, f64::NAN);

    if n < length.max(MIN_LENGTH_MEDIUM) {
        return lsma;
    }

    let length_f64 = length as f64;

    // Pre-calculate regression constants
    let sum_x = length_f64 * (length_f64 - 1.0) / 2.0;
    let sum_x2 = length_f64 * (length_f64 - 1.0) * (2.0 * length_f64 - 1.0) / 6.0;
    let divisor = length_f64 * sum_x2 - sum_x.powi(2);

    for i in (length - 1)..n {
        let start_idx = (i + 1) - length;

        // Use SIMD for sum calculations
        let chunks = length / 4;
        let mut vec_sum_y = f64x4::splat(0.0);
        let mut vec_sum_xy = f64x4::splat(0.0);

        for j in 0..chunks {
            let base_idx = start_idx + j * 4;
            if base_idx + 4 <= i + 1 {
                let prices = f64x4::from_array([
                    prices_arr[base_idx],
                    prices_arr[base_idx + 1],
                    prices_arr[base_idx + 2],
                    prices_arr[base_idx + 3],
                ]);
                let xs = f64x4::from_array([
                    (j * 4) as f64,
                    (j * 4 + 1) as f64,
                    (j * 4 + 2) as f64,
                    (j * 4 + 3) as f64,
                ]);
                vec_sum_y += prices;
                vec_sum_xy += prices * xs;
            }
        }

        let mut sum_y = vec_sum_y.reduce_sum();
        let mut sum_xy = vec_sum_xy.reduce_sum();

        // Process remaining elements
        for j in (chunks * 4)..length {
            let y = prices_arr[start_idx + j];
            if y.is_nan() {
                sum_y = f64::NAN;
                break;
            }
            let x = j as f64;
            sum_y += y;
            sum_xy += x * y;
        }

        if sum_y.is_nan() {
            continue;
        }

        let m = (length_f64 * sum_xy - sum_x * sum_y) / divisor;
        let c = (sum_y - m * sum_x) / length_f64;
        lsma[i] = m * (length_f64 - 1.0) + c;
    }

    lsma
}

/// SIMD-optimized KAMA (Kaufman Adaptive Moving Average)
///
/// KAMA is inherently sequential due to its recursive nature,
/// but we can use SIMD for the volatility calculation within each iteration.
#[cfg(feature = "simd")]
pub fn calculate_kama_simd(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let n = prices_arr.len();
    let mut kama = Array1::<f64>::from_elem(n, f64::NAN);

    if n < length {
        return kama;
    }

    let fast_end = 0.666;
    let slow_end = 0.0645;

    let start_idx = length;
    kama[start_idx - 1] = prices_arr[start_idx - 1];

    for i in start_idx..n {
        let price = prices_arr[i];
        let prev_kama = kama[i - 1];

        // Efficiency Ratio calculation
        let change = (price - prices_arr[i - length]).abs();

        // Use SIMD for volatility sum
        let chunks = length / 4;
        let mut vec_volatility = f64x4::splat(0.0);

        for j in 0..chunks {
            let base_idx = i - j * 4;
            if base_idx >= 3 && base_idx < n {
                let curr = f64x4::from_array([
                    prices_arr[base_idx],
                    prices_arr[base_idx - 1],
                    prices_arr[base_idx - 2],
                    prices_arr[base_idx - 3],
                ]);
                let prev = f64x4::from_array([
                    prices_arr[base_idx - 1],
                    prices_arr[base_idx - 2],
                    prices_arr[base_idx - 3],
                    // SAFETY: start_idx = length and length >= MIN_LENGTH_NARROW (5), so for all
                    // valid positions i >= length, base_idx = i - j*4 >= length - (chunks-1)*4 >= 4.
                    // The `continue` branch is unreachable for validated inputs but kept for safety.
                    if base_idx >= 4 {
                        prices_arr[base_idx - 4]
                    } else {
                        continue;
                    },
                ]);
                vec_volatility += (curr - prev).abs();
            }
        }

        let mut volatility = vec_volatility.reduce_sum();

        // Process remaining elements
        for j in (chunks * 4)..length {
            if i >= j + 1 {
                volatility += (prices_arr[i - j] - prices_arr[i - j - 1]).abs();
            }
        }

        let er = if volatility != 0.0 {
            change / volatility
        } else {
            0.0
        };
        let sc = (er * (fast_end - slow_end) + slow_end).powi(2);

        kama[i] = prev_kama + sc * (price - prev_kama);
    }

    kama
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

    #[test]
    fn test_dema_simd_vs_scalar() {
        use crate::ma_calculations::calculate_dema;

        let prices = Array1::from_vec(vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0,
        ]);
        let length = 5;

        let demas_scalar = calculate_dema(prices.view(), length);
        let demas_simd = calculate_dema_simd(prices.view(), length);

        for i in 0..prices.len() {
            if !demas_scalar[i].is_nan() && !demas_simd[i].is_nan() {
                let diff = (demas_scalar[i] - demas_simd[i]).abs();
                assert!(
                    diff < 1e-10,
                    "DEMA mismatch at index {}: scalar={}, simd={}",
                    i,
                    demas_scalar[i],
                    demas_simd[i]
                );
            }
        }
    }

    #[test]
    fn test_hma_simd_vs_scalar() {
        use crate::ma_calculations::calculate_hma;

        let prices = Array1::from_vec(vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
        ]);
        let length = 9;

        let hma_scalar = calculate_hma(prices.view(), length);
        let hma_simd = calculate_hma_simd(prices.view(), length);

        for i in 0..prices.len() {
            if !hma_scalar[i].is_nan() && !hma_simd[i].is_nan() {
                let diff = (hma_scalar[i] - hma_simd[i]).abs();
                assert!(
                    diff < 1e-10,
                    "HMA mismatch at index {}: scalar={}, simd={}",
                    i,
                    hma_scalar[i],
                    hma_simd[i]
                );
            }
        }
    }

    #[test]
    fn test_lsma_simd_vs_scalar() {
        use crate::ma_calculations::calculate_lsma;

        let prices = Array1::from_vec(vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
        ]);
        let length = 10;

        let lsma_scalar = calculate_lsma(prices.view(), length);
        let lsma_simd = calculate_lsma_simd(prices.view(), length);

        for i in 0..prices.len() {
            if !lsma_scalar[i].is_nan() && !lsma_simd[i].is_nan() {
                let diff = (lsma_scalar[i] - lsma_simd[i]).abs();
                assert!(
                    diff < 1e-10,
                    "LSMA mismatch at index {}: scalar={}, simd={}",
                    i,
                    lsma_scalar[i],
                    lsma_simd[i]
                );
            }
        }
    }

    #[test]
    fn test_kama_simd_vs_scalar() {
        use crate::ma_calculations::calculate_kama;

        let prices = Array1::from_vec(vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
        ]);
        let length = 10;

        let kama_scalar = calculate_kama(prices.view(), length);
        let kama_simd = calculate_kama_simd(prices.view(), length);

        for i in 0..prices.len() {
            if !kama_scalar[i].is_nan() && !kama_simd[i].is_nan() {
                let diff = (kama_scalar[i] - kama_simd[i]).abs();
                assert!(
                    diff < 1e-10,
                    "KAMA mismatch at index {}: scalar={}, simd={}",
                    i,
                    kama_scalar[i],
                    kama_simd[i]
                );
            }
        }
    }
}
