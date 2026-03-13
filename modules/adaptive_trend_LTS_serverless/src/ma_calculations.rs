use ndarray::{s, Array1, ArrayView1};

/// Calculate EMA with value-initialization (first valid value seed).
///
/// Used internally for DEMA pass-2 to preserve output availability after pass-1 EMA.
#[cfg(not(feature = "simd"))]
fn calculate_ema_value_init(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let n = prices_arr.len();
    let mut ema = Array1::<f64>::from_elem(n, f64::NAN);

    // Find first valid index
    let mut start_idx = 0;
    while start_idx < n && prices_arr[start_idx].is_nan() {
        start_idx += 1;
    }

    if start_idx >= n {
        return ema;
    }

    let alpha = 2.0 / (length as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;

    // Simple Initialization: Start with first valid value
    ema[start_idx] = prices_arr[start_idx];

    // Recursive calculation
    for i in (start_idx + 1)..n {
        ema[i] = alpha * prices_arr[i] + one_minus_alpha * ema[i - 1];
    }
    ema
}

/// Calculate EMA with Standard Initialization (SMA of first N valid values)
pub fn calculate_ema(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
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

    // SMA Initialization
    let mut sum = 0.0;
    for i in 0..length {
        sum += prices_arr[start_idx + i];
    }
    ema[start_idx + length - 1] = sum / length as f64;

    let alpha = 2.0 / (length as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;

    // Recursive calculation
    for i in (start_idx + length)..n {
        ema[i] = alpha * prices_arr[i] + one_minus_alpha * ema[i - 1];
    }

    ema
}

/// Calculate WMA (Weighted Moving Average)
pub fn calculate_wma(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let n = prices_arr.len();
    let mut wma = Array1::<f64>::from_elem(n, f64::NAN);

    if n < length {
        return wma;
    }

    // Pre-calculate denominator for efficiency
    let denominator = (length * (length + 1)) as f64 / 2.0;
    let length_f64 = length as f64;

    // Initial window (oldest->newest weights 1..length)
    let mut window_sum = 0.0;
    let mut weighted_sum = 0.0;
    let mut nan_count = 0usize;
    for idx in 0..length {
        let price = prices_arr[idx];
        if price.is_nan() {
            nan_count += 1;
        } else {
            window_sum += price;
            weighted_sum += (idx as f64 + 1.0) * price;
        }
    }

    if nan_count == 0 {
        wma[length - 1] = weighted_sum / denominator;
    }

    // Sliding window update:
    // weighted_sum[i] = weighted_sum[i-1] - window_sum[i-1] + length * price[i]
    for i in length..n {
        let previous_nan_count = nan_count;
        let previous_window_sum = window_sum;
        let exiting = prices_arr[i - length];
        let entering = prices_arr[i];

        if exiting.is_nan() {
            nan_count = nan_count.saturating_sub(1);
        } else {
            window_sum -= exiting;
        }

        if entering.is_nan() {
            nan_count += 1;
        } else {
            window_sum += entering;
        }

        if nan_count == 0 {
            if previous_nan_count == 0 {
                weighted_sum = weighted_sum - previous_window_sum + length_f64 * entering;
            } else {
                weighted_sum = 0.0;
                for offset in 0..length {
                    let value = prices_arr[(i + 1) - length + offset];
                    weighted_sum += (offset as f64 + 1.0) * value;
                }
            }
            wma[i] = weighted_sum / denominator;
        } else {
            wma[i] = f64::NAN;
        }
    }

    wma
}

/// Calculate DEMA (Double Exponential Moving Average)
#[cfg(not(feature = "simd"))]
pub fn calculate_dema(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    // Pass 1: Standard EMA (SMA Init)
    let ema1 = calculate_ema(prices_arr, length);

    // Pass 2: EMA with value-init seed to preserve availability.
    let ema2 = calculate_ema_value_init(ema1.view(), length);

    // DEMA = 2 * EMA1 - EMA2
    2.0 * &ema1 - &ema2
}

/// Calculate DEMA (Double Exponential Moving Average)
#[cfg(feature = "simd")]
pub fn calculate_dema(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    crate::ma_simd::calculate_dema_simd(prices_arr, length)
}

/// Calculate LSMA (Least Squares Moving Average)
pub fn calculate_lsma(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let n = prices_arr.len();
    let mut lsma = Array1::<f64>::from_elem(n, f64::NAN);

    if n < length {
        return lsma;
    }

    let length_f64 = length as f64;

    // Sum of x (0 to length-1) = n*(n-1)/2
    let sum_x = length_f64 * (length_f64 - 1.0) / 2.0;
    // Sum of x squared = n*(n-1)*(2n-1)/6
    let sum_x2 = length_f64 * (length_f64 - 1.0) * (2.0 * length_f64 - 1.0) / 6.0;
    // Divisor for slope: n*sum_x2 - sum_x^2
    let divisor = length_f64 * sum_x2 - sum_x.powi(2);

    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut nan_count = 0usize;

    // Initial window
    for j in 0..length {
        let y = prices_arr[j];
        if y.is_nan() {
            nan_count += 1;
        } else {
            let x = j as f64;
            sum_y += y;
            sum_xy += x * y;
        }
    }

    if nan_count == 0 {
        // Linear Regression: y = mx + c
        // m (slope) = (n*sum_xy - sum_x*sum_y) / divisor
        let m = (length_f64 * sum_xy - sum_x * sum_y) / divisor;
        // c (intercept) = (sum_y - m*sum_x) / n
        let c = (sum_y - m * sum_x) / length_f64;
        lsma[length - 1] = m * (length_f64 - 1.0) + c;
    }

    // Sliding window update:
    // sum_y[i]  = sum_y[i-1]  - old + new
    // sum_xy[i] = sum_xy[i-1] - (sum_y[i-1] - old) + (length - 1) * new
    for i in length..n {
        let previous_nan_count = nan_count;
        let previous_sum_y = sum_y;
        let exiting = prices_arr[i - length];
        let entering = prices_arr[i];

        if exiting.is_nan() {
            nan_count = nan_count.saturating_sub(1);
        } else {
            sum_y -= exiting;
        }

        if entering.is_nan() {
            nan_count += 1;
        } else {
            sum_y += entering;
        }

        if nan_count == 0 {
            if previous_nan_count == 0 {
                sum_xy = sum_xy - (previous_sum_y - exiting) + (length_f64 - 1.0) * entering;
            } else {
                sum_y = 0.0;
                sum_xy = 0.0;
                for offset in 0..length {
                    let y = prices_arr[(i + 1) - length + offset];
                    let x = offset as f64;
                    sum_y += y;
                    sum_xy += x * y;
                }
            }

            let m = (length_f64 * sum_xy - sum_x * sum_y) / divisor;
            let c = (sum_y - m * sum_x) / length_f64;
            lsma[i] = m * (length_f64 - 1.0) + c;
        } else {
            lsma[i] = f64::NAN;
        }
    }

    lsma
}

/// Calculate SMA (Simple Moving Average)
/// Uses sliding window algorithm for O(n) complexity instead of O(n*length)
pub fn calculate_sma(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let n = prices_arr.len();
    let mut sma = Array1::<f64>::from_elem(n, f64::NAN);

    if n < length {
        return sma;
    }

    let length_f64 = length as f64;

    // Calculate first window sum
    let mut window_sum: f64 = prices_arr.slice(s![0..length]).sum();
    sma[length - 1] = window_sum / length_f64;

    // Sliding window: O(n) instead of O(n*length)
    for i in length..n {
        window_sum += prices_arr[i] - prices_arr[i - length];
        sma[i] = window_sum / length_f64;
    }

    sma
}

/// Calculate HMA (Hull Moving Average)
pub fn calculate_hma(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let n = prices_arr.len();

    let half_len = std::cmp::max(length / 2, 1);
    let sqrt_len = std::cmp::max((length as f64).sqrt() as usize, 1);

    if n < length {
        return Array1::<f64>::from_elem(n, f64::NAN);
    }

    // Step 1: WMA(n/2)
    let wma_half = calculate_wma(prices_arr, half_len);

    // Step 2: WMA(n)
    let wma_full = calculate_wma(prices_arr, length);

    // Step 3: raw = 2 * WMA(n/2) - WMA(n)
    let raw = 2.0 * &wma_half - &wma_full;

    // Step 4: HMA = WMA(raw, sqrt(n))
    calculate_wma(raw.view(), sqrt_len)
}

/// Calculate KAMA (Kaufman Adaptive Moving Average)
///
/// KAMA adjusts its smoothing constant based on market efficiency.
/// It uses the Efficiency Ratio (ER) to determine how much to adapt:
/// - ER close to 1 (strong trend) = fast smoothing
/// - ER close to 0 (choppy market) = slow smoothing
///
/// # Arguments
/// * `prices_arr` - Price data array
/// * `length` - Period for ER calculation
///
/// # Returns
/// `Array1<f64>` containing KAMA values
pub fn calculate_kama(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let n = prices_arr.len();
    let mut kama = Array1::<f64>::from_elem(n, f64::NAN);

    if n < length {
        return kama;
    }

    // Fast/Slow constants
    let fast_end = 0.666; // 2 / (2 + 1)
    let slow_end = 0.0645; // 2 / (30 + 1)

    // Initialize
    let start_idx = length;
    kama[start_idx - 1] = prices_arr[start_idx - 1]; // Simple init

    // Initial volatility window for i = start_idx:
    // sum_{k=1..length} abs(price[k] - price[k-1])
    let mut volatility_window = 0.0;
    if start_idx < n {
        for k in 1..=length {
            volatility_window += (prices_arr[k] - prices_arr[k - 1]).abs();
        }
    }
    let mut recompute_volatility_window = false;

    for i in start_idx..n {
        let price = prices_arr[i];

        // Defense-in-depth: validation normally blocks NaNs before this function is called.
        // Keep an explicit guard so direct unit tests or internal callers fail safe.
        if price.is_nan() || prices_arr[i - 1].is_nan() {
            kama[i] = f64::NAN;
            recompute_volatility_window = true;
            continue;
        }

        let mut prev_kama = kama[i - 1];
        if !prev_kama.is_finite() {
            // If the previous KAMA is invalid due to a prior NaN segment, reseed from
            // the previous price so the series can recover once the volatility window
            // becomes valid again.
            prev_kama = prices_arr[i - 1];
            if !prev_kama.is_finite() {
                kama[i] = f64::NAN;
                recompute_volatility_window = true;
                continue;
            }
        }

        if i > start_idx && !recompute_volatility_window {
            let add_delta = price - prices_arr[i - 1];
            let remove_delta = prices_arr[i - length] - prices_arr[i - length - 1];
            if add_delta.is_finite() && remove_delta.is_finite() && volatility_window.is_finite() {
                volatility_window += add_delta.abs() - remove_delta.abs();
            } else {
                recompute_volatility_window = true;
            }
        }

        if i == start_idx || recompute_volatility_window {
            let mut recomputed_volatility = 0.0;
            let mut valid_window = true;
            for j in 0..length {
                let left = prices_arr[i - j];
                let right = prices_arr[i - j - 1];
                let delta = left - right;
                if !delta.is_finite() {
                    valid_window = false;
                    break;
                }
                recomputed_volatility += delta.abs();
            }

            if !valid_window {
                kama[i] = f64::NAN;
                recompute_volatility_window = true;
                continue;
            }

            volatility_window = recomputed_volatility;
            recompute_volatility_window = false;
        }

        // Efficiency Ratio
        let change = (price - prices_arr[i - length]).abs();
        let er = if volatility_window != 0.0 {
            change / volatility_window
        } else {
            0.0
        };
        let sc = (er * (fast_end - slow_end) + slow_end).powi(2);

        kama[i] = prev_kama + sc * (price - prev_kama);
    }

    kama
}

#[cfg(test)]
mod tests {
    use super::{calculate_kama, calculate_lsma, calculate_wma};
    use ndarray::{Array1, ArrayView1};

    fn naive_wma(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
        let n = prices_arr.len();
        let mut wma = Array1::<f64>::from_elem(n, f64::NAN);
        if n < length {
            return wma;
        }
        let denominator = (length * (length + 1)) as f64 / 2.0;
        for i in (length - 1)..n {
            let mut weighted_sum = 0.0;
            for j in 0..length {
                let weight = (length - j) as f64;
                weighted_sum += prices_arr[i - j] * weight;
            }
            wma[i] = weighted_sum / denominator;
        }
        wma
    }

    fn naive_lsma(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
        let n = prices_arr.len();
        let mut lsma = Array1::<f64>::from_elem(n, f64::NAN);
        if n < length {
            return lsma;
        }

        let length_f64 = length as f64;
        let sum_x = length_f64 * (length_f64 - 1.0) / 2.0;
        let sum_x2 = length_f64 * (length_f64 - 1.0) * (2.0 * length_f64 - 1.0) / 6.0;
        let divisor = length_f64 * sum_x2 - sum_x.powi(2);

        for i in (length - 1)..n {
            let start_idx = (i + 1) - length;
            let mut sum_y = 0.0;
            let mut sum_xy = 0.0;

            for j in 0..length {
                let y = prices_arr[start_idx + j];
                if y.is_nan() {
                    sum_y = f64::NAN;
                    break;
                }
                sum_y += y;
                sum_xy += j as f64 * y;
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

    fn naive_kama(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
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

            let change = (price - prices_arr[i - length]).abs();
            let mut volatility = 0.0;
            for j in 0..length {
                volatility += (prices_arr[i - j] - prices_arr[i - j - 1]).abs();
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

    fn assert_series_close(a: &Array1<f64>, b: &Array1<f64>, tolerance: f64) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            match (a[i].is_nan(), b[i].is_nan()) {
                (true, true) => {}
                (false, false) => {
                    assert!(
                        (a[i] - b[i]).abs() <= tolerance,
                        "Mismatch at index {}: {} vs {}",
                        i,
                        a[i],
                        b[i]
                    );
                }
                _ => panic!("NaN mismatch at index {}: {} vs {}", i, a[i], b[i]),
            }
        }
    }

    #[test]
    fn test_wma_optimized_matches_naive() {
        let prices = Array1::from_iter((0..128).map(|idx| 100.0 + (idx as f64 * 0.17).sin() * 3.0));
        let optimized = calculate_wma(prices.view(), 28);
        let baseline = naive_wma(prices.view(), 28);
        assert_series_close(&optimized, &baseline, 1e-12);
    }

    #[test]
    fn test_lsma_optimized_matches_naive() {
        let prices = Array1::from_iter((0..128).map(|idx| 100.0 + (idx as f64 * 0.11).cos() * 2.5));
        let optimized = calculate_lsma(prices.view(), 20);
        let baseline = naive_lsma(prices.view(), 20);
        assert_series_close(&optimized, &baseline, 1e-12);
    }

    #[test]
    fn test_kama_optimized_matches_naive() {
        let prices = Array1::from_iter((0..256).map(|idx| 100.0 + (idx as f64 * 0.07).sin() * 4.0));
        let optimized = calculate_kama(prices.view(), 28);
        let baseline = naive_kama(prices.view(), 28);
        assert_series_close(&optimized, &baseline, 1e-12);
    }

    #[test]
    fn test_kama_recovers_after_nan_window_expires() {
        let mut prices =
            Array1::from_iter((0..96).map(|idx| 100.0 + (idx as f64 * 0.13).sin() * 2.0));
        prices[40] = f64::NAN;

        let kama = calculate_kama(prices.view(), 10);

        assert!(kama[40].is_nan());
        assert!(kama[41].is_nan());
        assert!(
            kama[95].is_finite(),
            "KAMA should recover once the NaN value leaves the volatility window"
        );
    }
}
