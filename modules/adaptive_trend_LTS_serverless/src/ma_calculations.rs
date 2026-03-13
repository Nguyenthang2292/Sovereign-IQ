use ndarray::{s, Array1, ArrayView1};

/// Calculate EMA using pandas_ta semantics (presma=True, adjust=False, ignore_na=False).
///
/// Parity contract with source:
/// - Seed uses the mean of the first `length` samples (ignoring NaNs).
/// - Values before `length - 1` are NaN.
/// - If current source is NaN, EMA carries previous value.
/// - When finite values resume after `k` NaNs, pandas ewm(ignore_na=False)
///   uses a gap-adjusted alpha:
///   y_t = (decay * y_prev + alpha * x_t) / (decay + alpha)
///   where decay = (1 - alpha)^(k + 1).
/// - If previous EMA is NaN and current source is finite, EMA boots from source.
pub fn calculate_ema(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let n = prices_arr.len();
    let mut ema = Array1::<f64>::from_elem(n, f64::NAN);

    if length == 0 || n < length {
        return ema;
    }

    // Presma seed: mean(first length samples), ignoring NaNs.
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..length {
        let value = prices_arr[i];
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }
    ema[length - 1] = if count > 0 {
        sum / count as f64
    } else {
        f64::NAN
    };

    let alpha = 2.0 / (length as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;
    let mut nan_gap = 0usize;

    for i in length..n {
        let price = prices_arr[i];
        let prev = ema[i - 1];

        if price.is_finite() {
            ema[i] = if prev.is_finite() {
                if nan_gap == 0 {
                    alpha * price + one_minus_alpha * prev
                } else {
                    let decay = one_minus_alpha.powi((nan_gap + 1) as i32);
                    (decay * prev + alpha * price) / (decay + alpha)
                }
            } else {
                price
            };
            nan_gap = 0;
        } else {
            ema[i] = prev;
            nan_gap += 1;
        }
    }

    ema
}

/// Calculate WMA (Weighted Moving Average)
pub fn calculate_wma(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    let n = prices_arr.len();
    let mut wma = Array1::<f64>::from_elem(n, f64::NAN);

    if length == 0 || n < length {
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
    // Source parity: DEMA = 2 * EMA(close) - EMA(EMA(close)),
    // where both EMA passes use the same presma initialization behavior.
    let ema1 = calculate_ema(prices_arr, length);
    let ema2 = calculate_ema(ema1.view(), length);

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

    if length == 0 || n < length {
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

    if length == 0 || n < length {
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

    if length == 0 || n < length {
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

    if length == 0 || n < 1 {
        return kama;
    }

    // Fast/Slow constants
    let fast_end = 0.666; // 2 / (2 + 1)
    let slow_end = 0.064;

    for i in 0..n {
        if i == 0 {
            kama[i] = prices_arr[i];
            continue;
        }

        if i < length {
            kama[i] = kama[i - 1];
            continue;
        }

        let mut noise = 0.0;
        for j in (i - length + 1)..=i {
            noise += (prices_arr[j] - prices_arr[j - 1]).abs();
        }

        let signal = (prices_arr[i] - prices_arr[i - length]).abs();
        let ratio = if noise == 0.0 { 0.0 } else { signal / noise };
        let smooth = (ratio * (fast_end - slow_end) + slow_end).powi(2);

        let mut prev_kama = kama[i - 1];
        if prev_kama.is_nan() {
            prev_kama = prices_arr[i];
        }

        kama[i] = prev_kama + smooth * (prices_arr[i] - prev_kama);
    }

    kama
}

#[cfg(test)]
mod tests {
    use super::{
        calculate_dema, calculate_ema, calculate_hma, calculate_kama, calculate_lsma,
        calculate_sma, calculate_wma,
    };
    use ndarray::{Array1, ArrayView1};

    #[test]
    fn test_ma_length_zero_returns_nan_series() {
        let prices = Array1::from_vec(vec![100.0, 101.0, 102.0, 103.0]);

        let ema = calculate_ema(prices.view(), 0);
        let wma = calculate_wma(prices.view(), 0);
        let dema = calculate_dema(prices.view(), 0);
        let lsma = calculate_lsma(prices.view(), 0);
        let sma = calculate_sma(prices.view(), 0);
        let hma = calculate_hma(prices.view(), 0);
        let kama = calculate_kama(prices.view(), 0);

        for series in [&ema, &wma, &dema, &lsma, &sma, &hma, &kama] {
            assert!(
                series.iter().all(|value| value.is_nan()),
                "Expected all-NaN output for length=0"
            );
        }
    }
    fn naive_wma(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
        let n = prices_arr.len();
        let mut wma = Array1::<f64>::from_elem(n, f64::NAN);
        if length == 0 || n < length {
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
        if length == 0 || n < length {
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
        if length == 0 || n < 1 {
            return kama;
        }

        let fast_end = 0.666;
        let slow_end = 0.064;

        for i in 0..n {
            if i == 0 {
                kama[i] = prices_arr[i];
                continue;
            }
            if i < length {
                kama[i] = kama[i - 1];
                continue;
            }

            let mut noise = 0.0;
            for j in (i - length + 1)..=i {
                noise += (prices_arr[j] - prices_arr[j - 1]).abs();
            }

            let signal = (prices_arr[i] - prices_arr[i - length]).abs();
            let ratio = if noise == 0.0 { 0.0 } else { signal / noise };
            let sc = (ratio * (fast_end - slow_end) + slow_end).powi(2);

            let mut prev_kama = kama[i - 1];
            if prev_kama.is_nan() {
                prev_kama = prices_arr[i];
            }
            kama[i] = prev_kama + sc * (prices_arr[i] - prev_kama);
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
