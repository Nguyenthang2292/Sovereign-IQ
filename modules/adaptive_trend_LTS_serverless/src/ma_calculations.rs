use ndarray::{s, Array1, ArrayView1};

/// Calculate EMA (Exponential Moving Average) internally with SIMD optimizations.
#[cfg(not(feature = "simd"))]
fn calculate_ema_simple(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
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

    // Sequential processing - inner-loop Rayon was removed as it caused overhead
    // for typical MA lengths (20-50 elements)
    for i in (length - 1)..n {
        let weighted_sum: f64 = (0..length)
            .map(|j| {
                let weight = (length - j) as f64;
                prices_arr[i - j] * weight
            })
            .sum();
        wma[i] = weighted_sum / denominator;
    }

    wma
}

/// Calculate DEMA (Double Exponential Moving Average)
#[cfg(not(feature = "simd"))]
pub fn calculate_dema(prices_arr: ArrayView1<f64>, length: usize) -> Array1<f64> {
    // Pass 1: Standard EMA (SMA Init)
    let ema1 = calculate_ema(prices_arr, length);

    // Pass 2: Simple EMA (Value Init) to preserve availability
    let ema2 = calculate_ema_simple(ema1.view(), length);

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
            let x = j as f64;
            sum_y += y;
            sum_xy += x * y;
        }

        if sum_y.is_nan() {
            continue;
        }

        // Linear Regression: y = mx + c
        // m (slope) = (n*sum_xy - sum_x*sum_y) / divisor
        let m = (length_f64 * sum_xy - sum_x * sum_y) / divisor;
        // c (intercept) = (sum_y - m*sum_x) / n
        let c = (sum_y - m * sum_x) / length_f64;

        // LSMA value is the value of the regression line at the LAST point (x = length - 1)
        lsma[i] = m * (length_f64 - 1.0) + c;
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

    for i in start_idx..n {
        let price = prices_arr[i];
        let prev_kama = kama[i - 1];

        // Efficiency Ratio
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
