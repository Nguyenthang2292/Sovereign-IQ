use crate::equity::*;
use crate::ma_calculations::*;
use crate::ATCConfig;
use ndarray::{Array1, ArrayView1};

/// Robustness level for diflen calculation
///
/// Determines the range of length variations around the base length.
/// - Narrow: ±1, ±2, ±3, ±4 from base
/// - Medium: ±1, ±2, ±4, ±6 from base (default)
/// - Wide: ±1, ±3, ±5, ±7 from base
#[derive(Debug, Clone, Copy)]
pub enum Robustness {
    /// Narrow range: ±1, ±2, ±3, ±4 from base length
    Narrow,
    /// Medium range: ±1, ±2, ±4, ±6 from base length (default)
    Medium,
    /// Wide range: ±1, ±3, ±5, ±7 from base length
    Wide,
}

impl Robustness {
    /// Parse robustness level from string ("narrow", "medium", "wide")
    ///
    /// Returns Medium for any unrecognized input.
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "narrow" => Robustness::Narrow,
            "wide" => Robustness::Wide,
            _ => Robustness::Medium,
        }
    }
}

/// Calculate diflen (differential lengths) for a given base length and robustness
/// Returns 8 length values: (L1, L2, L3, L4, L_1, L_2, L_3, L_4)
pub fn calculate_diflen(length: usize, robustness: Robustness) -> Option<[usize; 8]> {
    if length == 0 {
        return None;
    }

    let min_required = match robustness {
        Robustness::Narrow => 5,
        Robustness::Medium => 7,
        Robustness::Wide => 8,
    };

    if length < min_required {
        eprintln!(
            "[WARN] Base length {} is too small for robustness {:?}. Minimum required: {}",
            length, robustness, min_required
        );
        return None;
    }

    let (l1, l_1, l2, l_2, l3, l_3, l4, l_4) = match robustness {
        Robustness::Narrow => (
            length + 1,
            length - 1,
            length + 2,
            length - 2,
            length + 3,
            length - 3,
            length + 4,
            length - 4,
        ),
        Robustness::Medium => (
            length + 1,
            length - 1,
            length + 2,
            length - 2,
            length + 4,
            length - 4,
            length + 6,
            length - 6,
        ),
        Robustness::Wide => (
            length + 1,
            length - 1,
            length + 3,
            length - 3,
            length + 5,
            length - 5,
            length + 7,
            length - 7,
        ),
    };

    let lengths = [l1, l2, l3, l4, l_1, l_2, l_3, l_4];
    if lengths.iter().any(|&l| l == 0) {
        eprintln!("[ERROR] Calculated length offsets contain zero values");
        return None;
    }

    Some(lengths)
}

fn calculate_ma_variation(prices: ArrayView1<f64>, ma_type: &str, length: usize) -> Array1<f64> {
    #[cfg(feature = "simd")]
    {
        match ma_type {
            "EMA" => return crate::ma_simd::calculate_ema_simd(prices, length),
            "WMA" => return crate::ma_simd::calculate_wma_simd(prices, length),
            "SMA" => return crate::ma_simd::calculate_sma_simd(prices, length),
            _ => {}
        }
    }

    match ma_type {
        "EMA" => calculate_ema(prices, length),
        "HMA" => calculate_hma(prices, length),
        "WMA" => calculate_wma(prices, length),
        "DEMA" => calculate_dema(prices, length),
        "LSMA" => calculate_lsma(prices, length),
        "KAMA" => calculate_kama(prices, length),
        _ => calculate_ema(prices, length),
    }
}

/// Calculate Layer 1 signal with full diflen variations (8 MA calculations)
pub fn calculate_layer1_signal(
    prices: ArrayView1<f64>,
    ma_type: &str,
    base_length: usize,
    lambda_scaled: f64,
    decay_scaled: f64,
    cutout: usize,
    equity_floor: f64,
    robustness: Robustness,
) -> (Array1<f64>, f64) {
    let n = prices.len();

    let diflen_result = match calculate_diflen(base_length, robustness) {
        Some(lengths) => lengths,
        None => {
            eprintln!(
                "[WARN] diflen failed for length {}, using base length only",
                base_length
            );
            return calculate_layer1_signal_single(
                prices,
                ma_type,
                base_length,
                lambda_scaled,
                decay_scaled,
                cutout,
                equity_floor,
            );
        }
    };

    let mut roc = Array1::<f64>::from_elem(n, f64::NAN);
    for i in 1..n {
        if prices[i - 1] != 0.0 && !prices[i - 1].is_nan() {
            roc[i] = (prices[i] - prices[i - 1]) / prices[i - 1];
        }
    }

    let growth = exp_growth(lambda_scaled, n, cutout);
    let r_adjusted = &roc * &growth;

    let mut all_signals: Vec<Array1<f64>> = Vec::with_capacity(8);
    let mut all_equities: Vec<f64> = Vec::with_capacity(8);

    for &length in &diflen_result {
        let ma = calculate_ma_variation(prices, ma_type, length);

        let mut signal = Array1::<f64>::from_elem(n, 0.0);
        for i in 0..n {
            if !prices[i].is_nan() && !ma[i].is_nan() {
                if prices[i] > ma[i] {
                    signal[i] = 1.0;
                } else if prices[i] < ma[i] {
                    signal[i] = -1.0;
                }
            }
        }

        let mut sig_shifted = Array1::<f64>::from_elem(n, f64::NAN);
        for i in 1..n {
            sig_shifted[i] = signal[i - 1];
        }

        let equity = calculate_equity(
            r_adjusted.view(),
            sig_shifted.view(),
            1.0,
            1.0 - decay_scaled,
            cutout,
            equity_floor,
        );

        let final_equity = equity[n - 1];
        all_equities.push(if final_equity.is_nan() { 1.0 } else { final_equity });
        all_signals.push(signal);
    }

    // Simple average of signals (original implementation that gave 88.9% consistency)
    // Layer 1 signal = mean of diflen variation signals (NOT weighted!)
    let mut combined_signal = Array1::<f64>::from_elem(n, 0.0);
    for i in 0..n {
        let mut sum = 0.0;
        let mut count = 0;
        for signal in &all_signals {
            if !signal[i].is_nan() {
                sum += signal[i];
                count += 1;
            }
        }
        if count > 0 {
            combined_signal[i] = sum / count as f64;
        } else {
            combined_signal[i] = f64::NAN;
        }
    }

    let avg_equity = if !all_equities.is_empty() {
        all_equities.iter().sum::<f64>() / all_equities.len() as f64
    } else {
        1.0
    };

    (combined_signal, avg_equity)
}

fn calculate_layer1_signal_single(
    prices: ArrayView1<f64>,
    ma_type: &str,
    length: usize,
    lambda_scaled: f64,
    decay_scaled: f64,
    cutout: usize,
    equity_floor: f64,
) -> (Array1<f64>, f64) {
    let n = prices.len();

    let ma = calculate_ma_variation(prices, ma_type, length);

    let mut roc = Array1::<f64>::from_elem(n, f64::NAN);
    for i in 1..n {
        if prices[i - 1] != 0.0 && !prices[i - 1].is_nan() {
            roc[i] = (prices[i] - prices[i - 1]) / prices[i - 1];
        }
    }

    let growth = exp_growth(lambda_scaled, n, cutout);
    let r_adjusted = &roc * &growth;

    let mut signal = Array1::<f64>::from_elem(n, 0.0);
    for i in 0..n {
        if !prices[i].is_nan() && !ma[i].is_nan() {
            if prices[i] > ma[i] {
                signal[i] = 1.0;
            } else if prices[i] < ma[i] {
                signal[i] = -1.0;
            }
        }
    }

    let mut sig_shifted = Array1::<f64>::from_elem(n, f64::NAN);
    for i in 1..n {
        sig_shifted[i] = signal[i - 1];
    }

    let equity = calculate_equity(
        r_adjusted.view(),
        sig_shifted.view(),
        1.0,
        1.0 - decay_scaled,
        cutout,
        equity_floor,
    );

    let final_weight = equity[n - 1];
    (
        signal,
        if final_weight.is_nan() {
            1.0
        } else {
            final_weight
        },
    )
}

/// Compute the final signal score for a symbol using multiple MA types
///
/// Calculates signals for all configured MA types, weights them by both
/// static configuration weights and dynamic equity weights, then aggregates
/// to produce a final score and signal classification.
///
/// # Arguments
/// * `prices` - Closing price array
/// * `config` - ATC configuration with MA types, lengths, and weights
///
/// # Returns
/// Tuple of (final_score, signal_type) where:
/// - score: -1.0 (strong SHORT) to +1.0 (strong LONG)
/// - signal_type: "LONG", "SHORT", or "NEUTRAL"
pub fn compute_symbol_score(prices: &[f64], config: &ATCConfig) -> (f64, String) {
    let prices_arr = ArrayView1::from(prices);
    let n = prices.len();
    let lambda_scaled = config.lambda_param / 1000.0;
    let decay_scaled = config.decay / 100.0;
    let robustness = Robustness::from_str(&config.robustness);

    let mut weighted_score_sum = 0.0;
    let mut total_weight = 0.0;

    for ma_config in &config.ma_configs {
        let (signal_series, equity_weight) = calculate_layer1_signal(
            prices_arr,
            &ma_config.ma_type,
            ma_config.length,
            lambda_scaled,
            decay_scaled,
            config.cutout,
            config.equity_floor,
            robustness,
        );

        let last_signal = if n > 0 { signal_series[n - 1] } else { 0.0 };

        // Note: Layer 1 signal is already continuous weighted average from diflen variations
        // Discretization happens in average_signal.py BEFORE final averaging with Layer 2 equities
        // Python: C = np.where(S > threshold, 1.0, np.where(S < -threshold, -1.0, 0.0))
        let discrete_signal = if last_signal > config.threshold {
            1.0
        } else if last_signal < -config.threshold {
            -1.0
        } else {
            0.0
        };

        let combined_weight = ma_config.weight * equity_weight;

        weighted_score_sum += discrete_signal * combined_weight;
        total_weight += combined_weight;
    }

    let final_score = if total_weight > 0.0 {
        weighted_score_sum / total_weight
    } else {
        0.0
    };

    let signal_type = if final_score > config.threshold {
        "LONG".to_string()
    } else if final_score < -config.threshold {
        "SHORT".to_string()
    } else {
        "NEUTRAL".to_string()
    };

    (final_score, signal_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diflen_narrow() {
        let result = calculate_diflen(10, Robustness::Narrow).unwrap();
        assert_eq!(result[0], 11);
        assert_eq!(result[1], 12);
        assert_eq!(result[2], 13);
        assert_eq!(result[3], 14);
        assert_eq!(result[4], 9);
        assert_eq!(result[5], 8);
        assert_eq!(result[6], 7);
        assert_eq!(result[7], 6);
    }

    #[test]
    fn test_diflen_medium() {
        let result = calculate_diflen(10, Robustness::Medium).unwrap();
        assert_eq!(result[0], 11);
        assert_eq!(result[1], 12);
        assert_eq!(result[2], 14);
        assert_eq!(result[3], 16);
        assert_eq!(result[4], 9);
        assert_eq!(result[5], 8);
        assert_eq!(result[6], 6);
        assert_eq!(result[7], 4);
    }

    #[test]
    fn test_diflen_too_small() {
        assert!(calculate_diflen(3, Robustness::Medium).is_none());
        assert!(calculate_diflen(3, Robustness::Narrow).is_none());
    }

    #[test]
    fn test_layer1_signal_with_variations() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5).collect();
        let prices_arr = ArrayView1::from(&prices);

        let (signal, equity) = calculate_layer1_signal(
            prices_arr,
            "EMA",
            20,
            0.02 / 1000.0,
            0.03 / 100.0,
            0,
            0.25,
            Robustness::Medium,
        );

        assert_eq!(signal.len(), 100);
        assert!(!equity.is_nan());
        assert!(signal[signal.len() - 1] >= 0.0);
    }
}
