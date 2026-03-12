use crate::buffer_pool::{get_buffer, return_buffer};
use crate::constants::{
    DECAY_SCALE, LAMBDA_SCALE, MIN_LENGTH_MEDIUM, MIN_LENGTH_NARROW, MIN_LENGTH_WIDE, SIGNAL_LONG,
    SIGNAL_NEUTRAL, SIGNAL_SHORT, STARTING_EQUITY,
};
use crate::equity::{calculate_equity, exp_growth};
#[cfg(not(feature = "simd"))]
use crate::ma_calculations::*;
use crate::ATCConfig;
use crate::SignalType;
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::str::FromStr;

// Module for signal detection algorithms
//
// This module implements the core signal detection logic including:
// - diflen (differential length) calculations for robustness
// - Layer 1 signal detection with equity-based weighting
// - Multiple MA type support (EMA, HMA, WMA, DEMA, LSMA, KAMA)
// - Error handling and fallback mechanisms
//
// The main functions are `calculate_layer1_signal()` and `compute_symbol_score()`
// which implement the core ATC algorithm logic.

/// Robustness level for diflen calculation
///
/// Determines the range of length variations around the base length.
/// - Narrow: ±1, ±2, ±3, ±4 from base
/// - Medium: ±1, ±2, ±4, ±6 from base (default)
/// - Wide: ±1, ±3, ±5, ±7 from base
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum Robustness {
    /// Narrow range: ±1, ±2, ±3, ±4 from base length
    Narrow,
    /// Medium range: ±1, ±2, ±4, ±6 from base length (default)
    Medium,
    /// Wide range: ±1, ±3, ±5, ±7 from base length
    Wide,
}

/// Parameters controlling Layer 1 signal and equity computation.
#[derive(Debug, Clone, Copy)]
pub struct SignalParams {
    /// Scaled lambda parameter for equity growth adjustment.
    pub lambda_scaled: f64,
    /// Scaled decay parameter for equity curve damping.
    pub decay_scaled: f64,
    /// Number of initial bars to ignore for equity calculation.
    pub cutout: usize,
    /// Lower bound for equity values to avoid numerical instability.
    pub equity_floor: f64,
    /// Robustness profile used for diflen variations.
    pub robustness: Robustness,
}

impl FromStr for Robustness {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "narrow" => Ok(Robustness::Narrow),
            "wide" => Ok(Robustness::Wide),
            "medium" => Ok(Robustness::Medium),
            _ => Err(format!("Unknown robustness level: '{}'", s)),
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
        Robustness::Narrow => MIN_LENGTH_NARROW,
        Robustness::Medium => MIN_LENGTH_MEDIUM,
        Robustness::Wide => MIN_LENGTH_WIDE,
    };

    if length < min_required {
        crate::log_warn!(
            "Base length {} is too small for robustness {:?}. Minimum required: {}",
            length,
            robustness,
            min_required
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
    if lengths.contains(&0) {
        crate::log_error!("Calculated length offsets contain zero values");
        return None;
    }

    Some(lengths)
}

fn calculate_ma_variation(
    prices: ArrayView1<f64>,
    ma_type: &crate::MAType,
    length: usize,
) -> Array1<f64> {
    #[cfg(feature = "simd")]
    {
        match ma_type {
            crate::MAType::Ema => return crate::ma_simd::calculate_ema_simd(prices, length),
            crate::MAType::Wma => return crate::ma_simd::calculate_wma_simd(prices, length),
            crate::MAType::Hma => return crate::ma_simd::calculate_hma_simd(prices, length),
            crate::MAType::Dema => return crate::ma_simd::calculate_dema_simd(prices, length),
            crate::MAType::Lsma => return crate::ma_simd::calculate_lsma_simd(prices, length),
            crate::MAType::Kama => return crate::ma_simd::calculate_kama_simd(prices, length),
        }
    }

    #[cfg(not(feature = "simd"))]
    match ma_type {
        crate::MAType::Ema => calculate_ema(prices, length),
        crate::MAType::Hma => calculate_hma(prices, length),
        crate::MAType::Wma => calculate_wma(prices, length),
        crate::MAType::Dema => calculate_dema(prices, length),
        crate::MAType::Lsma => calculate_lsma(prices, length),
        crate::MAType::Kama => calculate_kama(prices, length),
    }
}

fn calculate_roc(prices: ArrayView1<f64>, n: usize) -> Array1<f64> {
    let mut roc = get_buffer(n);
    roc[0] = 0.0;
    for i in 1..n {
        if prices[i - 1].is_finite() && prices[i - 1] != 0.0 && prices[i].is_finite() {
            roc[i] = (prices[i] - prices[i - 1]) / prices[i - 1];
        } else {
            roc[i] = 0.0;
        }
    }
    roc
}

/// Calculate Layer 1 signal with full diflen variations (8 MA calculations)
pub fn calculate_layer1_signal(
    prices: ArrayView1<f64>,
    ma_type: &crate::MAType,
    base_length: usize,
    params: &SignalParams,
) -> (Array1<f64>, f64) {
    let n = prices.len();

    let diflen_result = match calculate_diflen(base_length, params.robustness) {
        Some(lengths) => lengths,
        None => {
            crate::log_warn!(
                "diflen failed for length {}, using base length only",
                base_length
            );
            return calculate_layer1_signal_single(prices, ma_type, base_length, params);
        }
    };

    let roc = calculate_roc(prices, n);

    let growth = exp_growth(params.lambda_scaled, n, params.cutout);
    let mut r_adjusted = get_buffer(n);
    for i in 0..n {
        r_adjusted[i] = roc[i] * growth[i];
    }
    return_buffer(roc);

    let mut all_signals: SmallVec<[Array1<f64>; 8]> = SmallVec::new();
    let mut all_equities: SmallVec<[f64; 8]> = SmallVec::new();

    for &length in &diflen_result {
        let ma = calculate_ma_variation(prices, ma_type, length);

        let mut signal = Array1::<f64>::from_elem(n, 0.0);
        for i in 0..n {
            if !prices[i].is_nan() && !ma[i].is_nan() {
                if prices[i] > ma[i] {
                    signal[i] = SIGNAL_LONG;
                } else if prices[i] < ma[i] {
                    signal[i] = SIGNAL_SHORT;
                }
            }
        }

        let mut sig_shifted = get_buffer(n);
        for i in 1..n {
            sig_shifted[i] = signal[i - 1];
        }

        let equity = calculate_equity(
            r_adjusted.view(),
            sig_shifted.view(),
            STARTING_EQUITY,
            STARTING_EQUITY - params.decay_scaled,
            params.cutout,
            params.equity_floor,
        );

        let final_equity = equity[n - 1];
        all_equities.push(if final_equity.is_nan() {
            STARTING_EQUITY
        } else {
            final_equity
        });
        all_signals.push(signal);
        return_buffer(sig_shifted);
    }

    return_buffer(r_adjusted);

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

/// Calculate Layer 1 signal for a single length (no diflen variations)
///
/// This is a simplified version of `calculate_layer1_signal` that uses only the base length
/// without diflen variations. Used as a fallback when diflen calculation fails.
///
/// # Arguments
/// * `prices` - Price data array
/// * `ma_type` - Moving Average type ("EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA")
/// * `length` - Base length for MA calculation
/// * `lambda_scaled` - Scaled lambda parameter for equity calculation
/// * `decay_scaled` - Scaled decay parameter for equity calculation
/// * `cutout` - Number of initial bars to cut out
/// * `equity_floor` - Minimum equity value to prevent numerical instability
///
/// # Returns
/// Tuple of (signal_series, final_equity) where:
/// - signal_series: Array of signals (-1.0, 0.0, or 1.0) for each price bar
/// - final_equity: Final equity value used for weighting
fn calculate_layer1_signal_single(
    prices: ArrayView1<f64>,
    ma_type: &crate::MAType,
    length: usize,
    params: &SignalParams,
) -> (Array1<f64>, f64) {
    let n = prices.len();

    let ma = calculate_ma_variation(prices, ma_type, length);

    let roc = calculate_roc(prices, n);

    let growth = exp_growth(params.lambda_scaled, n, params.cutout);
    let mut r_adjusted = get_buffer(n);
    for i in 0..n {
        r_adjusted[i] = roc[i] * growth[i];
    }
    return_buffer(roc);

    let mut signal = Array1::<f64>::from_elem(n, 0.0);
    for i in 0..n {
        if !prices[i].is_nan() && !ma[i].is_nan() {
            if prices[i] > ma[i] {
                signal[i] = SIGNAL_LONG;
            } else if prices[i] < ma[i] {
                signal[i] = SIGNAL_SHORT;
            }
        }
    }

    let mut sig_shifted = get_buffer(n);
    for i in 1..n {
        sig_shifted[i] = signal[i - 1];
    }

    let equity = calculate_equity(
        r_adjusted.view(),
        sig_shifted.view(),
        STARTING_EQUITY,
        STARTING_EQUITY - params.decay_scaled,
        params.cutout,
        params.equity_floor,
    );
    return_buffer(sig_shifted);
    return_buffer(r_adjusted);

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
/// - signal_type: SignalType::Long, SignalType::Short, or SignalType::Neutral
pub fn compute_symbol_score(prices: &[f64], config: &ATCConfig) -> (f64, SignalType) {
    let prices_arr = ArrayView1::from(prices);
    let n = prices.len();
    let params = SignalParams {
        lambda_scaled: config.lambda_param / LAMBDA_SCALE,
        decay_scaled: config.decay / DECAY_SCALE,
        cutout: config.cutout,
        equity_floor: config.equity_floor,
        robustness: config.robustness,
    };

    let mut weighted_score_sum = 0.0;
    let mut total_weight = 0.0;

    for ma_config in &config.ma_configs {
        let (signal_series, equity_weight) =
            calculate_layer1_signal(prices_arr, &ma_config.ma_type, ma_config.length, &params);

        let last_signal = if n > 0 { signal_series[n - 1] } else { 0.0 };

        // Note: Layer 1 signal is already continuous weighted average from diflen variations
        // Discretization happens in average_signal.py BEFORE final averaging with Layer 2 equities
        // Python: C = np.where(S > threshold, 1.0, np.where(S < -threshold, -1.0, 0.0))
        let discrete_signal = if last_signal > config.threshold {
            SIGNAL_LONG
        } else if last_signal < -config.threshold {
            SIGNAL_SHORT
        } else {
            SIGNAL_NEUTRAL
        };

        let combined_weight = ma_config.weight * equity_weight;

        weighted_score_sum += discrete_signal * combined_weight;
        total_weight += combined_weight;
    }

    let final_score = if total_weight > 0.0 {
        weighted_score_sum / total_weight
    } else {
        SIGNAL_NEUTRAL
    };

    let signal_type = if final_score > config.threshold {
        SignalType::Long
    } else if final_score < -config.threshold {
        SignalType::Short
    } else {
        SignalType::Neutral
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
            &crate::MAType::Ema,
            20,
            &SignalParams {
                lambda_scaled: 0.02 / 1000.0,
                decay_scaled: 0.03 / 100.0,
                cutout: 0,
                equity_floor: 0.25,
                robustness: Robustness::Medium,
            },
        );

        assert_eq!(signal.len(), 100);
        assert!(!equity.is_nan());
        assert!(signal[signal.len() - 1] >= 0.0);
    }
}
