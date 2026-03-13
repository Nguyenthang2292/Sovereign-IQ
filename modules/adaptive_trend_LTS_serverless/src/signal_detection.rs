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
    ///
    /// Used by `equity::exp_growth`, which intentionally preserves Pine/legacy
    /// semantics where the first bar (`i=0`) is treated as `bar_index=1`.
    pub lambda_scaled: f64,
    /// Scaled decay parameter for equity curve damping.
    ///
    /// Effective per-bar decay is `decay_scaled` where `decay_scaled = config.decay / DECAY_SCALE`.
    /// Example: `config.decay = 0.03` => `decay_scaled = 0.0003` (0.03% per bar).
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
    match ma_type {
        crate::MAType::Ema => crate::ma_simd::calculate_ema_simd(prices, length),
        crate::MAType::Wma => crate::ma_simd::calculate_wma_simd(prices, length),
        crate::MAType::Hma => crate::ma_simd::calculate_hma_simd(prices, length),
        crate::MAType::Dema => crate::ma_simd::calculate_dema_simd(prices, length),
        crate::MAType::Lsma => crate::ma_simd::calculate_lsma_simd(prices, length),
        crate::MAType::Kama => crate::ma_simd::calculate_kama_simd(prices, length),
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
    if n == 0 {
        return roc;
    }
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

#[inline]
fn build_r_adjusted(prices: ArrayView1<f64>, n: usize, params: &SignalParams) -> Array1<f64> {
    let roc = calculate_roc(prices, n);
    let growth = exp_growth(params.lambda_scaled, n, params.cutout);
    let mut r_adjusted = get_buffer(n);
    for i in 0..n {
        r_adjusted[i] = roc[i] * growth[i];
    }
    return_buffer(roc);
    r_adjusted
}

fn generate_persistent_signal(prices: ArrayView1<f64>, ma: &Array1<f64>) -> Array1<f64> {
    let n = prices.len();
    let mut signal = Array1::<f64>::from_elem(n, SIGNAL_NEUTRAL);
    let mut state = SIGNAL_NEUTRAL;

    for i in 1..n {
        let p_now = prices[i];
        let p_prev = prices[i - 1];
        let ma_now = ma[i];
        let ma_prev = ma[i - 1];

        if p_now.is_finite() && p_prev.is_finite() && ma_now.is_finite() && ma_prev.is_finite() {
            let crossed_up = p_now > ma_now && p_prev <= ma_prev;
            let crossed_down = p_now < ma_now && p_prev >= ma_prev;

            if crossed_up {
                state = SIGNAL_LONG;
            } else if crossed_down {
                state = SIGNAL_SHORT;
            }
        }

        signal[i] = state;
    }

    signal
}

/// Calculate Layer 1 signal with full diflen variations (base + 8 offsets).
pub fn calculate_layer1_signal(
    prices: ArrayView1<f64>,
    ma_type: &crate::MAType,
    base_length: usize,
    params: &SignalParams,
) -> (Array1<f64>, f64) {
    let n = prices.len();
    if n == 0 {
        return (Array1::<f64>::from_elem(0, SIGNAL_NEUTRAL), STARTING_EQUITY);
    }

    let r_adjusted = build_r_adjusted(prices, n, params);
    let result = calculate_layer1_signal_with_precomputed_r_adjusted(
        prices,
        ma_type,
        base_length,
        params,
        &r_adjusted,
    );
    return_buffer(r_adjusted);
    result
}

/// Internal Layer 1 implementation that reuses precomputed `r_adjusted`.
///
/// This is used to avoid recomputing `roc -> exp_growth -> r_adjusted` for each MA config
/// in `compute_symbol_score`.
fn calculate_layer1_signal_with_precomputed_r_adjusted(
    prices: ArrayView1<f64>,
    ma_type: &crate::MAType,
    base_length: usize,
    params: &SignalParams,
    r_adjusted: &Array1<f64>,
) -> (Array1<f64>, f64) {
    let n = prices.len();
    if n == 0 {
        return (Array1::<f64>::from_elem(0, SIGNAL_NEUTRAL), STARTING_EQUITY);
    }

    let diflen_result = match calculate_diflen(base_length, params.robustness) {
        Some(lengths) => lengths,
        None => {
            crate::log_warn!(
                "diflen failed for length {}, using base length only",
                base_length
            );
            return calculate_layer1_signal_single(
                prices,
                ma_type,
                base_length,
                params,
                r_adjusted,
            );
        }
    };
    let mut all_lengths: SmallVec<[usize; 9]> = SmallVec::new();
    all_lengths.push(base_length);
    for &len in &diflen_result {
        all_lengths.push(len);
    }

    let mut all_signals: SmallVec<[Array1<f64>; 9]> = SmallVec::new();
    let mut all_equity_series: SmallVec<[Array1<f64>; 9]> = SmallVec::new();
    let mut all_final_equities: SmallVec<[f64; 9]> = SmallVec::new();

    for &length in &all_lengths {
        let ma = calculate_ma_variation(prices, ma_type, length);
        let signal = generate_persistent_signal(prices, &ma);

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
        all_final_equities.push(if final_equity.is_nan() {
            STARTING_EQUITY
        } else {
            final_equity
        });
        all_signals.push(signal);
        all_equity_series.push(equity);
        return_buffer(sig_shifted);
    }

    // Source parity: weighted average by per-variation equity curves.
    let mut combined_signal = Array1::<f64>::from_elem(n, 0.0);
    for i in 0..n {
        let mut num = 0.0;
        let mut den = 0.0;
        let mut has_nan = false;
        for (signal, equity) in all_signals.iter().zip(all_equity_series.iter()) {
            let s_val = signal[i];
            let e_val = equity[i];
            if !(s_val.is_finite() && e_val.is_finite()) {
                has_nan = true;
                break;
            }
            num += s_val * e_val;
            den += e_val;
        }
        if has_nan || den == 0.0 || !den.is_finite() || !num.is_finite() {
            combined_signal[i] = f64::NAN;
        } else {
            combined_signal[i] = (num / den * 100.0).round() / 100.0;
        }
    }

    let avg_equity = if !all_final_equities.is_empty() {
        all_final_equities.iter().sum::<f64>() / all_final_equities.len() as f64
    } else {
        STARTING_EQUITY
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
/// * `ma_type` - Moving Average type (`MAType`)
/// * `length` - Base length for MA calculation
/// * `params` - Layer 1/equity parameters (scaled lambda, scaled decay, cutout, equity floor, robustness)
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
    r_adjusted: &Array1<f64>,
) -> (Array1<f64>, f64) {
    let n = prices.len();
    if n == 0 {
        return (Array1::<f64>::from_elem(0, SIGNAL_NEUTRAL), STARTING_EQUITY);
    }

    let ma = calculate_ma_variation(prices, ma_type, length);

    let signal = generate_persistent_signal(prices, &ma);

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

#[inline]
fn discretize_layer1_vote(last_signal: f64, threshold: f64) -> f64 {
    if !last_signal.is_finite() || !threshold.is_finite() || threshold < 0.0 {
        return SIGNAL_NEUTRAL;
    }

    if last_signal > threshold {
        SIGNAL_LONG
    } else if last_signal < -threshold {
        SIGNAL_SHORT
    } else {
        SIGNAL_NEUTRAL
    }
}

/// Finalize score and map to `SignalType` using source-compatible semantics.
///
/// Contract (parity with Python `adaptive_trend` module):
/// - `threshold` is applied in Layer 1 discretization (`discretize_layer1_vote`),
///   not at final sign mapping.
/// - final `SignalType` uses the sign of score after `min_signal` dead-zone filtering.
/// - `abs(score) < min_signal` is neutral; exact boundary (`abs(score) == min_signal`)
///   keeps the directional sign.
#[inline]
fn finalize_score_and_signal_type(raw_score: f64, min_signal: f64) -> (f64, SignalType) {
    let mut final_score = if raw_score.is_finite() {
        raw_score
    } else {
        SIGNAL_NEUTRAL
    };

    if final_score.abs() < min_signal {
        final_score = SIGNAL_NEUTRAL;
    }

    let signal_type = if final_score > SIGNAL_NEUTRAL {
        SignalType::Long
    } else if final_score < SIGNAL_NEUTRAL {
        SignalType::Short
    } else {
        SignalType::Neutral
    };

    (final_score, signal_type)
}

/// Compute the final signal score for a symbol using multiple MA types.
///
/// Compatibility contract (must remain stable):
/// 1. Each MA contributes its Layer 1 signal (`last_signal`) and Layer 2 equity weight.
/// 2. Layer 1 signal is discretized to `{-1, 0, 1}` using `config.threshold`.
/// 3. Final score is weighted average of these discrete votes.
/// 4. Final `SignalType` is derived from score sign after `config.min_signal`
///    dead-zone filtering (sign-only mapping, no threshold re-application).
///
/// This mirrors the original adaptive trend pipeline where thresholding happens
/// before final aggregation. Do not replace this with weighting raw continuous
/// Layer 1 values unless model parity is intentionally broken and fully revalidated.
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
    let n = prices.len();
    if n == 0 {
        return (SIGNAL_NEUTRAL, SignalType::Neutral);
    }

    let prices_arr = ArrayView1::from(prices);
    let params = SignalParams {
        lambda_scaled: config.lambda_param / LAMBDA_SCALE,
        decay_scaled: config.decay / DECAY_SCALE,
        cutout: config.cutout,
        equity_floor: config.equity_floor,
        robustness: config.robustness,
    };

    let r_adjusted = build_r_adjusted(prices_arr, n, &params);

    let mut layer1_signals: Vec<Array1<f64>> = Vec::with_capacity(config.ma_configs.len());
    let mut layer2_equities: Vec<Array1<f64>> = Vec::with_capacity(config.ma_configs.len());

    for ma_config in &config.ma_configs {
        let (signal_series, _) = calculate_layer1_signal_with_precomputed_r_adjusted(
            prices_arr,
            &ma_config.ma_type,
            ma_config.length,
            &params,
            &r_adjusted,
        );

        let mut sig_shifted = get_buffer(n);
        for i in 1..n {
            sig_shifted[i] = signal_series[i - 1];
        }

        let layer2 = calculate_equity(
            r_adjusted.view(),
            sig_shifted.view(),
            ma_config.weight,
            STARTING_EQUITY - params.decay_scaled,
            params.cutout,
            params.equity_floor,
        );
        return_buffer(sig_shifted);

        layer1_signals.push(signal_series);
        layer2_equities.push(layer2);
    }

    return_buffer(r_adjusted);

    let mut average_signal = Array1::<f64>::from_elem(n, SIGNAL_NEUTRAL);
    for i in 0..n {
        let mut num = 0.0;
        let mut den = 0.0;
        let mut invalid = false;

        for (l1, l2) in layer1_signals.iter().zip(layer2_equities.iter()) {
            let s_val = l1[i];
            let e_val = l2[i];
            if !(s_val.is_finite() && e_val.is_finite()) {
                invalid = true;
                break;
            }

            let discrete = discretize_layer1_vote(s_val, config.threshold);
            num += discrete * e_val;
            den += e_val;
        }

        if invalid || den == 0.0 || !num.is_finite() || !den.is_finite() {
            average_signal[i] = SIGNAL_NEUTRAL;
        } else {
            average_signal[i] = num / den;
        }
    }

    finalize_score_and_signal_type(average_signal[n - 1], config.min_signal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn compute_raw_average_series_for_test(prices: &[f64], config: &ATCConfig) -> Vec<f64> {
        let n = prices.len();
        if n == 0 {
            return vec![];
        }

        let prices_arr = ArrayView1::from(prices);
        let params = SignalParams {
            lambda_scaled: config.lambda_param / LAMBDA_SCALE,
            decay_scaled: config.decay / DECAY_SCALE,
            cutout: config.cutout,
            equity_floor: config.equity_floor,
            robustness: config.robustness.clone(),
        };

        let r_adjusted = build_r_adjusted(prices_arr, n, &params);

        let mut layer1_signals: Vec<Array1<f64>> = Vec::with_capacity(config.ma_configs.len());
        let mut layer2_equities: Vec<Array1<f64>> = Vec::with_capacity(config.ma_configs.len());

        for ma_config in &config.ma_configs {
            let (signal_series, _) = calculate_layer1_signal_with_precomputed_r_adjusted(
                prices_arr,
                &ma_config.ma_type,
                ma_config.length,
                &params,
                &r_adjusted,
            );

            // This shift is for equity attribution causality only (t uses signal[t-1]).
            // It is not execution-layer strategy shift.
            let mut sig_shifted = get_buffer(n);
            for i in 1..n {
                sig_shifted[i] = signal_series[i - 1];
            }

            let layer2 = calculate_equity(
                r_adjusted.view(),
                sig_shifted.view(),
                ma_config.weight,
                STARTING_EQUITY - params.decay_scaled,
                params.cutout,
                params.equity_floor,
            );
            return_buffer(sig_shifted);

            layer1_signals.push(signal_series);
            layer2_equities.push(layer2);
        }

        return_buffer(r_adjusted);

        let mut average_signal = vec![SIGNAL_NEUTRAL; n];
        for i in 0..n {
            let mut num = 0.0;
            let mut den = 0.0;
            let mut invalid = false;

            for (l1, l2) in layer1_signals.iter().zip(layer2_equities.iter()) {
                let s_val = l1[i];
                let e_val = l2[i];
                if !(s_val.is_finite() && e_val.is_finite()) {
                    invalid = true;
                    break;
                }

                let discrete = discretize_layer1_vote(s_val, config.threshold);
                num += discrete * e_val;
                den += e_val;
            }

            average_signal[i] = if invalid || den == 0.0 || !num.is_finite() || !den.is_finite() {
                SIGNAL_NEUTRAL
            } else {
                num / den
            };
        }

        average_signal
    }

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

    #[test]
    fn test_calculate_layer1_signal_empty_prices_returns_empty_series() {
        let prices: Vec<f64> = vec![];
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

        assert!(signal.is_empty());
        assert_eq!(equity, STARTING_EQUITY);
    }

    #[test]
    fn test_compute_symbol_score_empty_prices_returns_neutral() {
        let mut weights = HashMap::new();
        weights.insert("1h".to_string(), 1.0);

        let config = ATCConfig {
            robustness: Robustness::Medium,
            weights,
            threshold: 0.3,
            min_signal: 0.0,
            use_signal_strength: true,
            lambda_param: 0.02,
            decay: 0.03,
            cutout: 0,
            equity_floor: 0.25,
            ma_configs: vec![crate::MAConfig {
                ma_type: crate::MAType::Ema,
                length: 20,
                weight: 1.0,
            }],
        };

        let (score, signal_type) = compute_symbol_score(&[], &config);
        assert_eq!(score, SIGNAL_NEUTRAL);
        assert_eq!(signal_type, SignalType::Neutral);
    }

    #[test]
    fn test_compute_symbol_score_uses_raw_current_bar_not_shifted_previous_bar() {
        let mut weights = HashMap::new();
        weights.insert("1h".to_string(), 1.0);

        let config = ATCConfig {
            robustness: Robustness::Medium,
            weights,
            threshold: 0.3,
            min_signal: 0.0,
            use_signal_strength: true,
            lambda_param: 0.02,
            decay: 0.03,
            cutout: 0,
            equity_floor: 0.25,
            ma_configs: vec![
                crate::MAConfig {
                    ma_type: crate::MAType::Ema,
                    length: 20,
                    weight: 1.0,
                },
                crate::MAConfig {
                    ma_type: crate::MAType::Wma,
                    length: 20,
                    weight: 1.0,
                },
            ],
        };

        let prices: Vec<f64> = (0..160)
            .map(|idx| 100.0 + (idx as f64 * 0.2) + ((idx as f64) / 7.0).sin() * 2.0)
            .collect();

        let raw_series = compute_raw_average_series_for_test(&prices, &config);
        assert!(raw_series.len() >= 2, "Need at least 2 bars for shift-check");

        let raw_last = raw_series[raw_series.len() - 1];
        let raw_prev = raw_series[raw_series.len() - 2];

        let (score, _) = compute_symbol_score(&prices, &config);

        assert!(
            (score - raw_last).abs() < 1e-12,
            "Score must equal current-bar raw average signal"
        );

        if (raw_last - raw_prev).abs() > 1e-9 {
            assert!(
                (score - raw_prev).abs() > 1e-9,
                "Score unexpectedly matches previous raw value (shifted behavior)"
            );
        }
    }

    #[test]
    fn test_discretize_layer1_vote_contract() {
        assert_eq!(discretize_layer1_vote(0.5, 0.3), SIGNAL_LONG);
        assert_eq!(discretize_layer1_vote(-0.5, 0.3), SIGNAL_SHORT);
        assert_eq!(discretize_layer1_vote(0.2, 0.3), SIGNAL_NEUTRAL);
        assert_eq!(discretize_layer1_vote(f64::NAN, 0.3), SIGNAL_NEUTRAL);
    }

    #[test]
    fn test_discretize_layer1_vote_threshold_edges() {
        // Strict inequalities must match Python cut_signal behavior: x > long, x < short.
        assert_eq!(discretize_layer1_vote(0.3, 0.3), SIGNAL_NEUTRAL);
        assert_eq!(discretize_layer1_vote(-0.3, 0.3), SIGNAL_NEUTRAL);
        assert_eq!(discretize_layer1_vote(0.3000001, 0.3), SIGNAL_LONG);
        assert_eq!(discretize_layer1_vote(-0.3000001, 0.3), SIGNAL_SHORT);
    }

    #[test]
    fn test_finalize_score_and_signal_type_min_signal_boundaries() {
        let (score, signal) = finalize_score_and_signal_type(0.009, 0.01);
        assert_eq!(score, SIGNAL_NEUTRAL);
        assert_eq!(signal, SignalType::Neutral);

        // Exact boundary is directional (not neutral) because filter is strict: abs(score) < min_signal.
        let (score, signal) = finalize_score_and_signal_type(0.01, 0.01);
        assert_eq!(score, 0.01);
        assert_eq!(signal, SignalType::Long);

        let (score, signal) = finalize_score_and_signal_type(-0.01, 0.01);
        assert_eq!(score, -0.01);
        assert_eq!(signal, SignalType::Short);

        let (score, signal) = finalize_score_and_signal_type(f64::NAN, 0.01);
        assert_eq!(score, SIGNAL_NEUTRAL);
        assert_eq!(signal, SignalType::Neutral);
    }
}
