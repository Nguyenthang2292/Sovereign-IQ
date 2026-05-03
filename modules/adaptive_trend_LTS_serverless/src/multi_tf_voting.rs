//! Multi-timeframe signal aggregation and weighted voting.
//!
//! ## Dual-Threshold Architecture
//!
//! `config.threshold` is applied at **two independent decision points** in the pipeline:
//!
//! 1. **Layer 1 discretization** (`signal_detection::discretize_layer1_vote`):
//!    The continuous combined-MA signal for each timeframe is discretized to `{-1, 0, +1}`
//!    before Layer 2 equity weighting and before this module is reached.
//!    Any timeframe signal with `|score| ≤ threshold` is already mapped to `0.0 (NEUTRAL)`.
//!
//! 2. **Timeframe aggregation** (`aggregate_timeframes`, this module):
//!    The weighted combination of per-timeframe scores is again compared against `threshold`
//!    to classify the final `SignalType` as `Long`, `Short`, or `Neutral`.
//!
//! **Tuning implication**: Increasing `threshold` tightens *both* gates simultaneously.
//! A timeframe must pass the Layer 1 gate before its vote can influence the Layer 2 outcome,
//! which means the effective sensitivity of the final signal is non-linearly coupled to this
//! single parameter. Do not assume a 1:1 mapping between `threshold` and the fraction of
//! LONG/SHORT signals emitted.

use crate::{ATCConfig, SignalResult};
use std::collections::HashMap;

/// Aggregate signals across multiple timeframes with weighted averaging
///
/// Combines signal scores from different timeframes (e.g., 1h, 4h) using
/// configured weights to produce a final signal classification.
///
/// # Arguments
/// * `symbol` - Symbol identifier
/// * `tf_scores` - Map of timeframe to signal score
/// * `tf_details` - Map of timeframe to signal type (LONG/SHORT/NEUTRAL)
/// * `config` - ATC configuration containing weights and threshold
///
/// # Returns
/// Final aggregated signal result with classification
pub fn aggregate_timeframes(
    symbol: String,
    tf_scores: HashMap<String, f64>,
    tf_details: HashMap<String, crate::SignalType>,
    config: &ATCConfig,
) -> SignalResult {
    let mut total_weighted_score = 0.0;

    let total_config_weight: f64 = config.weights.values().sum();

    for tf in tf_scores.keys() {
        // Keep configured timeframe weights invariant so missing timeframes do not
        // amplify remaining signals. Validation rejects missing configured timeframes,
        // but this guard keeps behavior stable if aggregate_timeframes is used directly.
        let tf_weight = if total_config_weight > 0.0 {
            config.weights.get(tf).copied().unwrap_or(0.0) / total_config_weight
        } else {
            0.0
        };

        let signal_type = tf_details.get(tf).unwrap_or(&crate::SignalType::Neutral);
        let strength = tf_scores.get(tf).copied().unwrap_or(0.0);

        let weighted_score =
            calculate_weighted_score(signal_type, tf_weight, strength, config.use_signal_strength);
        total_weighted_score += weighted_score;
    }

    let mut signal_type = if total_weighted_score > config.threshold {
        crate::SignalType::Long
    } else if total_weighted_score < -config.threshold {
        crate::SignalType::Short
    } else {
        crate::SignalType::Neutral
    };

    if total_weighted_score.abs() < config.min_signal {
        signal_type = crate::SignalType::Neutral;
    }

    SignalResult {
        symbol,
        score: total_weighted_score,
        signal_type,
        details: tf_details,
        strengths: tf_scores,
        average_signal_raw: Some(total_weighted_score),
        average_signal_exec: None,
    }
}

fn calculate_weighted_score(
    signal_type: &crate::SignalType,
    tf_weight: f64,
    strength: f64,
    use_signal_strength: bool,
) -> f64 {
    match signal_type {
        crate::SignalType::Long => {
            if use_signal_strength {
                tf_weight * strength.abs()
            } else {
                tf_weight
            }
        }
        crate::SignalType::Short => {
            if use_signal_strength {
                -tf_weight * strength.abs()
            } else {
                -tf_weight
            }
        }
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ATCConfig, MAConfig, MAType};
    use std::collections::HashMap;

    #[test]
    fn test_aggregate_no_matching_timeframe() {
        let mut config_weights = HashMap::new();
        config_weights.insert("1d".to_string(), 1.0);

        let config = ATCConfig {
            weights: config_weights,
            threshold: 0.3,
            min_signal: 0.0,
            use_signal_strength: false,
            lambda_param: 0.02,
            decay: 0.03,
            cutout: 0,
            equity_floor: 0.25,
            robustness: crate::Robustness::Medium,
            ma_configs: vec![MAConfig {
                ma_type: MAType::Ema,
                length: 12,
                weight: 1.0,
            }],
        };

        let mut tf_scores = HashMap::new();
        tf_scores.insert("1h".to_string(), 0.8);

        let mut tf_details = HashMap::new();
        tf_details.insert("1h".to_string(), crate::SignalType::Long);

        let result = aggregate_timeframes("BTCUSDT".to_string(), tf_scores, tf_details, &config);

        assert_eq!(result.signal_type, crate::SignalType::Neutral);
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn test_aggregate_timeframes_sets_raw_contract_fields() {
        let mut config_weights = HashMap::new();
        config_weights.insert("1h".to_string(), 1.0);

        let config = ATCConfig {
            weights: config_weights,
            threshold: 0.3,
            min_signal: 0.0,
            use_signal_strength: false,
            lambda_param: 0.02,
            decay: 0.03,
            cutout: 0,
            equity_floor: 0.25,
            robustness: crate::Robustness::Medium,
            ma_configs: vec![MAConfig {
                ma_type: MAType::Ema,
                length: 12,
                weight: 1.0,
            }],
        };

        let mut tf_scores = HashMap::new();
        tf_scores.insert("1h".to_string(), 0.8);

        let mut tf_details = HashMap::new();
        tf_details.insert("1h".to_string(), crate::SignalType::Long);

        let result = aggregate_timeframes("BTCUSDT".to_string(), tf_scores, tf_details, &config);

        assert_eq!(result.average_signal_raw, Some(result.score));
        assert_eq!(result.average_signal_exec, None);
    }
}
