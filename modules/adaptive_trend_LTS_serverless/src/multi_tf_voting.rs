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
/// * `tf_strengths` - Map of timeframe to signal strength
/// * `config` - ATC configuration containing weights and threshold
///
/// # Returns
/// Final aggregated signal result with classification
pub fn aggregate_timeframes(
    symbol: String,
    tf_scores: HashMap<String, f64>,
    tf_details: HashMap<String, String>,
    tf_strengths: HashMap<String, f64>,
    config: &ATCConfig,
) -> SignalResult {
    let mut total_weighted_score = 0.0;

    let total_config_weight: f64 = config.weights.values().sum();
    let active_weight: f64 = config
        .weights
        .iter()
        .filter(|(tf, _)| tf_scores.contains_key(*tf))
        .map(|(_, w)| *w)
        .sum();

    let weight_ratio = if total_config_weight > 0.0 {
        active_weight / total_config_weight
    } else {
        1.0
    };

    let adaptive_threshold = config.threshold * weight_ratio;

    for (tf, _) in &tf_scores {
        let tf_weight = if active_weight > 0.0 {
            config.weights.get(tf).copied().unwrap_or(0.0) / active_weight
        } else {
            0.0
        };

        let signal_type = tf_details.get(tf).map(String::as_str).unwrap_or("NEUTRAL");
        let strength = tf_strengths.get(tf).copied().unwrap_or(0.0);

        let weighted_score = calculate_weighted_score(signal_type, tf_weight, strength, config.use_signal_strength);
        total_weighted_score += weighted_score;
    }

    let mut signal_type = if total_weighted_score > adaptive_threshold {
        "LONG".to_string()
    } else if total_weighted_score < -adaptive_threshold {
        "SHORT".to_string()
    } else {
        "NEUTRAL".to_string()
    };

    if total_weighted_score.abs() < config.min_signal {
        signal_type = "NEUTRAL".to_string();
    }

    SignalResult {
        symbol,
        score: total_weighted_score,
        signal_type,
        details: tf_details,
        strengths: tf_strengths,
    }
}

fn calculate_weighted_score(
    signal_type: &str,
    tf_weight: f64,
    strength: f64,
    use_signal_strength: bool,
) -> f64 {
    match signal_type {
        "LONG" => {
            if use_signal_strength {
                tf_weight * strength.abs()
            } else {
                tf_weight
            }
        }
        "SHORT" => {
            if use_signal_strength {
                tf_weight * strength
            } else {
                -tf_weight
            }
        }
        _ => 0.0,
    }
}
