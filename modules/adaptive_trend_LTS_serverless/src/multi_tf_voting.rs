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
    let mut total_weight = 0.0;

    for (tf, score) in &tf_scores {
        if let Some(weight) = config.weights.get(tf) {
            total_weighted_score += score * weight;
            total_weight += weight;
        }
    }

    let final_score = if total_weight > 0.0 {
        total_weighted_score / total_weight
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

    SignalResult {
        symbol,
        score: final_score,
        signal_type,
        details: tf_details,
        strengths: tf_strengths,
    }
}
