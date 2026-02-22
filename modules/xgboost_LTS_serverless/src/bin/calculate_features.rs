use std::collections::BTreeMap;
use std::env;
use std::fs;

use serde_json::Value;
use xgboost_serverless::{FeatureEngine, OHLCVData, EXPECTED_FEATURE_COUNT};

const FEATURE_NAMES: [&str; EXPECTED_FEATURE_COUNT] = [
    "returns_1",
    "returns_5",
    "log_volume",
    "high_low_range",
    "close_open_diff",
    "sma_20",
    "sma_50",
    "sma_200",
    "rsi_9",
    "rsi_14",
    "rsi_25",
    "atr_14",
    "macd_12_26_9",
    "macdh_12_26_9",
    "macds_12_26_9",
    "bbp_5_2.0",
    "stochrsi_k_14_14_3_3",
    "stochrsi_d_14_14_3_3",
    "obv",
    "doji",
    "hammer",
    "engulfing_bullish",
    "engulfing_bearish",
    "morning_star",
    "evening_star",
    "inverted_hammer",
    "shooting_star",
    "marubozu_bull",
    "marubozu_bear",
    "spinning_top",
    "gravestone_doji",
    "dragonfly_doji",
    "long_legged_doji",
    "bullish_harami",
    "bearish_harami",
    "piercing",
    "dark_cloud_cover",
    "tweezer_top",
    "tweezer_bottom",
    "bullish_belt_hold",
    "bearish_belt_hold",
    "three_white_soldiers",
    "three_black_crows",
    "bullish_abandoned_baby",
    "bearish_abandoned_baby",
    "bullish_tri_star",
    "bearish_tri_star",
    "rising_three_methods",
    "falling_three_methods",
    "three_inside_up",
    "three_inside_down",
    "three_outside_up",
    "three_outside_down",
    "harami_cross_bull",
    "harami_cross_bear",
    "rising_window",
    "falling_window",
    "tasuki_gap_bull",
    "tasuki_gap_bear",
    "mat_hold_bull",
    "mat_hold_bear",
    "advance_block",
    "stalled_pattern",
    "kicker_bull",
    "kicker_bear",
    "hanging_man",
    "doji_star_bullish",
    "roc_3",
    "roc_5",
    "roc_10",
    "roc_20",
    "atr_ratio",
    "price_to_sma_20",
    "price_to_sma_50",
    "price_to_sma_200",
    "rolling_std_10",
    "rolling_std_20",
    "rolling_skew_10",
    "rolling_skew_20",
    "returns_1_lag_1",
    "returns_1_lag_2",
    "returns_1_lag_3",
    "rsi_14_lag_1",
    "rsi_14_lag_2",
    "rsi_14_lag_3",
    "macd_lag_1",
    "macd_lag_2",
    "macd_lag_3",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
];

fn parse_vec_f64(data: &Value, key: &str) -> Result<Vec<f64>, String> {
    let values = data
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array field: {key}"))?;

    values
        .iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| format!("invalid number in {key}"))
        })
        .collect()
}

fn parse_vec_i64(data: &Value, key: &str) -> Result<Vec<i64>, String> {
    let values = data
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array field: {key}"))?;

    values
        .iter()
        .map(|value| {
            value
                .as_i64()
                .or_else(|| value.as_f64().map(|number| number as i64))
                .ok_or_else(|| format!("invalid integer in {key}"))
        })
        .collect()
}

fn parse_ohlcv(input: &str) -> Result<OHLCVData, String> {
    let parsed: Value = serde_json::from_str(input).map_err(|error| error.to_string())?;

    if let Some(values) = parsed.as_array() {
        if values.len() % 6 != 0 {
            return Err("flat array length must be divisible by 6".to_string());
        }

        let mut timestamp = Vec::with_capacity(values.len() / 6);
        let mut open = Vec::with_capacity(values.len() / 6);
        let mut high = Vec::with_capacity(values.len() / 6);
        let mut low = Vec::with_capacity(values.len() / 6);
        let mut close = Vec::with_capacity(values.len() / 6);
        let mut volume = Vec::with_capacity(values.len() / 6);

        for chunk in values.chunks_exact(6) {
            timestamp.push(
                chunk[0]
                    .as_i64()
                    .or_else(|| chunk[0].as_f64().map(|number| number as i64))
                    .ok_or_else(|| "invalid timestamp value".to_string())?,
            );
            open.push(
                chunk[1]
                    .as_f64()
                    .ok_or_else(|| "invalid open value".to_string())?,
            );
            high.push(
                chunk[2]
                    .as_f64()
                    .ok_or_else(|| "invalid high value".to_string())?,
            );
            low.push(
                chunk[3]
                    .as_f64()
                    .ok_or_else(|| "invalid low value".to_string())?,
            );
            close.push(
                chunk[4]
                    .as_f64()
                    .ok_or_else(|| "invalid close value".to_string())?,
            );
            volume.push(
                chunk[5]
                    .as_f64()
                    .ok_or_else(|| "invalid volume value".to_string())?,
            );
        }

        return OHLCVData::new(timestamp, open, high, low, close, volume)
            .map_err(|error| error.to_string());
    }

    OHLCVData::new(
        parse_vec_i64(&parsed, "timestamp")?,
        parse_vec_f64(&parsed, "open")?,
        parse_vec_f64(&parsed, "high")?,
        parse_vec_f64(&parsed, "low")?,
        parse_vec_f64(&parsed, "close")?,
        parse_vec_f64(&parsed, "volume")?,
    )
    .map_err(|error| error.to_string())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        return Err(
            "usage: cargo run --bin calculate_features -- <path-to-ohlcv-json>".to_string(),
        );
    }

    let raw = fs::read_to_string(&args[1]).map_err(|error| error.to_string())?;
    let ohlcv = parse_ohlcv(&raw)?;

    let mut engine = FeatureEngine::new();
    let features = engine
        .calculate_all(&ohlcv)
        .map_err(|error| error.to_string())?;

    if features.len() != FEATURE_NAMES.len() {
        return Err(format!(
            "feature length mismatch: expected {}, got {}",
            FEATURE_NAMES.len(),
            features.len()
        ));
    }

    let mut output = BTreeMap::new();
    for (index, feature_name) in FEATURE_NAMES.iter().enumerate() {
        output.insert((*feature_name).to_string(), features[index]);
    }

    let json = serde_json::to_string_pretty(&output).map_err(|error| error.to_string())?;
    println!("{json}");

    Ok(())
}
