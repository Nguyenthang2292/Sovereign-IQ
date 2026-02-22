use crate::error::XGBoostError;
use crate::features;
use crate::ohlcv::OHLCVData;

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        f64::NAN
    } else {
        numerator / denominator
    }
}

fn normalize_timestamp(timestamp: i64) -> i64 {
    if timestamp > 1_000_000_000_000 {
        timestamp / 1000
    } else {
        timestamp
    }
}

fn temporal_features(timestamp: i64) -> [f64; 4] {
    let ts_seconds = normalize_timestamp(timestamp).max(0);
    let day_seconds = 86_400_i64;
    let hour_seconds = 3_600_i64;

    let seconds_in_day = ts_seconds % day_seconds;
    let hour = (seconds_in_day / hour_seconds) as f64;
    let day_index = ((ts_seconds / day_seconds) + 4) % 7;
    let day = day_index as f64;

    let hour_angle = 2.0 * std::f64::consts::PI * hour / 24.0;
    let day_angle = 2.0 * std::f64::consts::PI * day / 7.0;

    [
        hour_angle.sin(),
        hour_angle.cos(),
        day_angle.sin(),
        day_angle.cos(),
    ]
}

fn tail_lag_value(tail: &[f64], lag: usize) -> f64 {
    if tail.len() <= lag {
        return f64::NAN;
    }
    tail[tail.len() - 1 - lag]
}

fn returns_1_tail(close: &[f64], tail_len: usize) -> Vec<f64> {
    if tail_len == 0 {
        return vec![];
    }

    if close.len() < 2 {
        return vec![f64::NAN; tail_len];
    }

    let available = close.len() - 1;
    let take = available.min(tail_len);
    let start_index = close.len() - take;

    let mut result = Vec::with_capacity(tail_len);
    for i in start_index..close.len() {
        result.push(safe_ratio(close[i] - close[i - 1], close[i - 1]));
    }

    if result.len() < tail_len {
        let mut padded = vec![f64::NAN; tail_len - result.len()];
        padded.extend(result);
        return padded;
    }

    result
}

pub struct FeatureEngine;

impl Default for FeatureEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn calculate_all(
        &mut self,
        data: &OHLCVData,
    ) -> Result<[f64; crate::EXPECTED_FEATURE_COUNT], XGBoostError> {
        if data.is_empty() {
            return Err(XGBoostError::ValidationError("Empty data".to_string()));
        }

        let n = data.len();
        let i = n - 1;

        let mut features_arr = [0.0f64; crate::EXPECTED_FEATURE_COUNT];
        let mut idx = 0usize;

        let ret1_val = features::price_derived::returns_n_last(&data.close, 1);
        let ret5_val = features::price_derived::returns_n_last(&data.close, 5);
        let log_vol_val = features::price_derived::log_volume_last(&data.volume);

        features_arr[idx] = ret1_val;
        idx += 1;
        features_arr[idx] = ret5_val;
        idx += 1;
        features_arr[idx] = log_vol_val;
        idx += 1;

        features_arr[idx] = features::price_derived::high_low_range_last(
            &data.high,
            &data.low,
            &data.close,
        );
        idx += 1;
        features_arr[idx] = features::price_derived::close_open_diff_last(
            &data.open,
            &data.close,
        );
        idx += 1;

        let sma20_val = features::moving_averages::sma_last(&data.close, 20);
        let sma50_val = features::moving_averages::sma_last(&data.close, 50);
        let sma200_val = features::moving_averages::sma_last(&data.close, 200);
        features_arr[idx] = sma20_val;
        idx += 1;
        features_arr[idx] = sma50_val;
        idx += 1;
        features_arr[idx] = sma200_val;
        idx += 1;

        let rsi9_val = features::indicators::rsi_last_streaming(&data.close, 9);
        features_arr[idx] = rsi9_val;
        idx += 1;

        let rsi14_val = features::indicators::rsi_last_streaming(&data.close, 14);
        features_arr[idx] = rsi14_val;
        idx += 1;

        let rsi25_val = features::indicators::rsi_last_streaming(&data.close, 25);
        features_arr[idx] = rsi25_val;
        idx += 1;

        let atr14_val = features::indicators::atr_last(&data.high, &data.low, &data.close, 14);
        features_arr[idx] = atr14_val;
        idx += 1;

        let (macd_line_tail, macd_signal_val, macd_hist_val) =
            features::indicators::macd_tail(&data.close, 12, 26, 9, 4);
        let macd_line_val = tail_lag_value(&macd_line_tail, 0);
        features_arr[idx] = macd_line_val;
        idx += 1;
        features_arr[idx] = macd_hist_val;
        idx += 1;
        features_arr[idx] = macd_signal_val;
        idx += 1;

        features_arr[idx] = features::indicators::bollinger_band_percent_last(
            &data.close,
            5,
            2.0,
        );
        idx += 1;

        let (stoch_k_val, stoch_d_val) =
            features::indicators::stochastic_rsi_last(&data.close, 14, 14, 3, 3);
        features_arr[idx] = stoch_k_val;
        idx += 1;
        features_arr[idx] = stoch_d_val;
        idx += 1;

        features_arr[idx] = features::indicators::on_balance_volume_last(
            &data.close,
            &data.volume,
        );
        idx += 1;

        let patterns = features::candlestick::CandlestickPatterns::detect(
            &data.open,
            &data.high,
            &data.low,
            &data.close,
            i,
        );
        let pattern_arr = patterns.to_feature_array();
        let pattern_end = idx + pattern_arr.len();
        features_arr[idx..pattern_end].copy_from_slice(&pattern_arr);
        idx = pattern_end;

        features_arr[idx] = features::advanced::roc_last(&data.close, 3);
        idx += 1;
        features_arr[idx] = features::advanced::roc_last(&data.close, 5);
        idx += 1;
        features_arr[idx] = features::advanced::roc_last(&data.close, 10);
        idx += 1;
        features_arr[idx] = features::advanced::roc_last(&data.close, 20);
        idx += 1;

        let atr_ratio_val = safe_ratio(atr14_val, data.close[i]);
        features_arr[idx] = atr_ratio_val;
        idx += 1;

        features_arr[idx] = safe_ratio(data.close[i], sma20_val);
        idx += 1;
        features_arr[idx] = safe_ratio(data.close[i], sma50_val);
        idx += 1;
        features_arr[idx] = safe_ratio(data.close[i], sma200_val);
        idx += 1;

        let returns_1_tail_20 = returns_1_tail(&data.close, 20);
        let rolling_std_10 = features::advanced::rolling_std_last(&returns_1_tail_20, 10);
        let rolling_std_20 = features::advanced::rolling_std_last(&returns_1_tail_20, 20);
        let rolling_skew_10 = features::advanced::rolling_skewness_last(&returns_1_tail_20, 10);
        let rolling_skew_20 = features::advanced::rolling_skewness_last(&returns_1_tail_20, 20);
        features_arr[idx] = rolling_std_10;
        idx += 1;
        features_arr[idx] = rolling_std_20;
        idx += 1;
        features_arr[idx] = rolling_skew_10;
        idx += 1;
        features_arr[idx] = rolling_skew_20;
        idx += 1;

        let ret1_tail_4 = returns_1_tail(&data.close, 4);
        features_arr[idx] = tail_lag_value(&ret1_tail_4, 1);
        idx += 1;
        features_arr[idx] = tail_lag_value(&ret1_tail_4, 2);
        idx += 1;
        features_arr[idx] = tail_lag_value(&ret1_tail_4, 3);
        idx += 1;

        let rsi14_tail = features::indicators::rsi_tail_streaming(&data.close, 14, 4);
        features_arr[idx] = tail_lag_value(&rsi14_tail, 1);
        idx += 1;
        features_arr[idx] = tail_lag_value(&rsi14_tail, 2);
        idx += 1;
        features_arr[idx] = tail_lag_value(&rsi14_tail, 3);
        idx += 1;

        features_arr[idx] = tail_lag_value(&macd_line_tail, 1);
        idx += 1;
        features_arr[idx] = tail_lag_value(&macd_line_tail, 2);
        idx += 1;
        features_arr[idx] = tail_lag_value(&macd_line_tail, 3);
        idx += 1;

        let [hour_sin, hour_cos, day_sin, day_cos] = temporal_features(data.timestamp[i]);
        features_arr[idx] = hour_sin;
        idx += 1;
        features_arr[idx] = hour_cos;
        idx += 1;
        features_arr[idx] = day_sin;
        idx += 1;
        features_arr[idx] = day_cos;
        idx += 1;

        if idx != crate::EXPECTED_FEATURE_COUNT {
            return Err(XGBoostError::FeatureEngineeringError(format!(
                "Expected {} features, got {}",
                crate::EXPECTED_FEATURE_COUNT,
                idx
            )));
        }

        Ok(features_arr)
    }
}
