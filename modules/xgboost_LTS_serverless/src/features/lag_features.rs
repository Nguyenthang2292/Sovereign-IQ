#![allow(dead_code)]

pub fn create_lag_features(data: &[f64], max_lag: usize) -> Vec<Vec<f64>> {
    let mut lag_features = Vec::new();

    for lag in 1..=max_lag {
        let mut lag_column = vec![0.0; lag];
        for i in lag..data.len() {
            lag_column.push(data[i - lag]);
        }
        lag_features.push(lag_column);
    }

    lag_features
}

pub fn create_rolling_lags(data: &[f64], window_size: usize, lags: usize) -> Vec<Vec<f64>> {
    let mut rolling_lags = Vec::new();

    for lag in 1..=lags {
        let mut lag_column = vec![0.0; window_size + lag - 1];

        for i in (window_size + lag - 1)..data.len() {
            let window_start = i + 1 - window_size - lag;
            let window_end = i + 1 - lag;
            if window_end > window_start {
                let window = &data[window_start..window_end];
                lag_column.push(window.iter().sum::<f64>() / window.len() as f64);
            }
        }

        rolling_lags.push(lag_column);
    }

    rolling_lags
}
