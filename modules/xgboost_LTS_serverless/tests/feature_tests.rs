use xgboost_serverless::features::price_derived::*;

#[test]
fn test_price_derived_lengths() {
    let high = vec![12.0, 15.0, 14.0, 16.0];
    let low = vec![10.0, 11.0, 12.0, 13.0];
    let close = vec![11.0, 13.0, 13.5, 15.0];
    let open = vec![10.5, 11.5, 13.0, 14.0];
    let volume = vec![100.0, 200.0, 150.0, 300.0];

    assert_eq!(returns_n(&close, 1).len(), close.len());
    assert_eq!(log_returns(&close).len(), close.len());
    assert_eq!(high_low_range(&high, &low, &close).len(), close.len());
    assert_eq!(close_open_diff(&open, &close).len(), close.len());
    assert_eq!(log_volume(&volume).len(), volume.len());
}

use xgboost_serverless::features::indicators::*;

#[test]
fn test_atr_positive() {
    let high = vec![12.0, 15.0, 14.0, 16.0, 15.0, 17.0, 16.0];
    let low = vec![10.0, 11.0, 12.0, 13.0, 12.0, 14.0, 11.0];
    let close = vec![11.0, 13.0, 13.5, 15.0, 14.0, 15.5, 12.0];

    let a = atr(&high, &low, &close, 3);
    for (i, &val) in a.iter().enumerate() {
        if i >= 2 {
            assert!(
                val > 0.0,
                "ATR should be positive, got {} at index {}",
                val,
                i
            );
        }
    }
}

#[test]
fn test_rsi_bounds() {
    let close = vec![
        11.0, 13.0, 13.5, 15.0, 14.0, 15.5, 12.0, 11.0, 10.0, 9.0, 8.0, 10.0, 12.0, 13.0, 15.0,
    ];
    let r = rsi(&close, 5);
    for &val in r.iter() {
        assert!((0.0..=100.0).contains(&val), "RSI out of bounds: {}", val);
    }
}

use xgboost_serverless::features::candlestick::*;

#[test]
fn test_candlestick_48_fields() {
    let high = vec![12.0, 15.0, 14.0, 16.0];
    let low = vec![10.0, 11.0, 12.0, 13.0];
    let close = vec![11.0, 13.0, 13.5, 15.0];
    let open = vec![10.5, 11.5, 13.0, 14.0];

    let p = CandlestickPatterns::detect(&open, &high, &low, &close, 3);

    // Should detect at index 3
    assert!(!p.doji); // or whatever
    assert!(!p.doji_star_bullish);

    // Convert vector
    let vec_len = p.to_feature_vec().len();
    assert_eq!(
        vec_len, 48,
        "to_feature_vec should return exactly 48 fields"
    );
}

use std::fs;
use xgboost_serverless::FeatureEngine;
use xgboost_serverless::OHLCVData;

#[test]
fn test_feature_engine_92_features() {
    let mut ts = vec![];
    let mut o = vec![];
    let mut h = vec![];
    let mut l = vec![];
    let mut c = vec![];
    let mut v = vec![];
    for i in 0..500 {
        ts.push(i as i64 * 1000);
        o.push(10.5);
        h.push(15.0);
        l.push(10.0);
        c.push(12.0);
        v.push(100.0);
    }
    let data = OHLCVData::new(ts, o, h, l, c, v).expect("OHLCV data should be valid");

    let mut engine = FeatureEngine::new();
    let result = engine.calculate_all(&data).unwrap();
    assert_eq!(
        result.len(),
        92,
        "calculate_all should return exactly 92 features"
    );
}

#[test]
fn test_test_data_file_exists_and_is_valid_json() {
    let path = "tests/test_data/btc_usdt_1h.json";
    let raw = fs::read_to_string(path).expect("test data file should exist");
    let parsed: serde_json::Value =
        serde_json::from_str(&raw).expect("test data file should contain valid JSON");
    let candles = parsed
        .as_array()
        .expect("test data JSON should be a top-level array");
    assert!(
        !candles.is_empty(),
        "test data array should contain at least one candle"
    );
}
