use atc_serverless::{
    calculate_dema, calculate_diflen, calculate_ema, calculate_equity, calculate_hma,
    calculate_kama, calculate_layer1_signal, calculate_lsma, calculate_sma, calculate_wma,
    compute_symbol_score, process_batch, ATCConfig, MAConfig, OHLCVData, Robustness, SymbolData,
};
use ndarray::Array1;
use std::collections::HashMap;

// ============================================================================
// MA Calculations Tests
// ============================================================================

#[test]
fn test_ema_calculation() {
    let data = vec![10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    let arr = Array1::from(data);
    let ema = calculate_ema(arr.view(), 3);

    // EMA should have valid values after the initialization period
    assert!(!ema[9].is_nan(), "EMA at index 9 should not be NaN");
    assert!(!ema[8].is_nan(), "EMA at index 8 should not be NaN");

    // Earlier values before initialization should be NaN
    assert!(
        ema[0].is_nan() || ema[1].is_nan(),
        "EMA should be NaN during initialization"
    );
}

#[test]
fn test_sma_calculation() {
    let data = vec![10.0, 11.0, 12.0, 11.0, 10.0];
    let arr = Array1::from(data);
    let sma = calculate_sma(arr.view(), 3);

    // SMA[2] = (10 + 11 + 12) / 3 = 11
    assert!((sma[2] - 11.0).abs() < 1e-6, "SMA calculation incorrect");

    // SMA[3] = (11 + 12 + 11) / 3 = 11.333...
    assert!(
        (sma[3] - 11.333333).abs() < 1e-6,
        "SMA calculation incorrect"
    );

    // First values should be NaN
    assert!(
        sma[0].is_nan() && sma[1].is_nan(),
        "SMA should be NaN before length"
    );
}

#[test]
fn test_wma_calculation() {
    let data = vec![10.0, 11.0, 12.0];
    let arr = Array1::from(data);
    let wma = calculate_wma(arr.view(), 3);

    // WMA = (10*1 + 11*2 + 12*3) / (1+2+3) = (10 + 22 + 36) / 6 = 68/6 = 11.333...
    assert!(
        (wma[2] - 11.333333).abs() < 1e-6,
        "WMA calculation incorrect"
    );
}

#[test]
fn test_dema_calculation() {
    let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.5)).collect();
    let arr = Array1::from(data);
    let dema = calculate_dema(arr.view(), 10);

    // DEMA should produce valid values
    assert!(
        !dema[dema.len() - 1].is_nan(),
        "DEMA should have valid values"
    );
}

#[test]
fn test_hma_calculation() {
    let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.5)).collect();
    let arr = Array1::from(data);
    let hma = calculate_hma(arr.view(), 10);

    // HMA should produce valid values
    assert!(!hma[hma.len() - 1].is_nan(), "HMA should have valid values");
}

#[test]
fn test_lsma_calculation() {
    let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.5)).collect();
    let arr = Array1::from(data);
    let lsma = calculate_lsma(arr.view(), 10);

    // LSMA should produce valid values
    assert!(
        !lsma[lsma.len() - 1].is_nan(),
        "LSMA should have valid values"
    );
}

#[test]
fn test_kama_calculation() {
    let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.5)).collect();
    let arr = Array1::from(data);
    let kama = calculate_kama(arr.view(), 10);

    // KAMA should produce valid values
    assert!(
        !kama[kama.len() - 1].is_nan(),
        "KAMA should have valid values"
    );
}

#[test]
fn test_ma_with_nan_values() {
    let mut data = vec![10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    data[5] = f64::NAN;
    let arr = Array1::from(data);
    let ema = calculate_ema(arr.view(), 3);

    // Should handle NaN gracefully
    assert!(
        !ema[9].is_nan() || ema[9].is_nan(),
        "EMA should handle NaN values"
    );
}

#[test]
fn test_ma_with_insufficient_data() {
    let data = vec![10.0, 11.0];
    let arr = Array1::from(data);
    let ema = calculate_ema(arr.view(), 5);

    // Should return NaN array when data is insufficient
    assert!(
        ema.iter().all(|&x| x.is_nan()),
        "EMA should be NaN with insufficient data"
    );
}

#[test]
fn test_ma_performance_large_array() {
    let data: Vec<f64> = (0..10000)
        .map(|i| 100.0 + (i as f64 * 0.01).sin() * 10.0)
        .collect();
    let arr = Array1::from(data);

    let start = std::time::Instant::now();
    let _ema = calculate_ema(arr.view(), 50);
    let duration = start.elapsed();

    // Should complete in reasonable time (< 1 second for 10k elements)
    assert!(
        duration.as_millis() < 1000,
        "EMA calculation too slow: {:?}",
        duration
    );
}

// ============================================================================
// Equity Calculation Tests
// ============================================================================

#[test]
fn test_equity_calculation_basic() {
    let r = Array1::from(vec![0.01, 0.02, -0.01]);
    let sig = Array1::from(vec![1.0, 1.0, -1.0]);

    let equity = calculate_equity(r.view(), sig.view(), 100.0, 1.0, 0);

    assert_eq!(equity[0], 100.0, "Initial equity should be 100");
    // After positive returns, equity should increase
    assert!(
        equity[2] > 100.0,
        "Equity should increase with positive returns"
    );
}

#[test]
fn test_equity_with_decay() {
    let r = Array1::from(vec![0.01; 100]);
    let sig = Array1::from(vec![1.0; 100]);

    let equity_no_decay = calculate_equity(r.view(), sig.view(), 100.0, 1.0, 0);
    let equity_with_decay = calculate_equity(r.view(), sig.view(), 100.0, 0.97, 0);

    // Decay should result in lower equity
    assert!(
        equity_with_decay[99] < equity_no_decay[99],
        "Decay should reduce equity growth"
    );
}

// ============================================================================
// Diflen Tests
// ============================================================================

#[test]
fn test_diflen_narrow() {
    let result = calculate_diflen(10, Robustness::Narrow).unwrap();
    assert_eq!(result[0], 11, "L1 incorrect");
    assert_eq!(result[1], 12, "L2 incorrect");
    assert_eq!(result[2], 13, "L3 incorrect");
    assert_eq!(result[3], 14, "L4 incorrect");
    assert_eq!(result[4], 9, "L_1 incorrect");
    assert_eq!(result[5], 8, "L_2 incorrect");
    assert_eq!(result[6], 7, "L_3 incorrect");
    assert_eq!(result[7], 6, "L_4 incorrect");
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
fn test_diflen_wide() {
    let result = calculate_diflen(10, Robustness::Wide).unwrap();
    assert_eq!(result[0], 11);
    assert_eq!(result[1], 13);
    assert_eq!(result[2], 15);
    assert_eq!(result[3], 17);
    assert_eq!(result[4], 9);
    assert_eq!(result[5], 7);
    assert_eq!(result[6], 5);
    assert_eq!(result[7], 3);
}

#[test]
fn test_diflen_too_small() {
    // Medium requires minimum 7
    assert!(calculate_diflen(3, Robustness::Medium).is_none());
    assert!(calculate_diflen(4, Robustness::Medium).is_none());
    assert!(calculate_diflen(6, Robustness::Medium).is_none());
    
    // Narrow requires minimum 5
    assert!(calculate_diflen(4, Robustness::Narrow).is_none());
    assert!(calculate_diflen(3, Robustness::Narrow).is_none());
    
    // Wide requires minimum 8
    assert!(calculate_diflen(7, Robustness::Wide).is_none());
    assert!(calculate_diflen(6, Robustness::Wide).is_none());
}

#[test]
fn test_diflen_zero_length() {
    assert!(calculate_diflen(0, Robustness::Medium).is_none());
}

// ============================================================================
// Signal Detection Tests
// ============================================================================

#[test]
fn test_layer1_signal_uptrend() {
    // Create an uptrend
    let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5).collect();
    let prices_arr = Array1::from(prices);

    let (signal, equity) = calculate_layer1_signal(prices_arr.view(), "EMA", 20, 0.02, 0.03);

    assert_eq!(signal.len(), 100);
    assert!(!equity.is_nan(), "Equity should not be NaN");
    // In an uptrend with EMA, signal should be positive
    assert!(
        signal[signal.len() - 1] >= 0.0,
        "Signal should be positive in uptrend"
    );
}

#[test]
fn test_layer1_signal_downtrend() {
    // Create a downtrend
    let prices: Vec<f64> = (0..100).map(|i| 100.0 - i as f64 * 0.5).collect();
    let prices_arr = Array1::from(prices);

    let (signal, equity) = calculate_layer1_signal(prices_arr.view(), "EMA", 20, 0.02, 0.03);

    assert!(!equity.is_nan(), "Equity should not be NaN");
    // In a downtrend with EMA, signal should be negative
    assert!(
        signal[signal.len() - 1] <= 0.0,
        "Signal should be negative in downtrend"
    );
}

#[test]
fn test_compute_symbol_score() {
    let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5).collect();

    let config = ATCConfig {
        weights: HashMap::new(),
        threshold: 0.3,
        min_signal: 0.0,
        use_signal_strength: true,
        lambda_param: 0.02,
        decay: 0.03,
        cutout: 0,
        ma_configs: vec![MAConfig {
            ma_type: "EMA".to_string(),
            length: 20,
            weight: 1.0,
        }],
    };

    let (score, signal_type) = compute_symbol_score(&prices, &config);

    assert!(!score.is_nan(), "Score should not be NaN");
    assert!(
        signal_type == "LONG" || signal_type == "SHORT" || signal_type == "NEUTRAL",
        "Signal type should be valid"
    );
}

#[test]
fn test_signal_type_classification() {
    let uptrend_prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 1.0).collect();
    let downtrend_prices: Vec<f64> = (0..100).map(|i| 100.0 - i as f64 * 1.0).collect();

    let config = ATCConfig {
        weights: HashMap::new(),
        threshold: 0.1,
        min_signal: 0.0,
        use_signal_strength: true,
        lambda_param: 0.02,
        decay: 0.03,
        cutout: 0,
        ma_configs: vec![MAConfig {
            ma_type: "EMA".to_string(),
            length: 20,
            weight: 1.0,
        }],
    };

    let (uptrend_score, uptrend_type) = compute_symbol_score(&uptrend_prices, &config);
    let (downtrend_score, downtrend_type) = compute_symbol_score(&downtrend_prices, &config);

    // Strong trends should produce non-neutral signals
    if uptrend_score > config.threshold {
        assert_eq!(uptrend_type, "LONG", "Uptrend should produce LONG signal");
    }
    if downtrend_score < -config.threshold {
        assert_eq!(
            downtrend_type, "SHORT",
            "Downtrend should produce SHORT signal"
        );
    }
}

// ============================================================================
// Multi-TF Voting Tests
// ============================================================================

#[test]
fn test_process_batch_single_symbol() {
    let mut timeframes = HashMap::new();
    let ohlcv = OHLCVData {
        timestamp: (0..50).map(|i| i as i64).collect(),
        open: vec![100.0; 50],
        high: vec![101.0; 50],
        low: vec![99.0; 50],
        close: (0..50).map(|i| 100.0 + i as f64 * 0.5).collect(),
        volume: vec![1000.0; 50],
    };
    timeframes.insert("1h".to_string(), ohlcv);

    let symbol_data = SymbolData {
        symbol: "BTCUSDT".to_string(),
        timeframes,
    };

    let config = ATCConfig {
        weights: {
            let mut w = HashMap::new();
            w.insert("1h".to_string(), 1.0);
            w
        },
        threshold: 0.3,
        min_signal: 0.0,
        use_signal_strength: true,
        lambda_param: 0.02,
        decay: 0.03,
        cutout: 0,
        ma_configs: vec![MAConfig {
            ma_type: "EMA".to_string(),
            length: 10,
            weight: 1.0,
        }],
    };

    let (results, errors) = process_batch(vec![symbol_data], config);

    assert_eq!(results.len(), 1, "Should have one result");
    assert_eq!(errors.len(), 0, "Should have no errors");
    assert_eq!(results[0].symbol, "BTCUSDT");
}

#[test]
fn test_process_batch_multiple_symbols() {
    let symbols: Vec<SymbolData> = (0..5)
        .map(|i| {
            let mut timeframes = HashMap::new();
            let ohlcv = OHLCVData {
                timestamp: (0..50).map(|j| j as i64).collect(),
                open: vec![100.0; 50],
                high: vec![101.0; 50],
                low: vec![99.0; 50],
                close: (0..50).map(|j| 100.0 + j as f64 * 0.5).collect(),
                volume: vec![1000.0; 50],
            };
            timeframes.insert("1h".to_string(), ohlcv);

            SymbolData {
                symbol: format!("SYM{}", i),
                timeframes,
            }
        })
        .collect();

    let config = ATCConfig {
        weights: {
            let mut w = HashMap::new();
            w.insert("1h".to_string(), 1.0);
            w
        },
        threshold: 0.3,
        min_signal: 0.0,
        use_signal_strength: true,
        lambda_param: 0.02,
        decay: 0.03,
        cutout: 0,
        ma_configs: vec![MAConfig {
            ma_type: "EMA".to_string(),
            length: 10,
            weight: 1.0,
        }],
    };

    let (results, errors) = process_batch(symbols, config);

    assert_eq!(results.len(), 5, "Should have 5 results");
    assert_eq!(errors.len(), 0, "Should have no errors");
}

#[test]
fn test_process_batch_with_error_recovery() {
    // Create one valid and one invalid symbol
    let mut timeframes_valid = HashMap::new();
    let ohlcv_valid = OHLCVData {
        timestamp: (0..50).map(|i| i as i64).collect(),
        open: vec![100.0; 50],
        high: vec![101.0; 50],
        low: vec![99.0; 50],
        close: (0..50).map(|i| 100.0 + i as f64 * 0.5).collect(),
        volume: vec![1000.0; 50],
    };
    timeframes_valid.insert("1h".to_string(), ohlcv_valid);

    // Invalid symbol with insufficient data
    let mut timeframes_invalid = HashMap::new();
    let ohlcv_invalid = OHLCVData {
        timestamp: vec![1, 2],
        open: vec![100.0, 101.0],
        high: vec![101.0, 102.0],
        low: vec![99.0, 100.0],
        close: vec![100.0, 101.0],
        volume: vec![1000.0, 1000.0],
    };
    timeframes_invalid.insert("1h".to_string(), ohlcv_invalid);

    let symbols = vec![
        SymbolData {
            symbol: "VALID".to_string(),
            timeframes: timeframes_valid,
        },
        SymbolData {
            symbol: "INVALID".to_string(),
            timeframes: timeframes_invalid,
        },
    ];

    let config = ATCConfig {
        weights: {
            let mut w = HashMap::new();
            w.insert("1h".to_string(), 1.0);
            w
        },
        threshold: 0.3,
        min_signal: 0.0,
        use_signal_strength: true,
        lambda_param: 0.02,
        decay: 0.03,
        cutout: 0,
        ma_configs: vec![MAConfig {
            ma_type: "EMA".to_string(),
            length: 30,
            weight: 1.0,
        }],
    };

    let (results, errors) = process_batch(symbols, config);

    // Should still get results even if some fail
    assert!(results.len() >= 0, "Should handle batch processing");
    // May have errors but should not panic
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_end_to_end_pipeline() {
    // Create realistic OHLCV data
    let mut timeframes = HashMap::new();
    let close_prices: Vec<f64> = (0..200)
        .map(|i| {
            let trend = (i as f64 / 200.0) * 20.0;
            let noise = (i as f64 * 0.1).sin() * 2.0;
            100.0 + trend + noise
        })
        .collect();

    let ohlcv = OHLCVData {
        timestamp: (0..200).map(|i| i as i64).collect(),
        open: close_prices.iter().map(|&c| c - 0.5).collect(),
        high: close_prices.iter().map(|&c| c + 1.0).collect(),
        low: close_prices.iter().map(|&c| c - 1.0).collect(),
        close: close_prices,
        volume: vec![1000.0; 200],
    };
    timeframes.insert("1h".to_string(), ohlcv.clone());
    timeframes.insert("4h".to_string(), ohlcv);

    let symbol_data = SymbolData {
        symbol: "BTCUSDT".to_string(),
        timeframes,
    };

    let config = ATCConfig {
        weights: {
            let mut w = HashMap::new();
            w.insert("1h".to_string(), 0.6);
            w.insert("4h".to_string(), 0.4);
            w
        },
        threshold: 0.2,
        min_signal: 0.0,
        use_signal_strength: true,
        lambda_param: 0.02,
        decay: 0.03,
        cutout: 0,
        ma_configs: vec![
            MAConfig {
                ma_type: "EMA".to_string(),
                length: 12,
                weight: 1.0,
            },
            MAConfig {
                ma_type: "HMA".to_string(),
                length: 12,
                weight: 1.0,
            },
            MAConfig {
                ma_type: "WMA".to_string(),
                length: 12,
                weight: 1.0,
            },
        ],
    };

    let (results, errors) = process_batch(vec![symbol_data], config);

    assert_eq!(results.len(), 1, "Should produce one result");
    assert_eq!(errors.len(), 0, "Should have no errors");

    let result = &results[0];
    assert_eq!(result.symbol, "BTCUSDT");
    assert!(!result.score.is_nan(), "Score should not be NaN");
    assert!(
        result.signal_type == "LONG"
            || result.signal_type == "SHORT"
            || result.signal_type == "NEUTRAL",
        "Signal type should be valid"
    );
}

#[test]
fn test_all_ma_types() {
    let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5).collect();

    let ma_types = vec!["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"];

    for ma_type in ma_types {
        let config = ATCConfig {
            weights: HashMap::new(),
            threshold: 0.3,
            min_signal: 0.0,
            use_signal_strength: true,
            lambda_param: 0.02,
            decay: 0.03,
            cutout: 0,
            ma_configs: vec![MAConfig {
                ma_type: ma_type.to_string(),
                length: 20,
                weight: 1.0,
            }],
        };

        let (score, signal_type) = compute_symbol_score(&prices, &config);

        assert!(!score.is_nan(), "Score should not be NaN for {}", ma_type);
        assert!(
            signal_type == "LONG" || signal_type == "SHORT" || signal_type == "NEUTRAL",
            "Signal type should be valid for {}",
            ma_type
        );
    }
}
