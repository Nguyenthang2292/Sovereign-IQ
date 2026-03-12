use atc_serverless::{
    calculate_dema, calculate_ema, calculate_hma, calculate_wma, compute_symbol_score,
    process_batch, ATCConfig, MAConfig, SymbolData,
};
use ndarray::Array1;
use proptest::prelude::*;
use std::collections::HashMap;

// Test helper functions
fn create_test_prices(length: usize) -> Array1<f64> {
    Array1::from_iter((0..length).map(|i| i as f64))
}

fn create_test_config() -> ATCConfig {
    let mut weights = HashMap::new();
    weights.insert("1h".to_string(), 1.0);

    ATCConfig {
        weights,
        threshold: 0.3,
        min_signal: 0.0,
        use_signal_strength: true,
        lambda_param: 0.02,
        decay: 0.03,
        cutout: 0,
        equity_floor: 0.25,
        robustness: atc_serverless::Robustness::Medium,
        ma_configs: vec![MAConfig {
            ma_type: atc_serverless::MAType::Ema,
            length: 12,
            weight: 1.0,
        }],
    }
}

// Property: MA calculations output length equals input length
proptest! {
    #[test]
    fn test_ma_output_length_equals_input(
        length in 1..100usize,
        ma_type in prop_oneof![
            Just(atc_serverless::MAType::Ema),
              Just(atc_serverless::MAType::Wma),
            Just(atc_serverless::MAType::Dema),
            Just(atc_serverless::MAType::Hma),
        ]
    ) {
        let prices = create_test_prices(length);
        let result = match ma_type {
            atc_serverless::MAType::Ema => calculate_ema(prices.view(), 12),
            atc_serverless::MAType::Wma => calculate_wma(prices.view(), 12),
            atc_serverless::MAType::Dema => calculate_dema(prices.view(), 12),
            atc_serverless::MAType::Hma => calculate_hma(prices.view(), 12),
            _ => panic!("Invalid MA type"),
        };

        assert_eq!(result.len(), length);
    }
}

// Property: MA calculations produce valid outputs for sufficient input data
// Note: Different MA types require different minimum data lengths
proptest! {
    #[test]
    fn test_ma_produces_valid_outputs(
        length in 50..100usize,  // Need sufficient bars for all MA types including HMA
        ma_type in prop_oneof![
            Just(atc_serverless::MAType::Ema),
              Just(atc_serverless::MAType::Wma),
            Just(atc_serverless::MAType::Dema),
            Just(atc_serverless::MAType::Hma),
        ]
    ) {
        let prices = create_test_prices(length);
        let result = match ma_type {
            atc_serverless::MAType::Ema => calculate_ema(prices.view(), 12),
            atc_serverless::MAType::Wma => calculate_wma(prices.view(), 12),
            atc_serverless::MAType::Dema => calculate_dema(prices.view(), 12),
            atc_serverless::MAType::Hma => calculate_hma(prices.view(), 12),
            _ => panic!("Invalid MA type"),
        };

        // Check that at least some values are valid (not NaN)
        // (Initial values may be NaN during initialization period)
        let valid_count = result.iter().filter(|&&x| !x.is_nan()).count();
        assert!(valid_count > 0, "MA calculation produced all NaN values for type {} with length {}", ma_type, length);
    }
}

// Property: Signal detection output is in valid range [-1, 1]
proptest! {
    #[test]
    fn test_signal_detection_output_range(
        length in 10..100usize,
        threshold in 0.0..1.0f64,
        lambda_param in 0.0..0.1f64,
        decay in 0.0..0.1f64,
    ) {
        let prices = create_test_prices(length);
        let config = ATCConfig {
            weights: HashMap::new(),
            threshold,
            min_signal: 0.0,
            use_signal_strength: true,
            lambda_param,
            decay,
            cutout: 0,
            equity_floor: 0.25,
            robustness: atc_serverless::Robustness::Medium,
            ma_configs: vec![
                MAConfig { ma_type: atc_serverless::MAType::Ema, length: 12, weight: 1.0 },
            ],
        };

        let (score, _) = compute_symbol_score(prices.as_slice().unwrap(), &config);
        assert!(score >= -1.0 && score <= 1.0, "Signal score out of range: {}", score);
    }
}

// Property: Equity calculation produces valid results
proptest! {
    #[test]
    fn test_equity_calculation_valid(
        length in 10..100usize,
        _lambda_param in 0.0..0.1f64,
        decay in 0.0..0.1f64,
    ) {
        let prices = create_test_prices(length);
        let returns = Array1::from_iter(prices.iter().map(|&p| p / 100.0)); // Convert to returns

        let equity = atc_serverless::calculate_equity(
            returns.view(),
            Array1::from_elem(length, 1.0).view(),
            1.0,
            1.0 - decay,
            0,
            0.25,
        );

        // Check that equity calculation produces at least some valid values
        let valid_count = equity.iter().filter(|&&x| !x.is_nan()).count();
        assert!(valid_count > 0, "Equity calculation produced all NaN values");

        // Check that valid equity values are positive
        for &val in equity.iter().filter(|&&x| !x.is_nan()) {
            assert!(val > 0.0, "Equity value should be positive, got {}", val);
        }
    }
}

// Property: Batch processing results count <= input count
proptest! {
    #[test]
    fn test_batch_processing_results_count_le_input_count(
        num_symbols in 1..50usize,
    ) {
        use atc_serverless::parallelism::ParallelismConfig;

        let symbols: Vec<SymbolData> = (0..num_symbols).map(|i| SymbolData {
            symbol: format!("SYMBOL_{}", i),
            timeframes: HashMap::new(),
        }).collect();

        let config = create_test_config();
        let parallelism = ParallelismConfig::default();

        let (results, errors) = process_batch(symbols, config, Some(parallelism));
        assert!(results.len() + errors.len() <= num_symbols,
               "Results + errors > input symbols: {} + {} > {}",
               results.len(), errors.len(), num_symbols);
    }
}

// Fuzzing for edge cases
proptest! {
    #[test]
    fn fuzz_ma_calculation_edge_cases(
        length in 1..100usize,
        ma_type in prop_oneof![
            Just(atc_serverless::MAType::Ema),
              Just(atc_serverless::MAType::Wma),
            Just(atc_serverless::MAType::Dema),
            Just(atc_serverless::MAType::Hma),
        ],
        value in -1000.0..1000.0f64,
    ) {
        let prices = Array1::from_elem(length, value);
        let result = match ma_type {
            atc_serverless::MAType::Ema => calculate_ema(prices.view(), 12),
            atc_serverless::MAType::Wma => calculate_wma(prices.view(), 12),
            atc_serverless::MAType::Dema => calculate_dema(prices.view(), 12),
            atc_serverless::MAType::Hma => calculate_hma(prices.view(), 12),
            _ => panic!("Invalid MA type"),
        };

        // Check that all values are either the input value or NaN (for insufficient data)
        for &val in result.iter() {
            if length >= 12 {
                assert!((val - value).abs() < 1e-10 || val.is_nan(),
                       "Unexpected value in MA calculation: expected {}, got {}", value, val);
            } else {
                assert!(val.is_nan(), "Expected NaN for insufficient data, got {}", val);
            }
        }
    }
}

// Fuzzing for signal detection with extreme values
proptest! {
    #[test]
    fn fuzz_signal_detection_extreme_values(
        length in 10..100usize,
        extreme_value in -1e6..1e6f64,
    ) {
        let prices = Array1::from_elem(length, extreme_value);
        let config = create_test_config();

        let (score, _) = compute_symbol_score(prices.as_slice().unwrap(), &config);
        assert!(score >= -1.0 && score <= 1.0, "Signal score out of range with extreme values");
    }
}

// Fuzzing for zero-length arrays
proptest! {
    #[test]
    fn fuzz_zero_length_arrays(
        ma_type in prop_oneof![
            Just(atc_serverless::MAType::Ema),
              Just(atc_serverless::MAType::Wma),
            Just(atc_serverless::MAType::Dema),
            Just(atc_serverless::MAType::Hma),
        ]
    ) {
        let prices = Array1::from_elem(0, 0.0);
        let result = match ma_type {
            atc_serverless::MAType::Ema => calculate_ema(prices.view(), 12),
            atc_serverless::MAType::Wma => calculate_wma(prices.view(), 12),
            atc_serverless::MAType::Dema => calculate_dema(prices.view(), 12),
            atc_serverless::MAType::Hma => calculate_hma(prices.view(), 12),
            _ => panic!("Invalid MA type"),
        };

        assert!(result.iter().all(|&x| x.is_nan()), "Expected all NaN for zero-length input");
    }
}

// Fuzzing for all-NaN inputs
proptest! {
    #[test]
    fn fuzz_all_nan_inputs(
        length in 1..100usize,
        ma_type in prop_oneof![
            Just(atc_serverless::MAType::Ema),
              Just(atc_serverless::MAType::Wma),
            Just(atc_serverless::MAType::Dema),
            Just(atc_serverless::MAType::Hma),
        ]
    ) {
        let prices = Array1::from_iter(std::iter::repeat(f64::NAN).take(length));
        let result = match ma_type {
            atc_serverless::MAType::Ema => calculate_ema(prices.view(), 12),
            atc_serverless::MAType::Wma => calculate_wma(prices.view(), 12),
            atc_serverless::MAType::Dema => calculate_dema(prices.view(), 12),
            atc_serverless::MAType::Hma => calculate_hma(prices.view(), 12),
            _ => panic!("Invalid MA type"),
        };

        assert!(result.iter().all(|&x| x.is_nan()), "Expected all NaN for all-NaN input");
    }
}
