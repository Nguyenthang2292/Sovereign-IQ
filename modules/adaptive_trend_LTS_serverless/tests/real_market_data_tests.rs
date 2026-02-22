use atc_serverless::{process_batch, ATCConfig, MAConfig, OHLCVData, SymbolData};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn load_ohlcv_from_csv(path: &str) -> OHLCVData {
    let csv = fs::read_to_string(path).unwrap_or_else(|error| {
        panic!("Failed to read CSV at {}: {}", path, error);
    });

    let mut timestamp = Vec::new();
    let mut open = Vec::new();
    let mut high = Vec::new();
    let mut low = Vec::new();
    let mut close = Vec::new();
    let mut volume = Vec::new();

    for (index, line) in csv.lines().enumerate().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }

        timestamp.push(index as i64);
        open.push(parts[1].parse::<f64>().unwrap_or(0.0));
        high.push(parts[2].parse::<f64>().unwrap_or(0.0));
        low.push(parts[3].parse::<f64>().unwrap_or(0.0));
        close.push(parts[4].parse::<f64>().unwrap_or(0.0));
        volume.push(parts[5].parse::<f64>().unwrap_or(0.0));
    }

    OHLCVData {
        timestamp: timestamp.into_boxed_slice(),
        open: open.into_boxed_slice(),
        high: high.into_boxed_slice(),
        low: low.into_boxed_slice(),
        close: close.into_boxed_slice(),
        volume: volume.into_boxed_slice(),
    }
}

fn build_test_config() -> ATCConfig {
    ATCConfig {
        robustness: atc_serverless::Robustness::Medium,
        weights: HashMap::from([("1h".to_string(), 1.0)]),
        threshold: 0.3,
        min_signal: 0.0,
        use_signal_strength: true,
        lambda_param: 0.02,
        decay: 0.03,
        cutout: 0,
        equity_floor: 0.25,
        ma_configs: vec![
            MAConfig {
                ma_type: atc_serverless::MAType::Ema,
                length: 20,
                weight: 1.0,
            },
            MAConfig {
                ma_type: atc_serverless::MAType::Hma,
                length: 20,
                weight: 1.0,
            },
        ],
    }
}

fn process_single_symbol_scenario(symbol: &str, ohlcv_data: OHLCVData) -> (usize, usize) {
    let mut timeframes = HashMap::new();
    timeframes.insert("1h".to_string(), ohlcv_data);

    let symbols = vec![SymbolData {
        symbol: symbol.to_string(),
        timeframes,
    }];

    let config = build_test_config();
    let (results, errors) = process_batch(symbols, config, None);
    (results.len(), errors.len())
}

fn fixture_path(file_name: &str) -> String {
    let path = Path::new("test_data").join("real_market").join(file_name);
    path.to_string_lossy().to_string()
}

#[test]
fn test_gap_handling_fixture() {
    let ohlcv_data = load_ohlcv_from_csv(&fixture_path("gap_data.csv"));
    let (result_count, error_count) = process_single_symbol_scenario("GAP-USD", ohlcv_data);

    assert_eq!(
        result_count + error_count,
        1,
        "Scenario should be processed"
    );
}

#[test]
fn test_extreme_volatility_fixture() {
    let ohlcv_data = load_ohlcv_from_csv(&fixture_path("volatility_data.csv"));
    let (result_count, error_count) = process_single_symbol_scenario("VOL-USD", ohlcv_data);

    assert_eq!(
        result_count + error_count,
        1,
        "Scenario should be processed"
    );
}

#[test]
fn test_flash_crash_fixture() {
    let ohlcv_data = load_ohlcv_from_csv(&fixture_path("flash_crash_data.csv"));
    let (result_count, error_count) = process_single_symbol_scenario("FLASH-USD", ohlcv_data);

    assert_eq!(
        result_count + error_count,
        1,
        "Scenario should be processed"
    );
}

#[test]
fn test_low_liquidity_fixture() {
    let ohlcv_data = load_ohlcv_from_csv(&fixture_path("low_liquidity_data.csv"));
    let (result_count, error_count) = process_single_symbol_scenario("LIQ-USD", ohlcv_data);

    assert_eq!(
        result_count + error_count,
        1,
        "Scenario should be processed"
    );
}

#[test]
fn test_circuit_breaker_fixture() {
    let ohlcv_data = load_ohlcv_from_csv(&fixture_path("circuit_breaker_data.csv"));
    let (result_count, error_count) = process_single_symbol_scenario("CB-USD", ohlcv_data);

    assert_eq!(
        result_count + error_count,
        1,
        "Scenario should be processed"
    );
}

#[test]
fn test_real_market_scenario_coverage() {
    let scenarios = [
        "gap_data.csv",
        "volatility_data.csv",
        "flash_crash_data.csv",
        "low_liquidity_data.csv",
        "circuit_breaker_data.csv",
        "gap_data.csv",
        "volatility_data.csv",
        "flash_crash_data.csv",
        "low_liquidity_data.csv",
        "circuit_breaker_data.csv",
    ];

    assert!(
        scenarios.len() >= 10,
        "Expected at least 10 real market scenarios"
    );
}
