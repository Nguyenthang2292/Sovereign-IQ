use atc_serverless::signal_detection::Robustness;
use atc_serverless::{
    calculate_dema, calculate_diflen, calculate_ema, calculate_hma, calculate_kama, calculate_lsma,
    calculate_wma, MAType,
};
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::process;
use std::str::FromStr;

#[derive(Debug, Deserialize)]
struct FixtureConfig {
    ema_len: usize,
    hull_len: usize,
    wma_len: usize,
    dema_len: usize,
    lsma_len: usize,
    kama_len: usize,
    robustness: String,
}

#[derive(Debug, Deserialize)]
struct FixtureInput {
    scenario: String,
    prices: Vec<Option<f64>>,
    config: FixtureConfig,
}

#[derive(Debug, Serialize)]
struct RunnerOutput {
    scenario: String,
    ma_outputs: HashMap<String, Vec<Vec<Option<f64>>>>,
}

struct MAContract {
    name: &'static str,
    ma_type: MAType,
    length: usize,
}

fn parse_prices(raw: &[Option<f64>]) -> Vec<f64> {
    raw.iter()
        .map(|value| match value {
            Some(v) => *v,
            None => f64::NAN,
        })
        .collect()
}

fn to_optional_series(series: &Array1<f64>) -> Vec<Option<f64>> {
    series
        .iter()
        .map(|value| if value.is_finite() { Some(*value) } else { None })
        .collect()
}

fn build_contracts(config: &FixtureConfig) -> Vec<MAContract> {
    vec![
        MAContract {
            name: "EMA",
            ma_type: MAType::Ema,
            length: config.ema_len,
        },
        MAContract {
            name: "HMA",
            ma_type: MAType::Hma,
            length: config.hull_len,
        },
        MAContract {
            name: "WMA",
            ma_type: MAType::Wma,
            length: config.wma_len,
        },
        MAContract {
            name: "DEMA",
            ma_type: MAType::Dema,
            length: config.dema_len,
        },
        MAContract {
            name: "LSMA",
            ma_type: MAType::Lsma,
            length: config.lsma_len,
        },
        MAContract {
            name: "KAMA",
            ma_type: MAType::Kama,
            length: config.kama_len,
        },
    ]
}

fn compute_ma(prices: ArrayView1<f64>, ma_type: &MAType, length: usize) -> Array1<f64> {
    match ma_type {
        MAType::Ema => calculate_ema(prices, length),
        MAType::Hma => calculate_hma(prices, length),
        MAType::Wma => calculate_wma(prices, length),
        MAType::Dema => calculate_dema(prices, length),
        MAType::Lsma => calculate_lsma(prices, length),
        MAType::Kama => calculate_kama(prices, length),
    }
}

fn compute_outputs(input: &FixtureInput) -> Result<RunnerOutput, String> {
    let robustness = Robustness::from_str(&input.config.robustness)
        .map_err(|e| format!("invalid robustness '{}': {}", input.config.robustness, e))?;
    let prices = parse_prices(&input.prices);
    let prices_view = ArrayView1::from(prices.as_slice());

    let mut ma_outputs: HashMap<String, Vec<Vec<Option<f64>>>> = HashMap::new();
    for contract in build_contracts(&input.config) {
        let dif = calculate_diflen(contract.length, robustness).ok_or_else(|| {
            format!(
                "calculate_diflen failed for {} length {}",
                contract.name, contract.length
            )
        })?;

        let mut lengths: Vec<usize> = Vec::with_capacity(9);
        lengths.push(contract.length);
        lengths.extend(dif);

        let mut series_bundle: Vec<Vec<Option<f64>>> = Vec::with_capacity(9);
        for len in lengths {
            let series = compute_ma(prices_view, &contract.ma_type, len);
            series_bundle.push(to_optional_series(&series));
        }
        ma_outputs.insert(contract.name.to_string(), series_bundle);
    }

    Ok(RunnerOutput {
        scenario: input.scenario.clone(),
        ma_outputs,
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: cargo run --bin ma_parity_runner -- <fixture.json>");
        process::exit(2);
    }

    let fixture_path = &args[1];
    let content = match fs::read_to_string(fixture_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to read fixture '{}': {}", fixture_path, e);
            process::exit(2);
        }
    };

    let input: FixtureInput = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to parse fixture '{}': {}", fixture_path, e);
            process::exit(2);
        }
    };

    let output = match compute_outputs(&input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to compute MA outputs: {}", e);
            process::exit(1);
        }
    };

    match serde_json::to_string(&output) {
        Ok(json) => println!("{}", json),
        Err(e) => {
            eprintln!("failed to serialize output: {}", e);
            process::exit(1);
        }
    }
}
