use atc_serverless::constants::{DECAY_SCALE, DEFAULT_EQUITY_FLOOR, LAMBDA_SCALE};
use atc_serverless::equity::{calculate_equity, exp_growth};
use atc_serverless::signal_detection::{calculate_layer1_signal, Robustness, SignalParams};
use atc_serverless::MAType;
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
    ema_w: f64,
    hma_w: f64,
    wma_w: f64,
    dema_w: f64,
    lsma_w: f64,
    kama_w: f64,
    robustness: String,
    #[serde(rename = "La")]
    lambda_param: f64,
    #[serde(rename = "De")]
    decay: f64,
    cutout: usize,
    long_threshold: f64,
    short_threshold: f64,
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
    outputs: HashMap<String, Vec<Option<f64>>>,
    classification: String,
}

struct MAContract {
    name: &'static str,
    ma_type: MAType,
    length: usize,
    weight: f64,
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

fn classify_last(score: &Array1<f64>) -> String {
    if score.is_empty() {
        return "NEUTRAL".to_string();
    }
    let value = score[score.len() - 1];
    if !value.is_finite() {
        "NEUTRAL".to_string()
    } else if value > 0.0 {
        "LONG".to_string()
    } else if value < 0.0 {
        "SHORT".to_string()
    } else {
        "NEUTRAL".to_string()
    }
}

fn build_contracts(config: &FixtureConfig) -> Vec<MAContract> {
    vec![
        MAContract {
            name: "EMA",
            ma_type: MAType::Ema,
            length: config.ema_len,
            weight: config.ema_w,
        },
        MAContract {
            name: "HMA",
            ma_type: MAType::Hma,
            length: config.hull_len,
            weight: config.hma_w,
        },
        MAContract {
            name: "WMA",
            ma_type: MAType::Wma,
            length: config.wma_len,
            weight: config.wma_w,
        },
        MAContract {
            name: "DEMA",
            ma_type: MAType::Dema,
            length: config.dema_len,
            weight: config.dema_w,
        },
        MAContract {
            name: "LSMA",
            ma_type: MAType::Lsma,
            length: config.lsma_len,
            weight: config.lsma_w,
        },
        MAContract {
            name: "KAMA",
            ma_type: MAType::Kama,
            length: config.kama_len,
            weight: config.kama_w,
        },
    ]
}

fn calculate_roc(prices: ArrayView1<f64>) -> Array1<f64> {
    let n = prices.len();
    let mut roc = Array1::<f64>::from_elem(n, f64::NAN);
    if n == 0 {
        return roc;
    }

    roc[0] = 0.0;
    for i in 1..n {
        if prices[i - 1].is_finite() && prices[i - 1] != 0.0 && prices[i].is_finite() {
            roc[i] = (prices[i] - prices[i - 1]) / prices[i - 1];
        } else {
            roc[i] = f64::NAN;
        }
    }
    roc
}

fn compute_average_signal(
    layer1_signals: &[Array1<f64>],
    layer2_equities: &[Array1<f64>],
    long_threshold: f64,
    short_threshold: f64,
    cutout: usize,
) -> Array1<f64> {
    let n = if layer1_signals.is_empty() {
        0
    } else {
        layer1_signals[0].len()
    };
    let mut out = Array1::<f64>::zeros(n);

    for i in 0..n {
        if i < cutout {
            out[i] = 0.0;
            continue;
        }

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        let mut invalid = false;

        for (signal, weight_series) in layer1_signals.iter().zip(layer2_equities.iter()) {
            let weight = weight_series[i];
            let value = signal[i];
            if !weight.is_finite() || !value.is_finite() {
                invalid = true;
                break;
            }

            let discrete = if value > long_threshold {
                1.0
            } else if value < short_threshold {
                -1.0
            } else {
                0.0
            };

            numerator += discrete * weight;
            denominator += weight;
        }

        if !invalid && denominator > 0.0 && numerator.is_finite() && denominator.is_finite() {
            out[i] = numerator / denominator;
        } else {
            out[i] = 0.0;
        }
    }

    out
}

fn compute_outputs(input: &FixtureInput) -> Result<RunnerOutput, String> {
    let prices = parse_prices(&input.prices);
    let prices_view = ArrayView1::from(prices.as_slice());
    let n = prices_view.len();
    let robustness = Robustness::from_str(&input.config.robustness)
        .map_err(|e| format!("invalid robustness '{}': {}", input.config.robustness, e))?;

    let params = SignalParams {
        lambda_scaled: input.config.lambda_param / LAMBDA_SCALE,
        decay_scaled: input.config.decay / DECAY_SCALE,
        cutout: input.config.cutout,
        equity_floor: DEFAULT_EQUITY_FLOOR,
        robustness,
    };

    let contracts = build_contracts(&input.config);

    let roc = calculate_roc(prices_view);
    let growth = exp_growth(params.lambda_scaled, n, params.cutout);
    let mut r_adjusted = Array1::<f64>::from_elem(n, f64::NAN);
    for i in 0..n {
        r_adjusted[i] = roc[i] * growth[i];
    }

    let mut outputs: HashMap<String, Vec<Option<f64>>> = HashMap::new();
    let mut layer1_for_average: Vec<Array1<f64>> = Vec::with_capacity(contracts.len());
    let mut layer2_equities: Vec<Array1<f64>> = Vec::with_capacity(contracts.len());

    for contract in contracts {
        let (layer1_signal, _) =
            calculate_layer1_signal(prices_view, &contract.ma_type, contract.length, &params);

        let mut sig_shifted = Array1::<f64>::from_elem(n, f64::NAN);
        for i in 1..n {
            sig_shifted[i] = layer1_signal[i - 1];
        }

        let layer2 = calculate_equity(
            r_adjusted.view(),
            sig_shifted.view(),
            contract.weight,
            1.0 - params.decay_scaled,
            params.cutout,
            params.equity_floor,
        );

        outputs.insert(
            format!("{}_Signal", contract.name),
            to_optional_series(&layer1_signal),
        );

        layer1_for_average.push(layer1_signal);
        layer2_equities.push(layer2);
    }

    let average_signal = compute_average_signal(
        &layer1_for_average,
        &layer2_equities,
        input.config.long_threshold,
        input.config.short_threshold,
        input.config.cutout,
    );
    let classification = classify_last(&average_signal);

    outputs.insert("Average_Signal".to_string(), to_optional_series(&average_signal));

    Ok(RunnerOutput {
        scenario: input.scenario.clone(),
        outputs,
        classification,
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: cargo run --bin parity_runner -- <fixture.json>");
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
            eprintln!("failed to compute outputs: {}", e);
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
