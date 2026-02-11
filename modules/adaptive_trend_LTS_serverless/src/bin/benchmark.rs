use atc_serverless::{process_batch, ATCConfig, SymbolData};
use std::io::{self, Read};
use std::time::Instant;

#[derive(serde::Deserialize)]
struct BenchmarkInput {
    symbols: Vec<SymbolData>,
    config: ATCConfig,
}

fn main() {
    // Read JSON from stdin instead of command line
    let mut json_input = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut json_input) {
        eprintln!("Error reading from stdin: {}", e);
        std::process::exit(1);
    }

    let input: BenchmarkInput = match serde_json::from_str(&json_input) {
        Ok(req) => req,
        Err(e) => {
            eprintln!("Error parsing JSON: {}", e);
            std::process::exit(1);
        }
    };

    let start = Instant::now();
    let (results, errors) = process_batch(input.symbols, input.config);
    let duration = start.elapsed();

    let output = serde_json::json!({
        "duration_ms": duration.as_millis(),
        "duration_micros": duration.as_micros(),
        "success_count": results.len(),
        "error_count": errors.len(),
        "results": results,
        "errors": errors,
    });

    println!("{}", output.to_string());
}
