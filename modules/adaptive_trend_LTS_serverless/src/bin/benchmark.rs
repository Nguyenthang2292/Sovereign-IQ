use atc_serverless::{
    parallelism::ParallelismConfig, process_batch, validate_batch_request, ATCConfig, BatchRequest,
    SymbolData, SCHEMA_VERSION,
};
use std::io::{self, Read};
use std::time::Instant;

#[derive(serde::Deserialize)]
struct BenchmarkInput {
    symbols: Vec<SymbolData>,
    config: ATCConfig,
    #[serde(default)]
    parallelism: Option<ParallelismInput>,
}

#[derive(serde::Deserialize, Default)]
struct ParallelismInput {
    #[serde(default)]
    threads: Option<usize>,
    #[serde(default)]
    chunk_size: Option<usize>,
    #[serde(default)]
    min_len: Option<usize>,
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

    let parallelism_config = input.parallelism.map(|p| {
        let mut config = ParallelismConfig::default();
        if let Some(threads) = p.threads {
            config = config.with_threads(threads);
        }
        if let Some(chunk_size) = p.chunk_size {
            config = config.with_chunk_size(chunk_size);
        }
        if let Some(min_len) = p.min_len {
            config = config.with_min_len(min_len);
        }
        config
    });

    let request = BatchRequest {
        batch_id: "benchmark-batch".to_string(),
        version: Some(SCHEMA_VERSION.to_string()),
        symbols: input.symbols,
        config: input.config,
    };

    if let Err(validation_error) = validate_batch_request(&request) {
        eprintln!("Input validation failed: {}", validation_error);
        std::process::exit(1);
    }

    let start = Instant::now();
    let (results, errors) = process_batch(request.symbols, request.config, parallelism_config);
    let duration = start.elapsed();

    let output = serde_json::json!({
        "duration_ms": duration.as_millis(),
        "duration_micros": duration.as_micros(),
        "success_count": results.len(),
        "error_count": errors.len(),
        "results": results,
        "errors": errors,
    });

    println!("{}", output);
}
