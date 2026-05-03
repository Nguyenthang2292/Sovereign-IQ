use atc_serverless::{
    constants::{OHLCV_FIELDS_PER_BAR, WORKING_BUFFERS_PER_BAR},
    get_memory_usage_mb, parallelism::ParallelismConfig, process_batch, validate_batch_request,
    BatchRequest, ScanResult, SCHEMA_VERSION,
};
use lambda_runtime::{Error, LambdaEvent};
use rayon::current_num_threads;
use serde_json::json;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing::{error, info, warn};

// 1769 MB is the AWS Lambda memory point that maps to roughly 1 vCPU.
const DEFAULT_LAMBDA_MEMORY_MB: u64 = 1769;
const MEMORY_WARNING_RATIO: f64 = 0.68;
const MEMORY_CRITICAL_RATIO: f64 = 0.85;

/// Assumed total bars per symbol for the rough memory pre-check.
///
/// Based on the default serverless OHLCV limit (220 bars) × up to 3 timeframes.
/// This is intentionally conservative so the pre-validation guard stays an overestimate.
const ASSUMED_TOTAL_BARS_PER_SYMBOL: usize = 660; // 220 bars × 3 timeframes

#[derive(Debug, Clone)]
struct ProcessingMetrics {
    peak_memory_mb: u64,
    memory_delta_mb: u64,
    symbols_per_second: f64,
    thread_count: u64,
    error_rate_percent: f64,
}

fn current_time_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn lambda_function_name() -> String {
    std::env::var("AWS_LAMBDA_FUNCTION_NAME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn memory_thresholds_mb() -> (u64, u64) {
    let configured_memory_mb = std::env::var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_LAMBDA_MEMORY_MB);

    let warning_threshold = ((configured_memory_mb as f64) * MEMORY_WARNING_RATIO).round() as u64;
    let mut critical_threshold =
        ((configured_memory_mb as f64) * MEMORY_CRITICAL_RATIO).round() as u64;

    if critical_threshold <= warning_threshold {
        critical_threshold = warning_threshold.saturating_add(1);
    }

    (warning_threshold.max(1), critical_threshold.max(2))
}

/// Rough estimate of memory usage for batch size validation (before parsing data).
///
/// Uses the same `OHLCV_FIELDS_PER_BAR` and `WORKING_BUFFERS_PER_BAR` constants as
/// `aggregation::estimate_batch_memory_mb`, so adding a working buffer there will
/// automatically raise this estimate too — preventing silent underestimates.
///
/// For the exact data-driven estimate see `aggregation::estimate_batch_memory_mb`.
fn estimate_batch_memory_mb_rough(symbol_count: usize) -> u64 {
    let bytes_per_symbol =
        (OHLCV_FIELDS_PER_BAR + WORKING_BUFFERS_PER_BAR) * ASSUMED_TOTAL_BARS_PER_SYMBOL * 8;
    let kb_per_symbol = (bytes_per_symbol + 1023) / 1024; // ceiling KB
    ((symbol_count * kb_per_symbol + 1023) / 1024) as u64 // ceiling MB
}

fn calculate_error_rate_percent(error_count: usize, symbol_count: usize) -> f64 {
    if symbol_count == 0 {
        0.0
    } else {
        (error_count as f64 / symbol_count as f64) * 100.0
    }
}

fn calculate_symbols_per_second(symbol_count: usize, processing_duration_ms: u64) -> f64 {
    if symbol_count == 0 {
        0.0
    } else {
        let effective_ms = processing_duration_ms.max(1);
        (symbol_count as f64 / effective_ms as f64) * 1000.0
    }
}

fn build_cloudwatch_metrics_log(
    batch_id: &str,
    timestamp_ms: u64,
    function_name: &str,
    metrics: &ProcessingMetrics,
) -> String {
    json!({
        "_aws": {
            "Timestamp": timestamp_ms,
            "CloudWatchMetrics": [{
                "Namespace": "ATC/Serverless",
                "Dimensions": [["FunctionName"]],
                "Metrics": [
                    { "Name": "MemoryUsageMB", "Unit": "Megabytes" },
                    { "Name": "MemoryDeltaMB", "Unit": "Megabytes" },
                    { "Name": "SymbolsPerSecond", "Unit": "Count/Second" },
                    { "Name": "ThreadCount", "Unit": "Count" },
                    { "Name": "ErrorRate", "Unit": "Percent" }
                ]
            }]
        },
        "batch_id": batch_id,
        "FunctionName": function_name,
        "MemoryUsageMB": metrics.peak_memory_mb,
        "MemoryDeltaMB": metrics.memory_delta_mb,
        "SymbolsPerSecond": metrics.symbols_per_second,
        "ThreadCount": metrics.thread_count,
        "ErrorRate": metrics.error_rate_percent,
    })
    .to_string()
}

fn emit_cloudwatch_metrics(batch_id: &str, function_name: &str, metrics: &ProcessingMetrics) {
    // Emit a raw EMF JSON line so CloudWatch extracts real custom metrics.
    println!(
        "{}",
        build_cloudwatch_metrics_log(batch_id, current_time_millis(), function_name, metrics)
    );
}

/// Handle incoming Lambda request
///
/// Processes a batch of symbols and returns the scan result directly.
/// Includes comprehensive logging, error handling, and memory monitoring.
///
/// # Arguments
/// * `event` - Lambda event containing the batch request
///
/// # Returns
/// The completed batch scan result
pub async fn handle_request(event: LambdaEvent<BatchRequest>) -> Result<ScanResult, Error> {
    let request = event.payload;
    let batch_id = request.batch_id.clone();
    let symbol_count = request.symbols.len();
    let apply_strategy_shift = request.apply_strategy_shift.unwrap_or(false);
    let function_name = lambda_function_name();
    let (memory_warning_threshold_mb, memory_critical_threshold_mb) = memory_thresholds_mb();
    let start_time = Instant::now();

    if let Err(validation_error) = validate_batch_request(&request) {
        error!(
            batch_id = %batch_id,
            validation_error = %validation_error,
            "Input validation failed"
        );
        return Err(format!("Input validation failed: {}", validation_error).into());
    }

    // Memory monitoring: initial state
    let initial_memory_mb = get_memory_usage_mb();
    let estimated_memory_mb = estimate_batch_memory_mb_rough(symbol_count);

    // Log batch processing start
    info!(
        batch_id = %batch_id,
        symbol_count = symbol_count,
        initial_memory_mb = initial_memory_mb,
        estimated_memory_mb = estimated_memory_mb,
        "Processing batch start"
    );

    // Log configuration summary
    info!(
        batch_id = %batch_id,
        threshold = request.config.threshold,
        ma_types = request.config.ma_configs.len(),
        timeframes = request.config.weights.len(),
        apply_strategy_shift = apply_strategy_shift,
        "Configuration"
    );

    // Create parallelism config based on batch size (optimal for Lambda)
    let parallelism_config = ParallelismConfig::default().optimal_for_batch_size(symbol_count);
    let thread_count = parallelism_config
        .num_threads
        .unwrap_or_else(current_num_threads);

    info!(
        batch_id = %batch_id,
        parallelism_threads = thread_count,
        parallelism_chunk_size = parallelism_config.chunk_size,
        "Parallelism configuration"
    );

    // Process batch with error recovery (CPU intensive, runs on thread pool via rayon)
    let processing_start = Instant::now();
    let (mut results, errors) =
        process_batch(request.symbols, request.config, Some(parallelism_config));
    let processing_duration_ms = processing_start.elapsed().as_millis() as u64;

    let success_count = results.len();
    let error_count = errors.len();

    if apply_strategy_shift {
        warn!(
            batch_id = %batch_id,
            "apply_strategy_shift was requested but snapshot response has no historical series; returning raw-only signal fields"
        );
    }

    for result in &mut results {
        if result.average_signal_raw.is_none() {
            result.average_signal_raw = Some(result.score);
        }
    }

    // Memory monitoring: peak usage after processing
    let peak_memory_mb = get_memory_usage_mb();
    let memory_delta_mb = peak_memory_mb.saturating_sub(initial_memory_mb);

    // Calculate throughput
    let symbols_per_second = calculate_symbols_per_second(symbol_count, processing_duration_ms);

    let error_rate_percent = calculate_error_rate_percent(error_count, symbol_count);

    let metrics = ProcessingMetrics {
        peak_memory_mb,
        memory_delta_mb,
        symbols_per_second,
        thread_count: thread_count as u64,
        error_rate_percent,
    };

    // Log processing metrics with memory info
    info!(
        batch_id = %batch_id,
        processing_duration_ms = processing_duration_ms,
        symbols_per_second = symbols_per_second,
        success_count = success_count,
        error_count = error_count,
        peak_memory_mb = peak_memory_mb,
        memory_delta_mb = memory_delta_mb,
        "Processing completed"
    );

    // Memory threshold warnings
    if peak_memory_mb >= memory_critical_threshold_mb {
        error!(
            batch_id = %batch_id,
            peak_memory_mb = peak_memory_mb,
            threshold_mb = memory_critical_threshold_mb,
            "CRITICAL: Memory usage exceeds critical threshold"
        );
    } else if peak_memory_mb >= memory_warning_threshold_mb {
        warn!(
            batch_id = %batch_id,
            peak_memory_mb = peak_memory_mb,
            threshold_mb = memory_warning_threshold_mb,
            "WARNING: Memory usage exceeds warning threshold"
        );
    }

    emit_cloudwatch_metrics(&batch_id, &function_name, &metrics);

    // Log error summary if any symbols failed
    if !errors.is_empty() {
        warn!(
            batch_id = %batch_id,
            error_count = error_count,
            total_symbols = symbol_count,
            error_rate_percent = error_rate_percent,
            "Batch completed with errors"
        );

        for error in &errors {
            warn!(
                batch_id = %batch_id,
                symbol = %error.symbol,
                error = %error.error,
                "Symbol processing failed"
            );
        }
    }

    let scan_result = ScanResult {
        batch_id: batch_id.clone(),
        schema_version: SCHEMA_VERSION.to_string(),
        results,
        errors,
        success_count,
        error_count,
    };

    // Log total duration with final memory state
    let total_duration_ms = start_time.elapsed().as_millis() as u64;
    let final_memory_mb = get_memory_usage_mb();

    info!(
        batch_id = %batch_id,
        total_duration_ms = total_duration_ms,
        success_count = success_count,
        error_count = error_count,
        final_memory_mb = final_memory_mb,
        "Batch processing completed successfully"
    );

    Ok(scan_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use atc_serverless::constants::{MAX_BATCH_SIZE, MIN_DATA_LENGTH};
    use atc_serverless::{ATCConfig, BatchRequest, OHLCVData, SymbolData};
    use lambda_runtime::Context;
    use serde_json::Value;
    use std::collections::HashMap;

    fn valid_batch_request() -> BatchRequest {
        let mut weights = HashMap::new();
        weights.insert("1h".to_string(), 1.0);

        BatchRequest {
            batch_id: "test-batch".to_string(),
            version: Some("1.0.0".to_string()),
            symbols: vec![test_symbol("BTCUSDT", 64)],
            apply_strategy_shift: None,
            config: ATCConfig {
                weights,
                threshold: 0.3,
                min_signal: 0.0,
                use_signal_strength: false,
                lambda_param: 0.02,
                decay: 0.03,
                cutout: 0,
                equity_floor: 0.25,
                robustness: atc_serverless::Robustness::Medium,
                ma_configs: vec![atc_serverless::MAConfig {
                    ma_type: atc_serverless::MAType::Ema,
                    length: 12,
                    weight: 1.0,
                }],
            },
        }
    }

    fn test_symbol(symbol: &str, bars: usize) -> SymbolData {
        let close = (0..bars)
            .map(|index| 100.0 + index as f64 * 0.1)
            .collect::<Vec<_>>();

        let ohlcv = OHLCVData {
            timestamp: (1..=bars)
                .map(|value| value as i64)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            open: close
                .iter()
                .map(|value| value - 0.1)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            high: close
                .iter()
                .map(|value| value + 0.5)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            low: close
                .iter()
                .map(|value| value - 0.5)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            close: close.into_boxed_slice(),
            volume: vec![1000.0; bars].into_boxed_slice(),
        };

        let mut timeframes = HashMap::new();
        timeframes.insert("1h".to_string(), ohlcv);

        SymbolData {
            symbol: symbol.to_string(),
            timeframes,
        }
    }

    fn invalid_batch_request() -> BatchRequest {
        BatchRequest {
            batch_id: "test-batch".to_string(),
            version: None,
            symbols: vec![],
            apply_strategy_shift: None,
            config: ATCConfig {
                weights: HashMap::new(),
                threshold: 1.5,
                min_signal: 0.0,
                use_signal_strength: false,
                lambda_param: 0.02,
                decay: 0.03,
                cutout: 0,
                equity_floor: 0.25,
                robustness: atc_serverless::Robustness::Medium,
                ma_configs: vec![],
            },
        }
    }

    #[tokio::test]
    async fn test_handler_success_path() {
        let batch_req = valid_batch_request();
        let event = LambdaEvent::new(batch_req, Context::default());

        let result = handle_request(event).await;
        assert!(result.is_ok());

        let scan_result = result.unwrap();
        assert_eq!(scan_result.batch_id, "test-batch");
        assert_eq!(scan_result.success_count, 1);
        assert_eq!(scan_result.error_count, 0);
        assert_eq!(scan_result.results.len(), 1);
        let first = &scan_result.results[0];
        assert_eq!(first.average_signal_raw, Some(first.score));
        assert_eq!(first.average_signal_exec, None);
    }

    #[tokio::test]
    async fn test_handler_apply_strategy_shift_keeps_raw_snapshot_contract() {
        let mut batch_req = valid_batch_request();
        batch_req.apply_strategy_shift = Some(true);
        let event = LambdaEvent::new(batch_req, Context::default());

        let result = handle_request(event).await;
        assert!(result.is_ok());

        let scan_result = result.unwrap();
        assert_eq!(scan_result.success_count, 1);
        assert_eq!(scan_result.error_count, 0);
        assert_eq!(scan_result.results.len(), 1);

        let first = &scan_result.results[0];
        assert_eq!(first.average_signal_raw, Some(first.score));
        assert_eq!(
            first.average_signal_exec, None,
            "Snapshot API must not synthesize execution shift without history",
        );
    }

    #[tokio::test]
    async fn test_handler_validation_error() {
        let batch_req = invalid_batch_request();
        let event = LambdaEvent::new(batch_req, Context::default());

        let result = handle_request(event).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handler_rejects_batch_exceeding_max_size_before_processing() {
        let mut batch_req = valid_batch_request();
        batch_req.symbols = (0..(MAX_BATCH_SIZE + 1))
            .map(|index| test_symbol(&format!("SYM{}", index), 64))
            .collect();

        let event = LambdaEvent::new(batch_req, Context::default());

        let result = handle_request(event).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handler_rejects_symbol_history_below_min_length_before_processing() {
        let mut batch_req = valid_batch_request();
        let short_bars = MIN_DATA_LENGTH.saturating_sub(1).max(1);
        batch_req.symbols = vec![test_symbol("TOO_SHORT", short_bars)];

        let event = LambdaEvent::new(batch_req, Context::default());

        let result = handle_request(event).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_build_cloudwatch_metrics_log_uses_emf_namespace() {
        let metrics = ProcessingMetrics {
            peak_memory_mb: 640,
            memory_delta_mb: 128,
            symbols_per_second: 2048.0,
            thread_count: 8,
            error_rate_percent: 12.5,
        };

        let payload = build_cloudwatch_metrics_log(
            "batch-123",
            1_700_000_000_000,
            "atc-serverless-test",
            &metrics,
        );
        let json_payload: Value = serde_json::from_str(&payload).expect("valid EMF payload");

        assert_eq!(
            json_payload["_aws"]["CloudWatchMetrics"][0]["Namespace"],
            "ATC/Serverless"
        );
        assert_eq!(
            json_payload["_aws"]["CloudWatchMetrics"][0]["Dimensions"][0],
            Value::Array(vec![Value::String("FunctionName".to_string())])
        );
        assert_eq!(json_payload["batch_id"], "batch-123");
        assert_eq!(json_payload["FunctionName"], "atc-serverless-test");
        assert_eq!(json_payload["MemoryUsageMB"], 640);
        assert_eq!(json_payload["SymbolsPerSecond"], 2048.0);
        assert_eq!(json_payload["ErrorRate"], 12.5);
    }

    #[test]
    fn test_calculate_error_rate_percent() {
        assert_eq!(calculate_error_rate_percent(0, 0), 0.0);
        assert_eq!(calculate_error_rate_percent(0, 10), 0.0);
        assert_eq!(calculate_error_rate_percent(1, 4), 25.0);
        assert_eq!(calculate_error_rate_percent(3, 6), 50.0);
    }

    /// Ensures the rough pre-validation estimator never underestimates the
    /// accurate post-parse estimator for the same workload.
    ///
    /// The test reproduces the accurate formula using the same public constants so
    /// that any divergence (e.g. a new working buffer added to `WORKING_BUFFERS_PER_BAR`
    /// or a change to `ASSUMED_TOTAL_BARS_PER_SYMBOL`) is detected at compile/test time
    /// rather than silently in production.
    #[test]
    fn test_rough_estimate_never_underestimates_accurate_estimate() {
        let bars = ASSUMED_TOTAL_BARS_PER_SYMBOL;

        // Reproduce the accurate estimator formula from aggregation.rs:
        // total_bytes = bars * (OHLCV_FIELDS_PER_BAR + WORKING_BUFFERS_PER_BAR) * 8
        // accurate_mb = total_bytes / (1024 * 1024)  (floor division)
        let total_bytes =
            bars * (OHLCV_FIELDS_PER_BAR + WORKING_BUFFERS_PER_BAR) * 8;
        let accurate_mb = (total_bytes / (1024 * 1024)) as u64;

        let rough_mb = estimate_batch_memory_mb_rough(1);

        assert!(
            rough_mb >= accurate_mb,
            "Rough estimate ({rough_mb} MB) must be >= accurate estimate ({accurate_mb} MB). \
             Update ASSUMED_TOTAL_BARS_PER_SYMBOL or the field constants if this fails."
        );
    }

    #[test]
    fn test_calculate_symbols_per_second_uses_one_ms_floor() {
        assert_eq!(calculate_symbols_per_second(0, 0), 0.0);
        assert_eq!(calculate_symbols_per_second(5, 0), 5000.0);
        assert_eq!(calculate_symbols_per_second(5, 1), 5000.0);
        assert_eq!(calculate_symbols_per_second(5, 10), 500.0);
    }
}
