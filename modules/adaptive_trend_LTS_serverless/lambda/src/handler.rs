use lambda_runtime::{Error, LambdaEvent};
use atc_serverless::{
    BatchRequest, process_batch, ScanResult, parallelism::ParallelismConfig,
    validate_batch_request, get_memory_usage_mb,
};
use crate::sqs::SqsClient;
use rayon::current_num_threads;
use tracing::{info, warn, error};
use std::time::Instant;
use async_trait::async_trait;

// Memory thresholds for Lambda monitoring
const MEMORY_WARNING_THRESHOLD_MB: u64 = 512;  // Warn at 512MB
const MEMORY_CRITICAL_THRESHOLD_MB: u64 = 768; // Critical at 768MB (for 1GB Lambda)

#[async_trait]
pub trait SqsSender: Send + Sync {
    async fn send_scan_result(&self, result: &ScanResult) -> Result<(), Error>;
}

#[async_trait]
impl SqsSender for SqsClient {
    async fn send_scan_result(&self, result: &ScanResult) -> Result<(), Error> {
        SqsClient::send_scan_result(self, result)
            .await
            .map_err(|error| Box::new(error) as Error)
    }
}

/// Rough estimate of memory usage for batch size validation (before parsing data)
/// This is a fast heuristic used for pre-validation checks.
/// For actual data-driven estimates, see aggregation::estimate_batch_memory_mb()
fn estimate_batch_memory_mb_rough(symbol_count: usize) -> u64 {
    // Rough estimate: ~55KB per symbol
    // Use ceiling division: (symbol_count * 55KB + 1023) / 1024 = MB
    ((symbol_count * 55 + 1023) / 1024) as u64
}

/// Handle incoming Lambda request
/// 
/// Processes a batch of symbols and sends results to SQS.
/// Includes comprehensive logging, error handling, and memory monitoring.
///
/// # Arguments
/// * `event` - Lambda event containing the batch request
/// * `sqs_client` - SQS client for sending results
///
/// # Returns
/// Result indicating success or failure
pub async fn handle_request(
    event: LambdaEvent<BatchRequest>,
    sqs_client: &dyn SqsSender,
) -> Result<(), Error> {
    let request = event.payload;
    let batch_id = request.batch_id.clone();
    let symbol_count = request.symbols.len();
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
        "Configuration"
    );

    // Create parallelism config based on batch size (optimal for Lambda)
    let parallelism_config = ParallelismConfig::default().optimal_for_batch_size(symbol_count);
    let thread_count = parallelism_config.num_threads.unwrap_or_else(|| {
        current_num_threads()
    });
    
    info!(
        batch_id = %batch_id,
        parallelism_threads = thread_count,
        parallelism_chunk_size = parallelism_config.chunk_size,
        "Parallelism configuration"
    );

    // Process batch with error recovery (CPU intensive, runs on thread pool via rayon)
    let processing_start = Instant::now();
    let (results, errors) = process_batch(request.symbols, request.config, Some(parallelism_config));
    let processing_duration_ms = processing_start.elapsed().as_millis() as u64;
    
    let success_count = results.len();
    let error_count = errors.len();

    // Memory monitoring: peak usage after processing
    let peak_memory_mb = get_memory_usage_mb();
    let memory_delta_mb = if peak_memory_mb > initial_memory_mb {
        peak_memory_mb - initial_memory_mb
    } else {
        0
    };

    // Calculate throughput
    let symbols_per_second = if processing_duration_ms > 0 {
        (symbol_count as f64 / processing_duration_ms as f64) * 1000.0
    } else {
        0.0
    };

    let error_rate = if symbol_count > 0 {
        error_count as f64 / symbol_count as f64
    } else {
        0.0
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
    if peak_memory_mb >= MEMORY_CRITICAL_THRESHOLD_MB {
        error!(
            batch_id = %batch_id,
            peak_memory_mb = peak_memory_mb,
            threshold_mb = MEMORY_CRITICAL_THRESHOLD_MB,
            "CRITICAL: Memory usage exceeds critical threshold"
        );
    } else if peak_memory_mb >= MEMORY_WARNING_THRESHOLD_MB {
        warn!(
            batch_id = %batch_id,
            peak_memory_mb = peak_memory_mb,
            threshold_mb = MEMORY_WARNING_THRESHOLD_MB,
            "WARNING: Memory usage exceeds warning threshold"
        );
    }

    // Log CloudWatch custom metrics (structured logging for CloudWatch Insights)
    info!(
        metric_name = "MemoryUsageMB",
        metric_value = peak_memory_mb,
        metric_unit = "Megabytes",
        batch_id = %batch_id,
        "CloudWatch Metric"
    );

    info!(
        metric_name = "MemoryDeltaMB",
        metric_value = memory_delta_mb,
        metric_unit = "Megabytes",
        batch_id = %batch_id,
        "CloudWatch Metric"
    );

    info!(
        metric_name = "SymbolsPerSecond",
        metric_value = symbols_per_second,
        metric_unit = "Count/Second",
        batch_id = %batch_id,
        "CloudWatch Metric"
    );

    info!(
        metric_name = "ThreadCount",
        metric_value = thread_count,
        metric_unit = "Count",
        batch_id = %batch_id,
        "CloudWatch Metric"
    );

    info!(
        metric_name = "ErrorRate",
        metric_value = error_rate,
        metric_unit = "Percent",
        batch_id = %batch_id,
        "CloudWatch Metric"
    );

    // Log error summary if any symbols failed
    if !errors.is_empty() {
        warn!(
            batch_id = %batch_id,
            error_count = error_count,
            total_symbols = symbol_count,
            error_rate = error_rate,
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

    // Prepare result with error tracking
    let scan_result = ScanResult {
        batch_id: batch_id.clone(),
        results,
        errors,
        success_count,
        error_count,
    };

    // Send to SQS with timing
    let sqs_start = Instant::now();
    match sqs_client.send_scan_result(&scan_result).await {
        Ok(_) => {
            let sqs_duration_ms = sqs_start.elapsed().as_millis() as u64;
            info!(
                batch_id = %batch_id,
                sqs_duration_ms = sqs_duration_ms,
                "Results sent to SQS successfully"
            );
        }
        Err(e) => {
            error!(
                batch_id = %batch_id,
                error = %e,
                "Failed to send results to SQS"
            );
            return Err(e);
        }
    }

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

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use atc_serverless::{ATCConfig, BatchRequest};
    use lambda_runtime::Context;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct MockSqsClient {
        called: AtomicBool,
        should_fail: bool,
    }

    impl MockSqsClient {
        fn success() -> Self {
            Self {
                called: AtomicBool::new(false),
                should_fail: false,
            }
        }

        fn fail() -> Self {
            Self {
                called: AtomicBool::new(false),
                should_fail: true,
            }
        }

        fn was_called(&self) -> bool {
            self.called.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl SqsSender for MockSqsClient {
        async fn send_scan_result(&self, _result: &ScanResult) -> Result<(), Error> {
            self.called.store(true, Ordering::SeqCst);
            if self.should_fail {
                return Err("mock sqs failure".into());
            }
            Ok(())
        }
    }

    fn valid_batch_request() -> BatchRequest {
        let mut weights = HashMap::new();
        weights.insert("1h".to_string(), 1.0);

        let close = (0..64)
            .map(|i| 100.0 + i as f64 * 0.1)
            .collect::<Vec<_>>();

        let mut timeframes = HashMap::new();
        timeframes.insert(
            "1h".to_string(),
            atc_serverless::OHLCVData {
                timestamp: (1..=64).map(|value| value as i64).collect::<Vec<_>>().into_boxed_slice(),
                open: close.iter().map(|value| value - 0.1).collect::<Vec<_>>().into_boxed_slice(),
                high: close.iter().map(|value| value + 0.5).collect::<Vec<_>>().into_boxed_slice(),
                low: close.iter().map(|value| value - 0.5).collect::<Vec<_>>().into_boxed_slice(),
                close: close.into_boxed_slice(),
                volume: vec![1000.0; 64].into_boxed_slice(),
            },
        );

        BatchRequest {
            batch_id: "test-batch".to_string(),
            version: Some("1.0.0".to_string()),
            symbols: vec![atc_serverless::SymbolData {
                symbol: "BTCUSDT".to_string(),
                timeframes,
            }],
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

    fn invalid_batch_request() -> BatchRequest {
        BatchRequest {
            batch_id: "test-batch".to_string(),
            version: None,
            symbols: vec![],
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
        let mock_sqs_client = MockSqsClient::success();

        let result = handle_request(event, &mock_sqs_client).await;
        assert!(result.is_ok());
        assert!(mock_sqs_client.was_called());
    }

    #[tokio::test]
    async fn test_handler_validation_error() {
        let batch_req = invalid_batch_request();

        let event = LambdaEvent::new(batch_req, Context::default());
        let mock_sqs_client = MockSqsClient::fail();
        
        // This will reject at validation
        let result = handle_request(event, &mock_sqs_client).await;
        assert!(result.is_err());
        assert!(!mock_sqs_client.was_called());
    }
}
