use lambda_runtime::{Error, LambdaEvent};
use atc_serverless::{BatchRequest, process_batch, ScanResult};
use crate::sqs::SqsClient;
use tracing::{info, warn, error};
use std::time::Instant;

/// Handle incoming Lambda request
/// 
/// Processes a batch of symbols and sends results to SQS.
/// Includes comprehensive logging and error handling.
///
/// # Arguments
/// * `event` - Lambda event containing the batch request
/// * `sqs_client` - SQS client for sending results
///
/// # Returns
/// Result indicating success or failure
pub async fn handle_request(
    event: LambdaEvent<BatchRequest>,
    sqs_client: &SqsClient,
) -> Result<(), Error> {
    let request = event.payload;
    let batch_id = request.batch_id.clone();
    let symbol_count = request.symbols.len();
    let start_time = Instant::now();

    // Log batch processing start
    info!(
        batch_id = %batch_id,
        symbol_count = symbol_count,
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

    // Process batch with error recovery (CPU intensive, runs on thread pool via rayon)
    let processing_start = Instant::now();
    let (results, errors) = process_batch(request.symbols, request.config);
    let processing_duration_ms = processing_start.elapsed().as_millis() as u64;
    
    let success_count = results.len();
    let error_count = errors.len();

    // Calculate throughput
    let symbols_per_second = if processing_duration_ms > 0 {
        (symbol_count as f64 / processing_duration_ms as f64) * 1000.0
    } else {
        0.0
    };

    // Log processing metrics
    info!(
        batch_id = %batch_id,
        processing_duration_ms = processing_duration_ms,
        symbols_per_second = symbols_per_second,
        success_count = success_count,
        error_count = error_count,
        "Processing completed"
    );

    // Log error summary if any symbols failed
    if !errors.is_empty() {
        warn!(
            batch_id = %batch_id,
            error_count = error_count,
            total_symbols = symbol_count,
            error_rate = (error_count as f64 / symbol_count as f64),
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

    // Log total duration
    let total_duration_ms = start_time.elapsed().as_millis() as u64;
    info!(
        batch_id = %batch_id,
        total_duration_ms = total_duration_ms,
        success_count = success_count,
        error_count = error_count,
        "Batch processing completed successfully"
    );

    Ok(())
}
