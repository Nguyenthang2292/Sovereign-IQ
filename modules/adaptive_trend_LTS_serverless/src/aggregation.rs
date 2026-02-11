use crate::multi_tf_voting::aggregate_timeframes;
use crate::{ATCConfig, SignalResult, SymbolData, SymbolError};
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

/// Result of processing a single symbol
pub struct SymbolProcessingResult {
    /// Successful result if processing succeeded
    pub result: Option<SignalResult>,
    /// Error information if processing failed
    pub error: Option<SymbolError>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Process a batch of symbols with per-symbol error handling
///
/// This function processes all symbols in parallel using Rayon. If individual
/// symbols fail, they are tracked in the errors list rather than failing the
/// entire batch.
///
/// # Arguments
/// * `symbols` - Vector of symbols to process
/// * `config` - ATC configuration
///
/// # Returns
/// Tuple of (successful_results, errors)
///
/// # Example
///
/// ```ignore
/// use atc_serverless::{process_batch, SymbolData, ATCConfig};
///
/// // Setup your symbols and config
/// let symbols: Vec<SymbolData> = vec![/* ... */];
/// let config = ATCConfig { /* ... */ };
///
/// let (results, errors) = process_batch(symbols, config);
/// println!("Processed {} symbols successfully", results.len());
/// if !errors.is_empty() {
///     eprintln!("{} symbols failed", errors.len());
/// }
/// ```
pub fn process_batch(
    symbols: Vec<SymbolData>,
    config: ATCConfig,
) -> (Vec<SignalResult>, Vec<SymbolError>) {
    let batch_start = Instant::now();
    let batch_id = format!("batch-{}", batch_start.elapsed().as_millis());

    eprintln!(
        "[INFO] [{}] Starting batch processing with {} symbols",
        batch_id,
        symbols.len()
    );

    let results: Vec<SymbolProcessingResult> = symbols
        .into_par_iter()
        .map(|symbol_data| process_symbol_with_recovery(symbol_data, &config))
        .collect();

    let mut success_results = Vec::new();
    let mut error_results = Vec::new();
    let mut total_processing_time_ms: u64 = 0;

    for res in &results {
        if let Some(result) = &res.result {
            success_results.push(result.clone());
        }
        if let Some(error) = &res.error {
            error_results.push(error.clone());
        }
        total_processing_time_ms += res.processing_time_ms;
    }

    let batch_duration_ms = batch_start.elapsed().as_millis() as u64;
    let avg_symbol_time_ms = if !results.is_empty() {
        total_processing_time_ms / results.len() as u64
    } else {
        0
    };

    // Structured logging
    eprintln!("[INFO] [{}] Batch processing completed: {} successful, {} errors, total_time={}ms, avg_symbol_time={}ms",
              batch_id, 
              success_results.len(), 
              error_results.len(),
              batch_duration_ms,
              avg_symbol_time_ms);

    // Log individual errors
    for error in &error_results {
        eprintln!(
            "[ERROR] [{}] Symbol {} failed: {}",
            batch_id, error.symbol, error.error
        );
    }

    (success_results, error_results)
}

/// Process a single symbol with error recovery and timing
fn process_symbol_with_recovery(
    symbol_data: SymbolData,
    config: &ATCConfig,
) -> SymbolProcessingResult {
    let symbol = symbol_data.symbol.clone();
    let start_time = Instant::now();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        process_single_symbol(symbol_data, config)
    }));

    let processing_time_ms = start_time.elapsed().as_millis() as u64;

    match result {
        Ok(signal_result) => SymbolProcessingResult {
            result: Some(signal_result),
            error: None,
            processing_time_ms,
        },
        Err(_) => {
            eprintln!(
                "[ERROR] Panic while processing symbol: {} (time: {}ms)",
                symbol, processing_time_ms
            );
            SymbolProcessingResult {
                result: None,
                error: Some(SymbolError {
                    symbol,
                    error: "Processing panic - check data validity".to_string(),
                }),
                processing_time_ms,
            }
        }
    }
}

/// Process a single symbol (internal function)
fn process_single_symbol(symbol_data: SymbolData, config: &ATCConfig) -> SignalResult {
    let mut tf_scores = HashMap::new();
    let mut tf_details = HashMap::new();
    let mut tf_strengths = HashMap::new();

    for (tf, ohlcv) in symbol_data.timeframes {
        // Calculate score for this timeframe
        let (score, signal) = crate::signal_detection::compute_symbol_score(&ohlcv.close, config);

        tf_scores.insert(tf.clone(), score);
        tf_details.insert(tf.clone(), signal);
        tf_strengths.insert(tf, score);
    }

    // Aggregate across timeframes
    aggregate_timeframes(
        symbol_data.symbol,
        tf_scores,
        tf_details,
        tf_strengths,
        config,
    )
}
