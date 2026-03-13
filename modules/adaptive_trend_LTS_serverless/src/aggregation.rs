use crate::multi_tf_voting::aggregate_timeframes;
use crate::parallelism::{
    create_custom_thread_pool, get_rayon_thread_count, ParallelismConfig, ParallelismMetrics,
};
use crate::{ATCConfig, SignalResult, SymbolData, SymbolError};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

static CUSTOM_THREAD_POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> =
    OnceLock::new();

fn get_or_create_custom_thread_pool(num_threads: usize) -> Option<Arc<rayon::ThreadPool>> {
    let pools = CUSTOM_THREAD_POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut pools_guard = match pools.lock() {
        Ok(guard) => guard,
        Err(_poisoned) => {
            crate::log_warn!(
                "Custom thread pool map lock was poisoned. Creating uncached pool for threads={}.",
                num_threads
            );
            return match create_custom_thread_pool(num_threads) {
                Ok(pool) => Some(Arc::new(pool)),
                Err(err) => {
                    crate::log_warn!(
                        "Failed to create fallback custom thread pool (threads={}): {}. Falling back to Rayon default pool.",
                        num_threads,
                        err
                    );
                    None
                }
            };
        }
    };

    if let Some(existing) = pools_guard.get(&num_threads) {
        return Some(existing.clone());
    }

    match create_custom_thread_pool(num_threads) {
        Ok(pool) => {
            let pool = Arc::new(pool);
            pools_guard.insert(num_threads, pool.clone());
            Some(pool)
        }
        Err(err) => {
            crate::log_warn!(
                "Failed to create custom thread pool (threads={}): {}. Falling back to Rayon default pool.",
                num_threads,
                err
            );
            None
        }
    }
}

/// Module for batch processing and error recovery
///
/// This module provides high-level functions for processing batches of symbols with:
/// - Parallel processing using Rayon
/// - Per-symbol error handling and recovery
/// - Memory monitoring and optimization
/// - Performance metrics and logging
///
/// The main entry point is `process_batch()` which handles the complete pipeline
/// from input data to final signal results.
///
/// Static memory warning threshold for the core batch library.
///
/// This is an independent, environment-agnostic guard: if RSS memory exceeds 80 MB
/// during batch processing, a warning is emitted regardless of the Lambda deployment
/// context. This is separate from the Lambda handler's ratio-based thresholds
/// (`MEMORY_WARNING_RATIO = 0.68`, `MEMORY_CRITICAL_RATIO = 0.85` in `handler.rs`),
/// which are relative to the Lambda function's configured memory size.
const MEMORY_WARNING_THRESHOLD_MB: u64 = 80;

/// Get current memory usage in MB
///
/// Returns the RSS (Resident Set Size) memory in megabytes by reading `/proc/self/status` on Linux.
/// Returns 0 on all other platforms (macOS, Windows, etc.), which is expected behavior for
/// Lambda functions (which run on Linux) but may show 0 in local development or CI environments.
pub fn get_memory_usage_mb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let kb: u64 = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    return kb / 1024;
                }
            }
        }
    }
    0
}

fn estimate_batch_memory_mb(symbols: &[SymbolData]) -> u64 {
    let mut total_bytes: usize = 0;
    for symbol in symbols {
        for ohlcv in symbol.timeframes.values() {
            let bars = ohlcv.close.len();
            total_bytes += bars * crate::constants::OHLCV_FIELDS_PER_BAR * 8;
            total_bytes += bars * crate::constants::WORKING_BUFFERS_PER_BAR * 8;
        }
    }
    (total_bytes / (1024 * 1024)) as u64
}

fn process_symbols_parallel(
    symbols: Vec<SymbolData>,
    config: &ATCConfig,
    parallelism_config: Option<&ParallelismConfig>,
) -> Vec<SymbolProcessingResult> {
    if let Some(pconfig) = parallelism_config {
        if let Some(chunk_size) = pconfig.chunk_size {
            return symbols
                .into_par_iter()
                .with_min_len(pconfig.min_len.unwrap_or(5))
                .chunks(chunk_size)
                .flat_map(|chunk| {
                    chunk
                        .into_iter()
                        .map(|symbol_data| process_symbol_with_recovery(symbol_data, config))
                        .collect::<Vec<_>>()
                })
                .collect();
        }

        if let Some(min_len) = pconfig.min_len {
            return symbols
                .into_par_iter()
                .with_min_len(min_len)
                .map(|symbol_data| process_symbol_with_recovery(symbol_data, config))
                .collect();
        }
    }

    symbols
        .into_par_iter()
        .map(|symbol_data| process_symbol_with_recovery(symbol_data, config))
        .collect()
}

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
/// **Note:** This function assumes that the `config` and `symbols` have already
/// been validated. Callers should validate the inputs using the functions in the
/// `validation` module before passing them to this function.
///
/// # Arguments
/// * `symbols` - Vector of symbols to process
/// * `config` - ATC configuration
/// * `parallelism_config` - Optional parallelism configuration (uses defaults if None)
///
/// # Returns
/// Tuple of (successful_results, errors)
///
/// # Example
///
/// ```ignore
/// use atc_serverless::{process_batch, SymbolData, ATCConfig};
/// use atc_serverless::parallelism::ParallelismConfig;
///
/// // Setup your symbols and config
/// let symbols: Vec<SymbolData> = vec![/* ... */];
/// let config = ATCConfig { /* ... */ };
/// let parallelism = ParallelismConfig::default().optimal_for_batch_size(symbols.len());
///
/// let (results, errors) = process_batch(symbols, config, Some(parallelism));
/// println!("Processed {} symbols successfully", results.len());
/// if !errors.is_empty() {
///     eprintln!("{} symbols failed", errors.len());
/// }
/// ```
pub fn process_batch(
    symbols: Vec<SymbolData>,
    config: ATCConfig,
    parallelism_config: Option<ParallelismConfig>,
) -> (Vec<SignalResult>, Vec<SymbolError>) {
    debug_assert!(
        crate::validation::validate_config(&config).is_ok(),
        "Config validation failed"
    );

    let batch_start = Instant::now();
    let epoch_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let batch_id = format!("batch-{}", epoch_ms);

    let num_symbols = symbols.len();
    if num_symbols == 0 {
        crate::log_info!("[{}] Skipping batch processing for 0 symbols.", batch_id);
        return (Vec::new(), Vec::new());
    }

    let estimated_memory_mb = estimate_batch_memory_mb(&symbols);

    let configured_threads = parallelism_config
        .as_ref()
        .and_then(|cfg| cfg.num_threads)
        .filter(|threads| *threads > 0);
    let thread_count = configured_threads.unwrap_or_else(get_rayon_thread_count);

    crate::log_info!(
        "[{}] Starting batch processing with {} symbols (estimated memory: {}MB), threads: {}",
        batch_id,
        num_symbols,
        estimated_memory_mb,
        thread_count
    );

    let results: Vec<SymbolProcessingResult> = if let Some(pconfig) = parallelism_config.as_ref() {
        if pconfig.use_custom_pool {
            if let Some(num_threads) = configured_threads {
                if let Some(pool) = get_or_create_custom_thread_pool(num_threads) {
                    pool.install(|| process_symbols_parallel(symbols, &config, Some(pconfig)))
                } else {
                    process_symbols_parallel(symbols, &config, Some(pconfig))
                }
            } else {
                process_symbols_parallel(symbols, &config, Some(pconfig))
            }
        } else {
            process_symbols_parallel(symbols, &config, Some(pconfig))
        }
    } else {
        process_symbols_parallel(symbols, &config, None)
    };

    let mut success_results = Vec::with_capacity(num_symbols);
    let mut error_results = Vec::new();
    let mut total_processing_time_ms: u64 = 0;
    let results_count = results.len();

    for res in results {
        total_processing_time_ms += res.processing_time_ms;
        if let Some(result) = res.result {
            success_results.push(result);
        }
        if let Some(error) = res.error {
            error_results.push(error);
        }
    }

    let batch_duration_ms = batch_start.elapsed().as_millis() as u64;
    let avg_cpu_time_per_symbol_ms = if results_count > 0 {
        total_processing_time_ms / results_count as u64
    } else {
        0
    };
    let avg_wall_clock_per_symbol_ms = if num_symbols > 0 {
        batch_duration_ms / num_symbols as u64
    } else {
        0
    };

    let memory_usage_mb = get_memory_usage_mb();
    let memory_info = if memory_usage_mb > 0 {
        format!(", memory={}MB", memory_usage_mb)
    } else {
        String::new()
    };

    let parallelism_metrics =
        ParallelismMetrics::calculate(thread_count, num_symbols, batch_duration_ms);
    parallelism_metrics.log_metrics(&batch_id);

    crate::log_info!("[{}] Batch processing completed: {} successful, {} errors, total_time={}ms, avg_wall_clock_per_symbol={}ms, avg_cpu_time_per_symbol={}ms{}",
              batch_id,
              success_results.len(),
              error_results.len(),
              batch_duration_ms,
              avg_wall_clock_per_symbol_ms,
              avg_cpu_time_per_symbol_ms,
              memory_info);

    if memory_usage_mb > MEMORY_WARNING_THRESHOLD_MB {
        crate::log_warn!(
            "[{}] Memory usage {}MB exceeds threshold {}MB",
            batch_id,
            memory_usage_mb,
            MEMORY_WARNING_THRESHOLD_MB
        );
    }

    for error in &error_results {
        crate::log_error!(
            "[{}] Symbol {} failed: {}",
            batch_id,
            error.symbol,
            error.error
        );
    }

    (success_results, error_results)
}

/// Process a single symbol with error recovery and timing
///
/// This function processes a single symbol with comprehensive error handling:
/// - Catches panics to prevent batch failures
/// - Measures processing time for performance monitoring
/// - Returns structured result with success/error state
///
/// # Arguments
/// * `symbol_data` - Symbol data to process
/// * `config` - ATC configuration
///
/// # Returns
/// SymbolProcessingResult containing either the successful signal result or error information
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
        Ok(Ok(signal_result)) => SymbolProcessingResult {
            result: Some(signal_result),
            error: None,
            processing_time_ms,
        },
        Ok(Err(err)) => SymbolProcessingResult {
            result: None,
            error: Some(SymbolError { symbol, error: err }),
            processing_time_ms,
        },
        Err(_) => {
            crate::log_error!(
                "Panic while processing symbol: {} (time: {}ms)",
                symbol,
                processing_time_ms
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
///
/// This function processes a single symbol through the complete ATC pipeline:
/// 1. Calculates signals for all configured timeframes
/// 2. Aggregates signals across timeframes
/// 3. Returns final signal result
///
/// # Arguments
/// * `symbol_data` - Symbol data containing OHLCV data for multiple timeframes
/// * `config` - ATC configuration
///
/// # Returns
/// SignalResult containing the final score, signal type, and per-timeframe details
fn process_single_symbol(
    symbol_data: SymbolData,
    config: &ATCConfig,
) -> Result<SignalResult, String> {
    let num_timeframes = symbol_data.timeframes.len();
    if num_timeframes == 0 {
        return Err("Symbol has no timeframe data".to_string());
    }

    let mut tf_scores = HashMap::with_capacity(num_timeframes);
    let mut tf_details = HashMap::with_capacity(num_timeframes);

    for (tf, ohlcv) in symbol_data.timeframes {
        if ohlcv.close.len() < crate::constants::MIN_DATA_LENGTH {
            return Err(format!(
                "Insufficient OHLCV length for timeframe '{}': got {}, require at least {}",
                tf,
                ohlcv.close.len(),
                crate::constants::MIN_DATA_LENGTH
            ));
        }

        // PARITY CONTRACT: `compute_symbol_score` owns Layer-1 vote discretization
        // (`{-1, 0, 1}` via `config.threshold`) before final MA aggregation.
        // Keep this call path aligned with `signal_detection.rs` compatibility docs.
        let (score, signal) = crate::signal_detection::compute_symbol_score(&ohlcv.close, config);

        tf_scores.insert(tf.clone(), score);
        tf_details.insert(tf, signal);
    }

    Ok(aggregate_timeframes(
        symbol_data.symbol,
        tf_scores,
        tf_details,
        config,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ATCConfig, MAConfig, MAType, OHLCVData};

    #[test]
    fn test_estimate_batch_memory_mb() {
        let mut timeframes = HashMap::new();
        let bars = 200_000;
        let ohlcv = OHLCVData {
            timestamp: vec![0; bars].into_boxed_slice(),
            open: vec![0.0; bars].into_boxed_slice(),
            high: vec![0.0; bars].into_boxed_slice(),
            low: vec![0.0; bars].into_boxed_slice(),
            close: vec![0.0; bars].into_boxed_slice(),
            volume: vec![0.0; bars].into_boxed_slice(),
        };
        timeframes.insert("1h".to_string(), ohlcv.clone());
        timeframes.insert("4h".to_string(), ohlcv);

        let symbol = SymbolData {
            symbol: "BTCUSDT".to_string(),
            timeframes,
        };

        let mem_mb = estimate_batch_memory_mb(&[symbol]);

        let expected_bytes = (bars * crate::constants::OHLCV_FIELDS_PER_BAR * 8) * 2
            + (bars * crate::constants::WORKING_BUFFERS_PER_BAR * 8) * 2;
        let expected_mb = (expected_bytes / (1024 * 1024)) as u64;

        assert_eq!(mem_mb, expected_mb);
        assert!(mem_mb > 0, "Expected memory estimate to be greater than 0");
    }

    fn build_test_symbol(symbol: &str) -> SymbolData {
        let bars = 64;
        let mut timeframes = HashMap::new();
        timeframes.insert(
            "1h".to_string(),
            OHLCVData {
                timestamp: (0..bars as i64).collect::<Vec<i64>>().into_boxed_slice(),
                open: (0..bars)
                    .map(|i| 100.0 + i as f64)
                    .collect::<Vec<f64>>()
                    .into_boxed_slice(),
                high: (0..bars)
                    .map(|i| 101.0 + i as f64)
                    .collect::<Vec<f64>>()
                    .into_boxed_slice(),
                low: (0..bars)
                    .map(|i| 99.0 + i as f64)
                    .collect::<Vec<f64>>()
                    .into_boxed_slice(),
                close: (0..bars)
                    .map(|i| 100.5 + i as f64)
                    .collect::<Vec<f64>>()
                    .into_boxed_slice(),
                volume: vec![1000.0; bars].into_boxed_slice(),
            },
        );

        SymbolData {
            symbol: symbol.to_string(),
            timeframes,
        }
    }

    fn build_test_config() -> ATCConfig {
        let mut weights = HashMap::new();
        weights.insert("1h".to_string(), 1.0);

        ATCConfig {
            robustness: crate::Robustness::Medium,
            weights,
            threshold: 0.3,
            min_signal: 0.0,
            use_signal_strength: true,
            lambda_param: 0.02,
            decay: 0.03,
            cutout: 0,
            equity_floor: 0.25,
            ma_configs: vec![MAConfig {
                ma_type: MAType::Ema,
                length: 20,
                weight: 1.0,
            }],
        }
    }

    #[test]
    fn test_process_batch_reuses_custom_thread_pool() {
        let before_pools = CUSTOM_THREAD_POOLS
            .get()
            .map(|map| map.lock().expect("pool map lock").len())
            .unwrap_or(0);

        let parallelism_first = ParallelismConfig::default().with_threads(2);
        let _ = process_batch(
            vec![build_test_symbol("BTCUSDT")],
            build_test_config(),
            Some(parallelism_first),
        );

        let first_pool_ptr =
            get_or_create_custom_thread_pool(2).expect("thread pool should be creatable").as_ref()
                as *const rayon::ThreadPool as usize;

        let parallelism_second = ParallelismConfig::default().with_threads(2);
        let _ = process_batch(
            vec![build_test_symbol("ETHUSDT")],
            build_test_config(),
            Some(parallelism_second),
        );

        let second_pool_ptr =
            get_or_create_custom_thread_pool(2).expect("thread pool should be creatable").as_ref()
                as *const rayon::ThreadPool as usize;
        let after_pools = CUSTOM_THREAD_POOLS
            .get()
            .map(|map| map.lock().expect("pool map lock").len())
            .unwrap_or(0);

        assert_eq!(first_pool_ptr, second_pool_ptr);
        assert!(after_pools >= before_pools);
    }

    #[test]
    fn test_process_batch_uses_distinct_pool_per_thread_count() {
        let pool_2 = get_or_create_custom_thread_pool(2).expect("thread pool should be creatable");
        let pool_4 = get_or_create_custom_thread_pool(4).expect("thread pool should be creatable");

        let pool_2_ptr = pool_2.as_ref() as *const rayon::ThreadPool as usize;
        let pool_4_ptr = pool_4.as_ref() as *const rayon::ThreadPool as usize;

        assert_ne!(pool_2_ptr, pool_4_ptr);
    }
}
