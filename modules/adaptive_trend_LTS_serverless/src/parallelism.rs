//! Parallelism configuration and metrics for batch processing
//!
//! This module provides utilities for tuning parallelism in the ATC batch processing pipeline.

/// Configuration for parallel processing
///
/// Controls thread pool settings, chunking, and min partition lengths for Rayon.
#[derive(Debug, Clone, Default)]
pub struct ParallelismConfig {
    /// Number of threads to use (None = use Rayon default)
    pub num_threads: Option<usize>,
    /// Chunk size for parallel iteration
    pub chunk_size: Option<usize>,
    /// Minimum length for parallel partitions
    pub min_len: Option<usize>,
    /// Whether to use a custom thread pool
    pub use_custom_pool: bool,
}

impl ParallelismConfig {
    fn default_threads_for_batch_size(batch_size: usize) -> Option<usize> {
        match batch_size {
            0..=10 => Some(2),
            11..=50 => Some(4),
            51..=100 => Some(6),
            101..=500 => Some(8),
            _ => None,
        }
    }

    fn default_chunk_size_for_batch_size(batch_size: usize) -> usize {
        match batch_size {
            0..=10 => 1,
            11..=50 => 5,
            51..=100 => 10,
            101..=500 => 25,
            _ => 50,
        }
    }

    fn lambda_memory_thread_cap_for_mb(memory_mb: usize) -> usize {
        // Operational Lambda rule used by this module:
        // - 1769 MB maps to ~1 vCPU
        // - 3008 MB maps to ~2 vCPU equivalent for scheduling
        //
        // We convert memory to vCPU-equivalent by ceiling division, then multiply by 2
        // to allow one worker per hyperthread.
        //
        // Examples:
        // - 1769 MB -> ceil(1769 / 1769) * 2 = 2 threads
        // - 3008 MB -> ceil(3008 / 1769) * 2 = 4 threads
        let vcpu_equivalent = memory_mb.max(1).div_ceil(1769);
        vcpu_equivalent * 2
    }

    fn lambda_memory_thread_cap() -> Option<usize> {
        std::env::var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .map(Self::lambda_memory_thread_cap_for_mb)
    }

    fn apply_lambda_thread_cap(
        default_threads: Option<usize>,
        lambda_cap_threads: Option<usize>,
    ) -> Option<usize> {
        match (default_threads, lambda_cap_threads) {
            (Some(default_threads), Some(cap_threads)) => Some(default_threads.min(cap_threads)),
            (Some(default_threads), None) => Some(default_threads),
            // For large batches where tier defaults return None, still honor Lambda
            // memory cap to avoid oversubscription on constrained runtimes.
            (None, Some(cap_threads)) => Some(cap_threads),
            (None, None) => None,
        }
    }

    /// Set the number of threads for parallel processing
    pub fn with_threads(mut self, threads: usize) -> Self {
        if threads > 0 {
            self.num_threads = Some(threads);
            self.use_custom_pool = true;
        }
        self
    }

    /// Set the chunk size for parallel iteration
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        if chunk_size > 0 {
            self.chunk_size = Some(chunk_size);
        }
        self
    }

    /// Set the minimum length for parallel partitions
    pub fn with_min_len(mut self, min_len: usize) -> Self {
        if min_len > 0 {
            self.min_len = Some(min_len);
        }
        self
    }

    /// Generate optimal configuration based on batch size
    ///
    /// This provides sensible defaults for different batch sizes:
    /// - 0-10 symbols: 2 threads, chunk_size=1
    /// - 11-50 symbols: 4 threads, chunk_size=5
    /// - 51-100 symbols: 6 threads, chunk_size=10
    /// - 101-500 symbols: 8 threads, chunk_size=25
    /// - 500+ symbols: use Lambda memory-derived cap when available, otherwise
    ///   fallback to Rayon default thread pool (chunk_size=50)
    pub fn optimal_for_batch_size(&self, batch_size: usize) -> Self {
        // 1. Check for Environment Variable overrides (for tuning on AWS Lambda without redeploy)
        if let Ok(val) = std::env::var("ATC_FORCE_THREADS") {
            match val.parse::<usize>() {
                Ok(threads) if threads > 0 => {
                    let chunk_size = match std::env::var("ATC_FORCE_CHUNK_SIZE") {
                        Ok(chunk_val) => match chunk_val.parse::<usize>() {
                            Ok(chunk_size) if chunk_size > 0 => chunk_size,
                            Ok(_) => {
                                crate::log_warn!(
                                    "ATC_FORCE_CHUNK_SIZE must be > 0 (received 0). Using default for batch size {}.",
                                    batch_size
                                );
                                Self::default_chunk_size_for_batch_size(batch_size)
                            }
                            Err(_) => {
                                crate::log_warn!(
                                    "Invalid ATC_FORCE_CHUNK_SIZE='{}'. Using default for batch size {}.",
                                    chunk_val, batch_size
                                );
                                Self::default_chunk_size_for_batch_size(batch_size)
                            }
                        },
                        Err(_) => Self::default_chunk_size_for_batch_size(batch_size),
                    };

                    return Self {
                        num_threads: Some(threads),
                        chunk_size: Some(chunk_size),
                        min_len: Some(5),
                        use_custom_pool: true,
                    };
                }
                Ok(_) => {
                    crate::log_warn!(
                        "ATC_FORCE_THREADS must be > 0 (received 0). Falling back to batch-size defaults."
                    );
                }
                Err(_) => {
                    crate::log_warn!(
                        "Invalid ATC_FORCE_THREADS='{}'. Falling back to batch-size defaults.",
                        val
                    );
                }
            }
        }

        // 2. Default logic based on benchmarks
        let num_threads = Self::apply_lambda_thread_cap(
            Self::default_threads_for_batch_size(batch_size),
            Self::lambda_memory_thread_cap(),
        );
        let chunk_size = Self::default_chunk_size_for_batch_size(batch_size);

        Self {
            num_threads,
            chunk_size: Some(chunk_size),
            min_len: Some(5),
            use_custom_pool: num_threads.is_some(),
        }
    }
}

/// Create a custom Rayon thread pool with specified number of threads
///
/// This is useful for scenarios where you want to limit parallelism
/// or use a dedicated thread pool.
pub fn create_custom_thread_pool(
    num_threads: usize,
) -> Result<rayon::ThreadPool, rayon::ThreadPoolBuildError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
}

/// Metrics for measuring parallelism performance
#[derive(Debug, Clone)]
pub struct ParallelismMetrics {
    /// Number of threads used
    pub thread_count: usize,
    /// Number of symbols processed
    pub batch_size: usize,
    /// Total processing time in milliseconds
    pub processing_time_ms: u64,
    /// Average time per symbol in milliseconds
    pub avg_time_per_symbol_ms: u64,
    /// Throughput in symbols per second
    pub symbols_per_second: f64,
    /// Parallel efficiency (0.0 to 1.0)
    /// - 1.0 = perfect linear speedup
    /// - < 1.0 = some overhead from parallelism
    ///
    /// NOTE: None means "not measured" because sequential baseline is unavailable.
    pub parallel_efficiency: Option<f64>,
}

impl ParallelismMetrics {
    /// Calculate parallelism metrics from measured values
    ///
    /// Parallel efficiency is calculated as:
    /// (single_thread_time) / (parallel_time * num_threads)
    ///
    /// This represents how close we are to ideal linear speedup.
    /// NOTE: To calculate meaningful parallel efficiency, we need either:
    /// 1. Actual sequential processing time (from a single-threaded benchmark run), or
    /// 2. Per-symbol individual timings from the parallel run
    ///
    /// The current implementation leaves this as 0.0 (placeholder)
    pub fn calculate(thread_count: usize, batch_size: usize, processing_time_ms: u64) -> Self {
        let avg_time_per_symbol_ms = if batch_size > 0 {
            processing_time_ms / batch_size as u64
        } else {
            0
        };

        let symbols_per_second = if processing_time_ms > 0 {
            (batch_size as f64 / processing_time_ms as f64) * 1000.0
        } else {
            0.0
        };

        // NOTE: parallel_efficiency calculation requires sequential baseline
        // which is not available at runtime.
        let parallel_efficiency = None;

        Self {
            thread_count,
            batch_size,
            processing_time_ms,
            avg_time_per_symbol_ms,
            symbols_per_second,
            parallel_efficiency,
        }
    }

    /// Log metrics to stderr
    pub fn log_metrics(&self, batch_id: &str) {
        let efficiency_str = match self.parallel_efficiency {
            Some(value) => format!("{:.2}%", value * 100.0),
            None => "N/A (needs baseline)".to_string(),
        };

        crate::log_info!(
            "[{}] Parallelism metrics: threads={}, batch_size={}, time={}ms, throughput={:.2} symbols/s, efficiency={}",
            batch_id,
            self.thread_count,
            self.batch_size,
            self.processing_time_ms,
            self.symbols_per_second,
            efficiency_str
        );
    }
}

/// Get the current Rayon thread count
pub fn get_rayon_thread_count() -> usize {
    rayon::current_num_threads()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallelism_config_defaults() {
        let config = ParallelismConfig::default();
        assert!(config.num_threads.is_none());
        assert!(config.chunk_size.is_none());
    }

    #[test]
    fn test_parallelism_config_with_threads() {
        let config = ParallelismConfig::default().with_threads(4);
        assert_eq!(config.num_threads, Some(4));
        assert!(config.use_custom_pool);
    }

    #[test]
    fn test_parallelism_config_optimal_for_batch_size() {
        let config = ParallelismConfig::default().optimal_for_batch_size(50);
        assert_eq!(config.num_threads, Some(4));
        assert_eq!(config.chunk_size, Some(5));
    }

    #[test]
    fn test_lambda_memory_thread_cap_formula_examples() {
        assert_eq!(ParallelismConfig::lambda_memory_thread_cap_for_mb(1769), 2);
        assert_eq!(ParallelismConfig::lambda_memory_thread_cap_for_mb(3008), 4);
    }

    #[test]
    fn test_apply_lambda_thread_cap_uses_cap_for_large_batches() {
        assert_eq!(
            ParallelismConfig::apply_lambda_thread_cap(None, Some(4)),
            Some(4)
        );
        assert_eq!(
            ParallelismConfig::apply_lambda_thread_cap(Some(8), Some(4)),
            Some(4)
        );
    }

    #[test]
    fn test_parallelism_metrics_calculation() {
        let metrics = ParallelismMetrics::calculate(4, 100, 50);
        assert_eq!(metrics.thread_count, 4);
        assert_eq!(metrics.batch_size, 100);
        assert_eq!(metrics.avg_time_per_symbol_ms, 0);
        assert!(metrics.symbols_per_second > 0.0);
    }

    #[test]
    fn test_parallelism_metrics_efficiency_none() {
        let metrics = ParallelismMetrics::calculate(4, 100, 400);
        assert!(metrics.parallel_efficiency.is_none());
    }

    #[test]
    fn test_create_custom_thread_pool_succeeds() {
        let pool = create_custom_thread_pool(2);
        assert!(pool.is_ok());
    }
}
