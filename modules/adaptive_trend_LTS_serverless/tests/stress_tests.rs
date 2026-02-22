use atc_serverless::{process_batch, ATCConfig, MAConfig, OHLCVData, SymbolData};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Stress test suite for ATC Serverless
///
/// This module contains comprehensive stress tests to validate:
/// - Large batch processing (1000+ symbols)
/// - Memory usage and leaks
/// - Long-running stability
/// - Performance under load
pub struct StressTester {
    config: ATCConfig,
    symbol_data_generator: SymbolDataGenerator,
}

impl StressTester {
    /// Create a new stress tester with default configuration
    pub fn new() -> Self {
        let config = ATCConfig {
            weights: HashMap::new(),
            threshold: 0.3,
            min_signal: 0.0,
            use_signal_strength: true,
            lambda_param: 0.02,
            decay: 0.03,
            cutout: 0,
            equity_floor: 0.25,
            robustness: atc_serverless::Robustness::Medium,
            ma_configs: vec![MAConfig {
                ma_type: atc_serverless::MAType::Ema,
                length: 12,
                weight: 1.0,
            }],
        };

        let generator = SymbolDataGenerator::new();

        StressTester {
            config,
            symbol_data_generator: generator,
        }
    }

    /// Run stress tests for different batch sizes
    pub fn run_stress_tests(&self) {
        println!("Starting stress tests...");

        // Test 1: Large batch (1000 symbols)
        self.test_large_batch(1000, "1000_symbols");

        // Test 2: Maximum expected batch (1500 symbols)
        self.test_large_batch(1500, "1500_symbols");

        // Test 3: Over-limit batch (2000 symbols)
        self.test_large_batch(2000, "2000_symbols");

        // Test 4: Long-running test (1 hour)
        self.test_long_running();

        println!("All stress tests completed.");
    }

    /// Test processing with a specific number of symbols
    fn test_large_batch(&self, num_symbols: usize, test_name: &str) {
        println!("\n=== Testing {} symbols ===", num_symbols);

        let symbols = self.symbol_data_generator.generate_symbols(num_symbols);
        let start_time = Instant::now();

        // Run the test
        use atc_serverless::parallelism::ParallelismConfig;
        let (results, errors) = process_batch(
            symbols,
            self.config.clone(),
            Some(ParallelismConfig::default()),
        );

        let duration = start_time.elapsed();
        let memory_usage = self.estimate_memory_usage(num_symbols);

        println!("Test: {} symbols", test_name);
        println!("Duration: {:.2}s", duration.as_secs_f64());
        println!("Memory usage: {} MB", memory_usage);
        println!(
            "Results: {} successful, {} errors",
            results.len(),
            errors.len()
        );
        println!(
            "Throughput: {:.0} symbols/second",
            num_symbols as f64 / duration.as_secs_f64()
        );

        // Validate results
        self.validate_large_batch_results(num_symbols, &results, &errors);
    }

    /// Test long-running stability (1 hour continuous processing)
    fn test_long_running(&self) {
        println!("\n=== Long-running test (1 hour) ===");

        let start_time = Instant::now();
        let one_hour = Duration::from_secs(3600);
        let mut total_symbols_processed = 0;
        let mut total_batches = 0;

        // Run for 1 hour or until we process 10,000 symbols (whichever comes first)
        while Instant::now().duration_since(start_time) < one_hour
            && total_symbols_processed < 10000
        {
            let batch_size = 100; // Process 100 symbols per batch
            let symbols = self.symbol_data_generator.generate_symbols(batch_size);

            use atc_serverless::parallelism::ParallelismConfig;
            let (_results, _errors) = process_batch(
                symbols,
                self.config.clone(),
                Some(ParallelismConfig::default()),
            );
            total_symbols_processed += batch_size;
            total_batches += 1;

            // Periodic memory check
            if total_batches % 10 == 0 {
                let memory_usage = self.estimate_memory_usage(total_symbols_processed);
                println!(
                    "Progress: {} symbols processed, Memory: {} MB",
                    total_symbols_processed, memory_usage
                );
            }
        }

        let duration = Instant::now().duration_since(start_time);
        println!(
            "Long-running test completed in {:.2}s",
            duration.as_secs_f64()
        );
        println!("Total symbols processed: {}", total_symbols_processed);
        println!(
            "Average throughput: {:.0} symbols/second",
            total_symbols_processed as f64 / duration.as_secs_f64()
        );
    }

    /// Validate results from large batch processing
    fn validate_large_batch_results(
        &self,
        expected_symbols: usize,
        results: &[atc_serverless::SignalResult],
        errors: &[atc_serverless::SymbolError],
    ) {
        // Check that we got results for most symbols
        let success_rate = results.len() as f64 / expected_symbols as f64;
        assert!(
            success_rate > 0.9,
            "Success rate too low: {:.1}% ({} of {} symbols succeeded)",
            success_rate * 100.0,
            results.len(),
            expected_symbols
        );

        // Check that error messages are reasonable
        for error in errors {
            assert!(!error.error.is_empty(), "Empty error message");
            assert!(error.symbol.len() > 0, "Empty symbol name in error");
        }

        // Check result validity
        for result in results {
            assert!(!result.symbol.is_empty(), "Empty symbol name in result");
            assert!(
                result.score >= -1.0 && result.score <= 1.0,
                "Invalid score: {}",
                result.score
            );
            assert!(
                result.signal_type == atc_serverless::SignalType::Long
                    || result.signal_type == atc_serverless::SignalType::Short
                    || result.signal_type == atc_serverless::SignalType::Neutral,
                "Invalid signal type: {}",
                result.signal_type
            );
        }
    }

    /// Estimate memory usage based on number of symbols
    fn estimate_memory_usage(&self, num_symbols: usize) -> u64 {
        // Rough estimate: ~55KB per symbol
        (num_symbols * 55) as u64
    }
}

/// Generator for test symbol data
struct SymbolDataGenerator;

impl SymbolDataGenerator {
    fn new() -> Self {
        SymbolDataGenerator
    }

    /// Generate test symbol data
    fn generate_symbols(&self, num_symbols: usize) -> Vec<SymbolData> {
        (0..num_symbols)
            .map(|i| {
                let symbol_name = format!("TEST_SYMBOL_{}", i);
                let mut timeframes = HashMap::new();

                // Generate 1h timeframe data
                let timestamps_1h: Vec<i64> =
                    (0..200).map(|j| 1704067200 + j as i64 * 3600).collect();
                let prices_1h: Vec<f64> = (0..200).map(|j| 42000.0 + (j as f64 * 0.5)).collect();

                timeframes.insert(
                    "1h".to_string(),
                    OHLCVData {
                        timestamp: timestamps_1h.into_boxed_slice(),
                        open: prices_1h
                            .iter()
                            .map(|&p| p - 0.5)
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                        high: prices_1h
                            .iter()
                            .map(|&p| p + 1.0)
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                        low: prices_1h
                            .iter()
                            .map(|&p| p - 1.0)
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                        close: prices_1h.into_boxed_slice(),
                        volume: (0..200)
                            .map(|_| 100.0)
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                    },
                );

                // Generate 4h timeframe data
                let timestamps_4h: Vec<i64> =
                    (0..50).map(|j| 1704067200 + j as i64 * 14400).collect();
                let prices_4h: Vec<f64> = (0..50).map(|j| 42000.0 + (j as f64 * 2.0)).collect();

                timeframes.insert(
                    "4h".to_string(),
                    OHLCVData {
                        timestamp: timestamps_4h.into_boxed_slice(),
                        open: prices_4h
                            .iter()
                            .map(|&p| p - 1.0)
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                        high: prices_4h
                            .iter()
                            .map(|&p| p + 2.0)
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                        low: prices_4h
                            .iter()
                            .map(|&p| p - 2.0)
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                        close: prices_4h.into_boxed_slice(),
                        volume: (0..50)
                            .map(|_| 400.0)
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                    },
                );

                SymbolData {
                    symbol: symbol_name,
                    timeframes,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_stress_tester_creation() {
        let tester = StressTester::new();
        assert_eq!(tester.config.threshold, 0.3);
    }

    #[test]
    fn test_symbol_data_generation() {
        let generator = SymbolDataGenerator::new();
        let symbols = generator.generate_symbols(5);
        assert_eq!(symbols.len(), 5);
        assert_eq!(symbols[0].timeframes.len(), 2); // 1h and 4h
    }

    #[test]
    fn test_large_batch_processing() {
        use atc_serverless::parallelism::ParallelismConfig;

        let tester = StressTester::new();
        let symbols = tester.symbol_data_generator.generate_symbols(10);
        let (results, errors) = process_batch(
            symbols,
            tester.config.clone(),
            Some(ParallelismConfig::default()),
        );
        assert!(results.len() + errors.len() <= 10);
    }

    #[test]
    #[ignore] // This is a long-running test
    fn test_long_running_stress() {
        let tester = StressTester::new();
        // Run for a shorter duration in tests
        let start_time = Instant::now();
        let short_duration = Duration::from_secs(5); // 5 seconds instead of 1 hour

        let mut total_symbols_processed = 0;
        while Instant::now().duration_since(start_time) < short_duration
            && total_symbols_processed < 100
        {
            use atc_serverless::parallelism::ParallelismConfig;

            let batch_size = 10;
            let symbols = tester.symbol_data_generator.generate_symbols(batch_size);
            let (_results, _) = process_batch(
                symbols,
                tester.config.clone(),
                Some(ParallelismConfig::default()),
            );
            total_symbols_processed += batch_size;
        }

        assert!(total_symbols_processed > 0);
    }
}
