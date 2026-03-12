//! # ATC Serverless - Adaptive Trend Classification
//!
//! A high-performance Rust implementation of the Adaptive Trend Classification (ATC) algorithm,
//! optimized for AWS Lambda serverless deployment. This crate provides real-time trading signal
//! detection for cryptocurrency markets with sub-second latency and high throughput.
//!
//! ## Overview
//!
//! This crate implements a multi-timeframe trend classification system that:
//! - Calculates 6 types of Moving Averages (EMA, HMA, WMA, DEMA, LSMA, KAMA)
//! - Uses 8 length variations per MA type (via diflen) for robustness
//! - Applies Layer 1 signal detection with equity-based weighting
//! - Aggregates signals across multiple timeframes
//! - Returns LONG/SHORT/NEUTRAL classifications with confidence scores
//!
//! ## Key Features
//!
//! - **High Performance**: Optimized for AWS Lambda with SIMD accelerations (nightly Rust)
//! - **Error Recovery**: Per-symbol error handling prevents batch failures
//! - **Parallel Processing**: Rayon-based parallelism for maximum throughput
//! - **Memory Efficiency**: SmallVec and buffer pooling for reduced memory usage
//! - **Configurable**: Flexible configuration for different trading strategies
//! - **Robust**: Built-in validation and error handling
//!
//! ## Example
//!
//! ```ignore
//! use atc_serverless::{ATCConfig, MAConfig, process_batch, SymbolData, OHLCVData};
//! use std::collections::HashMap;
//!
//! // Create configuration
//! let config = ATCConfig {
//!     weights: HashMap::new(),
//!     threshold: 0.3,
//!     min_signal: 0.0,
//!     use_signal_strength: true,
//!     robustness: Robustness::Medium,
//!     lambda_param: 0.02,
//!     decay: 0.03,
//!     cutout: 0,
//!     equity_floor: 0.25,
//!     ma_configs: vec![
//!         MAConfig { ma_type: MAType::Ema, length: 20, weight: 1.0 },
//!     ],
//! };
//!
//! // Create symbols data (would come from your data source)
//! let symbols: Vec<SymbolData> = vec![]; // Add your symbol data here
//!
//! // Process a batch of symbols
//! let (results, errors) = process_batch(symbols, config, None);
//! ```

#![warn(missing_docs)]
#![cfg_attr(feature = "simd", feature(portable_simd))]

/// Batch processing and error recovery
pub mod aggregation;
/// Buffer pool for memory optimization
pub mod buffer_pool;
/// Centralized algorithm constants
pub mod constants;
/// Equity curve calculations for Layer 2 weighting
pub mod equity;
/// Moving Average calculations (EMA, HMA, WMA, DEMA, LSMA, KAMA)
pub mod ma_calculations;
/// SIMD-optimized Moving Average calculations (requires `simd` feature)
#[cfg(feature = "simd")]
pub mod ma_simd;
/// Multi-timeframe signal aggregation and voting
pub mod multi_tf_voting;
/// Parallelism configuration and metrics
pub mod parallelism;
/// Signal detection algorithms with diflen and trend classification
pub mod signal_detection;
/// Input validation for OHLCV data, configuration, and schema
pub mod validation;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Macro to log a warning message using tracing or eprintln! based on feature flag
#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {
        #[cfg(feature = "tracing")]
        tracing::warn!($($arg)*);
        #[cfg(not(feature = "tracing"))]
        eprintln!("[WARN] {}", format_args!($($arg)*));
    };
}

/// Macro to log an error message using tracing or eprintln! based on feature flag
#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        #[cfg(feature = "tracing")]
        tracing::error!($($arg)*);
        #[cfg(not(feature = "tracing"))]
        eprintln!("[ERROR] {}", format_args!($($arg)*));
    };
}

/// Macro to log an info message using tracing or eprintln! based on feature flag
#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        #[cfg(feature = "tracing")]
        tracing::info!($($arg)*);
        #[cfg(not(feature = "tracing"))]
        eprintln!("[INFO] {}", format_args!($($arg)*));
    };
}

// Explicit public API re-exports
pub use aggregation::{get_memory_usage_mb, process_batch};
pub use constants::{
    DEFAULT_CUTOUT, DEFAULT_DECAY, DEFAULT_EQUITY_FLOOR, DEFAULT_LAMBDA_PARAM, DEFAULT_MA_LENGTH,
    DEFAULT_MA_WEIGHT, DEFAULT_ROBUSTNESS,
};
pub use equity::calculate_equity;
pub use ma_calculations::{
    calculate_dema, calculate_ema, calculate_hma, calculate_kama, calculate_lsma, calculate_sma,
    calculate_wma,
};
#[cfg(feature = "simd")]
pub use ma_simd::{
    calculate_dema_simd, calculate_ema_simd, calculate_hma_simd, calculate_kama_simd,
    calculate_lsma_simd, calculate_sma_simd, calculate_wma_simd,
};
pub use parallelism::ParallelismConfig;
pub use signal_detection::{
    calculate_diflen, calculate_layer1_signal, compute_symbol_score, Robustness, SignalParams,
};
pub use validation::{
    validate_batch_request, validate_config, validate_ohlcv_data, SCHEMA_VERSION,
};

/// Strategy signal classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum SignalType {
    /// Long / Buy signal
    Long,
    /// Short / Sell signal
    Short,
    /// Neutral / No clear signal
    Neutral,
}

impl std::str::FromStr for SignalType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "LONG" => Ok(SignalType::Long),
            "SHORT" => Ok(SignalType::Short),
            "NEUTRAL" => Ok(SignalType::Neutral),
            _ => Err(format!("Unknown signal type: {}", s)),
        }
    }
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            SignalType::Long => "LONG",
            SignalType::Short => "SHORT",
            SignalType::Neutral => "NEUTRAL",
        };
        write!(f, "{}", s)
    }
}

// --- Data Models ---

/// Request to process a batch of symbols
///
/// This structure represents the input to the batch processing pipeline.
/// It contains a batch ID for tracking, a list of symbols with their OHLCV data,
/// and the configuration for the ATC algorithm.
///
/// # Example
///
/// ```json
/// {
///   "batch_id": "batch-001",
///   "version": "1.0.0",
///   "symbols": [...],
///   "config": {...}
/// }
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchRequest {
    /// Unique identifier for this batch (used for tracking and logging)
    pub batch_id: String,
    /// Schema version for compatibility checking (e.g., "1.0.0")
    /// If omitted, the latest version is assumed
    #[serde(default)]
    pub version: Option<String>,
    /// List of symbols to process in this batch
    pub symbols: Vec<SymbolData>,
    /// Configuration for the ATC algorithm
    pub config: ATCConfig,
}

/// Data for a single symbol across multiple timeframes
///
/// Contains the symbol name and OHLCV data for each timeframe.
#[derive(Debug, Serialize, Deserialize)]
pub struct SymbolData {
    /// Symbol identifier (e.g., "BTCUSDT")
    pub symbol: String,
    /// Map of timeframe name to OHLCV data (e.g., {"1h": OHLCVData, "4h": OHLCVData})
    pub timeframes: HashMap<String, OHLCVData>,
}

/// OHLCV (Open, High, Low, Close, Volume) data structure
///
/// Standard financial data format containing price and volume information.
/// All arrays should have the same length.
///
/// Uses `Box<[f64]>` instead of `Vec<f64>` for immutable arrays to reduce memory overhead.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OHLCVData {
    /// Unix timestamps for each bar
    pub timestamp: Box<[i64]>,
    /// Opening prices
    pub open: Box<[f64]>,
    /// High prices
    pub high: Box<[f64]>,
    /// Low prices
    pub low: Box<[f64]>,
    /// Closing prices
    pub close: Box<[f64]>,
    /// Trading volumes
    pub volume: Box<[f64]>,
}

/// Configuration for the ATC algorithm
///
/// Controls the behavior of signal detection including thresholds,
/// timeframe weights, and MA configurations.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ATCConfig {
    /// Robustness level for diflen calculation: Narrow, Medium, or Wide
    #[serde(default = "default_robustness")]
    pub robustness: Robustness,
    /// Timeframe weights for multi-timeframe aggregation
    /// Example: {"1h": 0.6, "4h": 0.4}
    pub weights: HashMap<String, f64>,
    /// Threshold for LONG/SHORT signal classification
    /// Signals with absolute value > threshold are classified as LONG or SHORT
    pub threshold: f64,
    /// Minimum signal strength to consider
    pub min_signal: f64,
    /// Enable signal strength weighting in aggregation
    pub use_signal_strength: bool,
    /// Lambda parameter for equity calculation (default: 0.02)
    /// Controls the responsiveness of the equity curve
    #[serde(default = "default_lambda_param")]
    pub lambda_param: f64,
    /// Decay factor for equity weighting (default: 0.03)
    /// Controls how quickly past performance is forgotten
    #[serde(default = "default_decay")]
    pub decay: f64,
    /// Number of initial bars to cut out (default: 0)
    #[serde(default = "default_cutout")]
    pub cutout: usize,
    /// Minimum equity floor value to prevent numerical instability
    #[serde(default = "default_equity_floor")]
    pub equity_floor: f64,
    /// Configuration for each Moving Average type
    #[serde(default = "default_ma_configs")]
    pub ma_configs: Vec<MAConfig>,
}

fn default_lambda_param() -> f64 {
    DEFAULT_LAMBDA_PARAM
}
fn default_decay() -> f64 {
    DEFAULT_DECAY
}
fn default_cutout() -> usize {
    DEFAULT_CUTOUT
}
fn default_robustness() -> Robustness {
    Robustness::Medium
}
fn default_equity_floor() -> f64 {
    DEFAULT_EQUITY_FLOOR
}
fn default_ma_configs() -> Vec<MAConfig> {
    vec![
        MAConfig {
            ma_type: MAType::Ema,
            length: DEFAULT_MA_LENGTH,
            weight: DEFAULT_MA_WEIGHT,
        },
        MAConfig {
            ma_type: MAType::Hma,
            length: DEFAULT_MA_LENGTH,
            weight: DEFAULT_MA_WEIGHT,
        },
        MAConfig {
            ma_type: MAType::Wma,
            length: DEFAULT_MA_LENGTH,
            weight: DEFAULT_MA_WEIGHT,
        },
        MAConfig {
            ma_type: MAType::Dema,
            length: DEFAULT_MA_LENGTH,
            weight: DEFAULT_MA_WEIGHT,
        },
        MAConfig {
            ma_type: MAType::Lsma,
            length: DEFAULT_MA_LENGTH,
            weight: DEFAULT_MA_WEIGHT,
        },
        MAConfig {
            ma_type: MAType::Kama,
            length: DEFAULT_MA_LENGTH,
            weight: DEFAULT_MA_WEIGHT,
        },
    ]
}

/// Type of Moving Average
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum MAType {
    /// Exponential Moving Average
    Ema,
    /// Hull Moving Average
    Hma,
    /// Weighted Moving Average
    Wma,
    /// Double Exponential Moving Average
    Dema,
    /// Least Squares Moving Average
    Lsma,
    /// Kaufman Adaptive Moving Average
    Kama,
}

impl std::str::FromStr for MAType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "EMA" => Ok(MAType::Ema),
            "HMA" => Ok(MAType::Hma),
            "WMA" => Ok(MAType::Wma),
            "DEMA" => Ok(MAType::Dema),
            "LSMA" => Ok(MAType::Lsma),
            "KAMA" => Ok(MAType::Kama),
            _ => Err(format!("Unknown MA type: {}", s)),
        }
    }
}

impl std::fmt::Display for MAType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            MAType::Ema => "EMA",
            MAType::Hma => "HMA",
            MAType::Wma => "WMA",
            MAType::Dema => "DEMA",
            MAType::Lsma => "LSMA",
            MAType::Kama => "KAMA",
        };
        write!(f, "{}", s)
    }
}

/// Configuration for a single Moving Average type
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct MAConfig {
    /// Type of Moving Average
    pub ma_type: MAType,
    /// Base length for the MA calculation
    pub length: usize,
    /// Static weight for this MA type in aggregation
    pub weight: f64,
}

/// Error information for a failed symbol processing
///
/// Contains the symbol identifier and error message.
/// Used to track which symbols failed without failing the entire batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolError {
    /// Symbol that failed processing
    pub symbol: String,
    /// Error message describing what went wrong
    pub error: String,
}

/// Result of scanning a batch of symbols
///
/// Contains successful results, errors, and summary statistics.
#[derive(Debug, Serialize, Deserialize)]
pub struct ScanResult {
    /// Batch identifier (matches the input batch_id)
    pub batch_id: String,
    /// Successful signal results
    pub results: Vec<SignalResult>,
    /// Errors that occurred during processing
    pub errors: Vec<SymbolError>,
    /// Number of symbols processed successfully
    pub success_count: usize,
    /// Number of symbols that failed
    pub error_count: usize,
}

/// Signal result for a single symbol
///
/// Contains the final score, signal classification, and detailed breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalResult {
    /// Symbol identifier
    pub symbol: String,
    /// Final aggregated score (range: -1.0 to 1.0)
    pub score: f64,
    /// Signal classification
    pub signal_type: SignalType,
    /// Per-timeframe signal details
    pub details: HashMap<String, SignalType>,
    /// Per-timeframe signal strengths
    pub strengths: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::{ATCConfig, MAConfig, MAType, Robustness};
    use std::collections::HashMap;

    #[test]
    fn test_ma_type_deserialize_valid_uppercase() {
        let parsed: MAConfig =
            serde_json::from_str(r#"{"ma_type":"EMA","length":20,"weight":1.0}"#)
                .expect("EMA should deserialize successfully");

        assert_eq!(parsed.ma_type, MAType::Ema);
    }

    #[test]
    fn test_ma_type_deserialize_invalid_value() {
        let parse_result =
            serde_json::from_str::<MAConfig>(r#"{"ma_type":"BAD","length":20,"weight":1.0}"#);

        assert!(
            parse_result.is_err(),
            "BAD MA type must fail deserialization"
        );
    }

    #[test]
    fn test_atc_config_deserialize_invalid_robustness_value() {
        let parse_result = serde_json::from_str::<ATCConfig>(
            r#"{
                "robustness":"Bad",
                "weights":{},
                "threshold":0.3,
                "min_signal":0.0,
                "use_signal_strength":true,
                "lambda_param":0.02,
                "decay":0.03,
                "cutout":0,
                "equity_floor":0.25,
                "ma_configs":[{"ma_type":"EMA","length":20,"weight":1.0}]
            }"#,
        );

        assert!(
            parse_result.is_err(),
            "Invalid robustness value must fail deserialization"
        );
    }

    #[test]
    fn test_atc_config_deserialize_valid_robustness_value() {
        let parsed = serde_json::from_str::<ATCConfig>(
            r#"{
                "robustness":"Medium",
                "weights":{},
                "threshold":0.3,
                "min_signal":0.0,
                "use_signal_strength":true,
                "lambda_param":0.02,
                "decay":0.03,
                "cutout":0,
                "equity_floor":0.25,
                "ma_configs":[{"ma_type":"EMA","length":20,"weight":1.0}]
            }"#,
        )
        .expect("Valid robustness should deserialize successfully");

        assert_eq!(parsed.robustness, Robustness::Medium);
        assert_eq!(parsed.weights, HashMap::new());
    }
}
