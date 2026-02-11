//! # ATC Serverless - Adaptive Trend Classification
//!
//! A high-performance Rust implementation of the Adaptive Trend Classification (ATC) algorithm,
//! optimized for AWS Lambda serverless deployment.
//!
//! ## Overview
//!
//! This crate provides real-time trading signal detection for cryptocurrency markets with:
//! - 6 types of Moving Averages (EMA, HMA, WMA, DEMA, LSMA, KAMA)
//! - 8 length variations per MA type (via diflen) for robustness
//! - Layer 1 signal detection with equity-based weighting
//! - Multi-timeframe aggregation
//! - LONG/SHORT/NEUTRAL classifications with confidence scores
//!
//! ## Example
//!
//! ```rust
//! use atc_serverless::{ATCConfig, MAConfig, process_batch, SymbolData, OHLCVData};
//! use std::collections::HashMap;
//!
//! // Create configuration
//! let config = ATCConfig {
//!     weights: HashMap::new(),
//!     threshold: 0.3,
//!     min_signal: 0.0,
//!     use_signal_strength: true,
//!     robustness: "Medium".to_string(),
//!     lambda_param: 0.02,
//!     decay: 0.03,
//!     cutout: 0,
//!     equity_floor: 0.25,
//!     ma_configs: vec![
//!         MAConfig { ma_type: "EMA".to_string(), length: 20, weight: 1.0 },
//!     ],
//! };
//! ```

#![warn(missing_docs)]

/// Batch processing and error recovery
pub mod aggregation;
/// Equity curve calculations for Layer 2 weighting
pub mod equity;
/// Moving Average calculations (EMA, HMA, WMA, DEMA, LSMA, KAMA)
pub mod ma_calculations;
/// Multi-timeframe signal aggregation and voting
pub mod multi_tf_voting;
/// Signal detection algorithms with diflen and trend classification
pub mod signal_detection;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export common types
pub use aggregation::*;
pub use equity::*;
pub use ma_calculations::*;
pub use multi_tf_voting::*;
pub use signal_detection::*;

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
///   "symbols": [...],
///   "config": {...}
/// }
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchRequest {
    /// Unique identifier for this batch (used for tracking and logging)
    pub batch_id: String,
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
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OHLCVData {
    /// Unix timestamps for each bar
    pub timestamp: Vec<i64>,
    /// Opening prices
    pub open: Vec<f64>,
    /// High prices
    pub high: Vec<f64>,
    /// Low prices
    pub low: Vec<f64>,
    /// Closing prices
    pub close: Vec<f64>,
    /// Trading volumes
    pub volume: Vec<f64>,
}

/// Configuration for the ATC algorithm
///
/// Controls the behavior of signal detection including thresholds,
/// timeframe weights, and MA configurations.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ATCConfig {
    /// Robustness level for diflen calculation: "Narrow", "Medium", or "Wide"
    #[serde(default = "default_robustness")]
    pub robustness: String,
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
    0.02
}
fn default_decay() -> f64 {
    0.03
}
fn default_cutout() -> usize {
    0
}
fn default_robustness() -> String {
    "Medium".to_string()
}
fn default_equity_floor() -> f64 {
    0.25
}
fn default_ma_configs() -> Vec<MAConfig> {
    vec![
        MAConfig {
            ma_type: "EMA".to_string(),
            length: 28,
            weight: 1.0,
        },
        MAConfig {
            ma_type: "HMA".to_string(),
            length: 28,
            weight: 1.0,
        },
        MAConfig {
            ma_type: "WMA".to_string(),
            length: 28,
            weight: 1.0,
        },
        MAConfig {
            ma_type: "DEMA".to_string(),
            length: 28,
            weight: 1.0,
        },
        MAConfig {
            ma_type: "LSMA".to_string(),
            length: 28,
            weight: 1.0,
        },
        MAConfig {
            ma_type: "KAMA".to_string(),
            length: 28,
            weight: 1.0,
        },
    ]
}

/// Configuration for a single Moving Average type
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MAConfig {
    /// Type of Moving Average ("EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA")
    pub ma_type: String,
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
    /// Signal classification ("LONG", "SHORT", "NEUTRAL")
    pub signal_type: String,
    /// Per-timeframe signal details
    pub details: HashMap<String, String>,
    /// Per-timeframe signal strengths
    pub strengths: HashMap<String, f64>,
}
