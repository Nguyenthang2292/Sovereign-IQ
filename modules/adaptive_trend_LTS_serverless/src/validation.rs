//! Input validation for ATC serverless module
//!
//! This module provides validation functions for OHLCV data, ATC configuration,
//! and schema versioning.

use crate::constants::{
    MAX_BARS_PER_TIMEFRAME, MAX_BATCH_SIZE, MAX_MA_LENGTH, MAX_NORMALIZED_VALUE, MIN_DATA_LENGTH,
    MIN_NORMALIZED_VALUE, MIN_SIGNAL_VALUE, WEIGHT_SUM_TOLERANCE,
};
use crate::{ATCConfig, BatchRequest, OHLCVData};

/// Validation error types
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// OHLCV data validation errors
    Ohlcv {
        /// Name of the OHLCV field that failed validation
        field: String,
        /// Human-readable validation failure details
        message: String,
        /// Optional symbol identifier associated with the invalid data
        symbol: Option<String>,
    },
    /// Configuration validation errors
    Config {
        /// Name of the configuration field that failed validation
        field: String,
        /// Human-readable validation failure details
        message: String,
    },
    /// Schema version errors
    Schema {
        /// Human-readable schema compatibility failure details
        message: String,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::Ohlcv {
                field,
                message,
                symbol,
            } => {
                if let Some(s) = symbol {
                    write!(
                        f,
                        "OHLCV validation error for symbol {}: {} - {}",
                        s, field, message
                    )
                } else {
                    write!(f, "OHLCV validation error: {} - {}", field, message)
                }
            }
            ValidationError::Config { field, message } => {
                write!(f, "Config validation error: {} - {}", field, message)
            }
            ValidationError::Schema { message } => {
                write!(f, "Schema validation error: {}", message)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Result type for validation
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Validate OHLCV data for a single symbol
///
/// Checks:
/// - Timestamp ordering (must be monotonically increasing)
/// - No negative prices (open, high, low, close)
/// - Volume must be non-negative
/// - Arrays must have matching lengths
pub fn validate_ohlcv_data(data: &OHLCVData, symbol: &str) -> ValidationResult<()> {
    let len = data.timestamp.len();

    if len == 0 {
        return Err(ValidationError::Ohlcv {
            field: "timestamp".to_string(),
            message: "OHLCV data cannot be empty".to_string(),
            symbol: Some(symbol.to_string()),
        });
    }

    if len < MIN_DATA_LENGTH {
        return Err(ValidationError::Ohlcv {
            field: "timestamp".to_string(),
            message: format!(
                "OHLCV data length {} is below minimum required length {}",
                len, MIN_DATA_LENGTH
            ),
            symbol: Some(symbol.to_string()),
        });
    }

    if len > MAX_BARS_PER_TIMEFRAME {
        return Err(ValidationError::Ohlcv {
            field: "timestamp".to_string(),
            message: format!(
                "OHLCV data length {} exceeds maximum allowed length {}",
                len, MAX_BARS_PER_TIMEFRAME
            ),
            symbol: Some(symbol.to_string()),
        });
    }

    let arrays = [
        ("open", &data.open),
        ("high", &data.high),
        ("low", &data.low),
        ("close", &data.close),
        ("volume", &data.volume),
    ];

    for (name, arr) in &arrays {
        if arr.len() != len {
            return Err(ValidationError::Ohlcv {
                field: name.to_string(),
                message: format!(
                    "Array length {} does not match timestamp length {}",
                    arr.len(),
                    len
                ),
                symbol: Some(symbol.to_string()),
            });
        }
    }

    for i in 1..len {
        if data.timestamp[i] <= data.timestamp[i - 1] {
            return Err(ValidationError::Ohlcv {
                field: "timestamp".to_string(),
                message: format!(
                    "Timestamp at index {} ({}) is not greater than previous ({})",
                    i,
                    data.timestamp[i],
                    data.timestamp[i - 1]
                ),
                symbol: Some(symbol.to_string()),
            });
        }
    }

    for (name, arr) in &[
        ("open", &data.open),
        ("high", &data.high),
        ("low", &data.low),
        ("close", &data.close),
    ] {
        for (i, &price) in arr.iter().enumerate() {
            if price.is_nan() {
                return Err(ValidationError::Ohlcv {
                    field: name.to_string(),
                    message: format!("NaN value found at index {}", i),
                    symbol: Some(symbol.to_string()),
                });
            }
            // OHLC prices must be strictly positive (not zero)
            if price <= MIN_NORMALIZED_VALUE {
                return Err(ValidationError::Ohlcv {
                    field: name.to_string(),
                    message: format!("Price must be positive, got {} at index {}", price, i),
                    symbol: Some(symbol.to_string()),
                });
            }
        }
    }

    for (i, &vol) in data.volume.iter().enumerate() {
        if vol.is_nan() {
            return Err(ValidationError::Ohlcv {
                field: "volume".to_string(),
                message: format!("NaN volume found at index {}", i),
                symbol: Some(symbol.to_string()),
            });
        }
        if vol < MIN_NORMALIZED_VALUE {
            return Err(ValidationError::Ohlcv {
                field: "volume".to_string(),
                message: format!("Negative volume {} found at index {}", vol, i),
                symbol: Some(symbol.to_string()),
            });
        }
    }

    for (i, &high) in data.high.iter().enumerate() {
        if high < data.low[i] {
            return Err(ValidationError::Ohlcv {
                field: "high/low".to_string(),
                message: format!(
                    "High ({}) is less than low ({}) at index {}",
                    high, data.low[i], i
                ),
                symbol: Some(symbol.to_string()),
            });
        }
    }

    for (i, &high) in data.high.iter().enumerate() {
        if high < data.open[i] || high < data.close[i] {
            return Err(ValidationError::Ohlcv {
                field: "high".to_string(),
                message: format!(
                    "High ({}) is less than open ({}) or close ({}) at index {}",
                    high, data.open[i], data.close[i], i
                ),
                symbol: Some(symbol.to_string()),
            });
        }
    }

    for (i, &low) in data.low.iter().enumerate() {
        if low > data.open[i] || low > data.close[i] {
            return Err(ValidationError::Ohlcv {
                field: "low".to_string(),
                message: format!(
                    "Low ({}) is greater than open ({}) or close ({}) at index {}",
                    low, data.open[i], data.close[i], i
                ),
                symbol: Some(symbol.to_string()),
            });
        }
    }

    Ok(())
}

/// Validate ATC configuration
///
/// Checks:
/// - Threshold is between 0.0 and 1.0
/// - MA lengths are positive
/// - Weights sum to approximately 1.0 (with tolerance)
pub fn validate_config(config: &ATCConfig) -> ValidationResult<()> {
    if config.threshold < MIN_NORMALIZED_VALUE || config.threshold > MAX_NORMALIZED_VALUE {
        return Err(ValidationError::Config {
            field: "threshold".to_string(),
            message: format!(
                "Threshold must be between 0.0 and 1.0, got {}",
                config.threshold
            ),
        });
    }

    // min_signal = 0.0 is valid and means "no minimum filter is applied"
    if config.min_signal < MIN_SIGNAL_VALUE || config.min_signal > MAX_NORMALIZED_VALUE {
        return Err(ValidationError::Config {
            field: "min_signal".to_string(),
            message: format!(
                "min_signal must be between 0.0 and 1.0, got {}",
                config.min_signal
            ),
        });
    }

    if config.lambda_param <= MIN_NORMALIZED_VALUE || config.lambda_param > MAX_NORMALIZED_VALUE {
        return Err(ValidationError::Config {
            field: "lambda_param".to_string(),
            message: format!(
                "lambda_param must be between 0.0 and 1.0, got {}",
                config.lambda_param
            ),
        });
    }

    if config.decay <= MIN_NORMALIZED_VALUE || config.decay > MAX_NORMALIZED_VALUE {
        return Err(ValidationError::Config {
            field: "decay".to_string(),
            message: format!("decay must be between 0.0 and 1.0, got {}", config.decay),
        });
    }

    if config.equity_floor <= MIN_NORMALIZED_VALUE || config.equity_floor > MAX_NORMALIZED_VALUE {
        return Err(ValidationError::Config {
            field: "equity_floor".to_string(),
            message: format!(
                "equity_floor must be between 0.0 and 1.0, got {}",
                config.equity_floor
            ),
        });
    }

    if config.ma_configs.is_empty() {
        return Err(ValidationError::Config {
            field: "ma_configs".to_string(),
            message: "ma_configs cannot be empty".to_string(),
        });
    }

    if config.weights.is_empty() {
        return Err(ValidationError::Config {
            field: "weights".to_string(),
            message: "weights cannot be empty".to_string(),
        });
    }

    let mut has_positive_ma_weight = false;
    for (i, ma_config) in config.ma_configs.iter().enumerate() {
        if ma_config.length == 0 {
            return Err(ValidationError::Config {
                field: "ma_configs".to_string(),
                message: format!("MA length at index {} must be positive, got 0", i),
            });
        }

        if ma_config.length > MAX_MA_LENGTH {
            return Err(ValidationError::Config {
                field: "ma_configs".to_string(),
                message: format!(
                    "MA length at index {} is too large (max {}), got {}",
                    i, MAX_MA_LENGTH, ma_config.length
                ),
            });
        }

        if ma_config.weight < 0.0 {
            return Err(ValidationError::Config {
                field: "ma_configs".to_string(),
                message: format!(
                    "MA weight at index {} cannot be negative, got {}",
                    i, ma_config.weight
                ),
            });
        }

        if ma_config.weight > 0.0 {
            has_positive_ma_weight = true;
        }
    }

    if !has_positive_ma_weight {
        return Err(ValidationError::Config {
            field: "ma_configs".to_string(),
            message: "At least one MA config must have weight > 0.0".to_string(),
        });
    }

    for (timeframe, weight) in &config.weights {
        if timeframe.trim().is_empty() {
            return Err(ValidationError::Config {
                field: "weights".to_string(),
                message: "Timeframe keys in weights cannot be empty".to_string(),
            });
        }

        if !weight.is_finite() || *weight < 0.0 {
            return Err(ValidationError::Config {
                field: "weights".to_string(),
                message: format!(
                    "Weight for timeframe '{}' must be finite and non-negative, got {}",
                    timeframe, weight
                ),
            });
        }
    }

    let weight_sum: f64 = config.weights.values().sum();
    if (weight_sum - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
        return Err(ValidationError::Config {
            field: "weights".to_string(),
            message: format!(
                "Weights must sum to 1.0 (with tolerance {}), got {}",
                WEIGHT_SUM_TOLERANCE, weight_sum
            ),
        });
    }

    Ok(())
}

/// Current schema version
pub const SCHEMA_VERSION: &str = "1.0.0";

/// Validate batch request schema version
///
/// Returns Ok if:
/// - Version field is absent (backwards compatible)
/// - Version is compatible (major version matches)
pub fn validate_schema_version(request: &BatchRequest) -> ValidationResult<()> {
    if let Some(ref version) = request.version {
        let raw_segments: Vec<&str> = version.split('.').collect();
        let mut parts = Vec::with_capacity(raw_segments.len());
        for segment in &raw_segments {
            match segment.parse::<u32>() {
                Ok(v) => parts.push(v),
                Err(_) => {
                    return Err(ValidationError::Schema {
                        message: format!(
                            "Invalid version format: '{}' (non-numeric segment '{}')",
                            version, segment
                        ),
                    });
                }
            }
        }

        if parts.is_empty() {
            return Err(ValidationError::Schema {
                message: format!("Invalid version format: '{}'", version),
            });
        }

        let major = parts[0];
        let schema_parts: Vec<u32> = SCHEMA_VERSION
            .split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        let schema_major = schema_parts.first().copied().unwrap_or(0);

        if major != schema_major {
            return Err(ValidationError::Schema {
                message: format!(
                    "Incompatible schema version: request uses '{}', expected major version {}",
                    version, schema_major
                ),
            });
        }
    }

    Ok(())
}

/// Validate a batch request completely
///
/// This includes:
/// - OHLCV data validation for all symbols
/// - Configuration validation
/// - Schema version validation
pub fn validate_batch_request(request: &BatchRequest) -> ValidationResult<()> {
    validate_schema_version(request)?;

    validate_config(&request.config)?;

    if request.symbols.len() > MAX_BATCH_SIZE {
        return Err(ValidationError::Config {
            field: "symbols".to_string(),
            message: format!(
                "Batch size {} exceeds maximum allowed {}",
                request.symbols.len(),
                MAX_BATCH_SIZE
            ),
        });
    }

    for symbol in &request.symbols {
        if symbol.symbol.is_empty() {
            return Err(ValidationError::Ohlcv {
                field: "symbol".to_string(),
                message: "Symbol identifier cannot be empty".to_string(),
                symbol: None,
            });
        }

        if symbol.timeframes.is_empty() {
            return Err(ValidationError::Ohlcv {
                field: "timeframes".to_string(),
                message: "Symbol must have at least one timeframe".to_string(),
                symbol: Some(symbol.symbol.clone()),
            });
        }

        for (timeframe, ohlcv) in &symbol.timeframes {
            if timeframe.is_empty() {
                return Err(ValidationError::Ohlcv {
                    field: "timeframe".to_string(),
                    message: "Timeframe identifier cannot be empty".to_string(),
                    symbol: Some(symbol.symbol.clone()),
                });
            }
            validate_ohlcv_data(ohlcv, &symbol.symbol).map_err(|err| match err {
                ValidationError::Ohlcv {
                    field,
                    message,
                    symbol: error_symbol,
                } => ValidationError::Ohlcv {
                    field: format!("{} (timeframe: {})", field, timeframe),
                    message,
                    symbol: error_symbol.or_else(|| Some(symbol.symbol.clone())),
                },
                other => other,
            })?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MAConfig, SymbolData};
    use std::collections::HashMap;

    fn create_test_ohlcv() -> OHLCVData {
        OHLCVData {
            timestamp: vec![1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
                .into_boxed_slice(),
            open: vec![
                100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
            ]
            .into_boxed_slice(),
            high: vec![
                105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0,
            ]
            .into_boxed_slice(),
            low: vec![
                99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
            ]
            .into_boxed_slice(),
            close: vec![
                104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0,
            ]
            .into_boxed_slice(),
            volume: vec![
                1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0,
            ]
            .into_boxed_slice(),
        }
    }

    #[test]
    fn test_validate_ohlcv_valid() {
        let data = create_test_ohlcv();
        let result = validate_ohlcv_data(&data, "BTCUSDT");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_ohlcv_empty() {
        let data = OHLCVData {
            timestamp: vec![].into_boxed_slice(),
            open: vec![].into_boxed_slice(),
            high: vec![].into_boxed_slice(),
            low: vec![].into_boxed_slice(),
            close: vec![].into_boxed_slice(),
            volume: vec![].into_boxed_slice(),
        };
        let result = validate_ohlcv_data(&data, "BTCUSDT");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ohlcv_too_short() {
        let data = OHLCVData {
            timestamp: vec![1000, 2000].into_boxed_slice(),
            open: vec![100.0, 101.0].into_boxed_slice(),
            high: vec![101.0, 102.0].into_boxed_slice(),
            low: vec![99.0, 100.0].into_boxed_slice(),
            close: vec![100.5, 101.5].into_boxed_slice(),
            volume: vec![1000.0, 1100.0].into_boxed_slice(),
        };
        let result = validate_ohlcv_data(&data, "BTCUSDT");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ohlcv_too_long() {
        let len = MAX_BARS_PER_TIMEFRAME + 1;

        let data = OHLCVData {
            timestamp: (0..len).map(|i| i as i64 + 1).collect::<Vec<_>>().into_boxed_slice(),
            open: vec![100.0; len].into_boxed_slice(),
            high: vec![101.0; len].into_boxed_slice(),
            low: vec![99.0; len].into_boxed_slice(),
            close: vec![100.5; len].into_boxed_slice(),
            volume: vec![1000.0; len].into_boxed_slice(),
        };

        let result = validate_ohlcv_data(&data, "BTCUSDT");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ohlcv_negative_price() {
        let mut data = create_test_ohlcv();
        data.close[2] = -10.0;
        let result = validate_ohlcv_data(&data, "BTCUSDT");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ohlcv_negative_volume() {
        let mut data = create_test_ohlcv();
        data.volume[2] = -100.0;
        let result = validate_ohlcv_data(&data, "BTCUSDT");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ohlcv_timestamp_order() {
        let mut data = create_test_ohlcv();
        data.timestamp[3] = 1000;
        let result = validate_ohlcv_data(&data, "BTCUSDT");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_valid() {
        let config = ATCConfig {
            robustness: crate::Robustness::Medium,
            weights: {
                let mut w = HashMap::new();
                w.insert("1h".to_string(), 1.0);
                w
            },
            threshold: 0.5,
            min_signal: 0.1,
            use_signal_strength: true,
            lambda_param: 0.02,
            decay: 0.03,
            cutout: 0,
            equity_floor: 0.25,
            ma_configs: vec![MAConfig {
                ma_type: crate::MAType::Ema,
                length: 20,
                weight: 1.0,
            }],
        };
        let result = validate_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_config_min_signal_zero_valid() {
        let config = ATCConfig {
            robustness: crate::Robustness::Medium,
            weights: {
                let mut w = HashMap::new();
                w.insert("1h".to_string(), 1.0);
                w
            },
            threshold: 0.5,
            min_signal: 0.0, // zero is valid
            use_signal_strength: true,
            lambda_param: 0.02,
            decay: 0.03,
            cutout: 0,
            equity_floor: 0.25,
            ma_configs: vec![MAConfig {
                ma_type: crate::MAType::Ema,
                length: 20,
                weight: 1.0,
            }],
        };
        let result = validate_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_config_threshold() {
        let mut config = ATCConfig {
            robustness: crate::Robustness::Medium,
            weights: HashMap::new(),
            threshold: 1.5,
            min_signal: 0.1,
            use_signal_strength: true,
            lambda_param: 0.02,
            decay: 0.03,
            cutout: 0,
            equity_floor: 0.25,
            ma_configs: vec![],
        };
        assert!(validate_config(&config).is_err());

        config.threshold = -0.1;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_weights_sum() {
        let config = ATCConfig {
            robustness: crate::Robustness::Medium,
            weights: {
                let mut w = HashMap::new();
                w.insert("1h".to_string(), 0.6);
                w.insert("4h".to_string(), 0.6);
                w
            },
            threshold: 0.5,
            min_signal: 0.1,
            use_signal_strength: true,
            lambda_param: 0.02,
            decay: 0.03,
            cutout: 0,
            equity_floor: 0.25,
            ma_configs: vec![MAConfig {
                ma_type: crate::MAType::Ema,
                length: 20,
                weight: 1.0,
            }],
        };
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_ma_length() {
        let config = ATCConfig {
            robustness: crate::Robustness::Medium,
            weights: {
                let mut w = HashMap::new();
                w.insert("1h".to_string(), 1.0);
                w
            },
            threshold: 0.5,
            min_signal: 0.1,
            use_signal_strength: true,
            lambda_param: 0.02,
            decay: 0.03,
            cutout: 0,
            equity_floor: 0.25,
            ma_configs: vec![MAConfig {
                ma_type: crate::MAType::Ema,
                length: 0,
                weight: 1.0,
            }],
        };
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_config_rejects_empty_weights() {
        let config = ATCConfig {
            robustness: crate::Robustness::Medium,
            weights: HashMap::new(),
            threshold: 0.5,
            min_signal: 0.1,
            use_signal_strength: true,
            lambda_param: 0.02,
            decay: 0.03,
            cutout: 0,
            equity_floor: 0.25,
            ma_configs: vec![MAConfig {
                ma_type: crate::MAType::Ema,
                length: 20,
                weight: 1.0,
            }],
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_config_rejects_empty_ma_configs() {
        let config = ATCConfig {
            robustness: crate::Robustness::Medium,
            weights: {
                let mut w = HashMap::new();
                w.insert("1h".to_string(), 1.0);
                w
            },
            threshold: 0.5,
            min_signal: 0.1,
            use_signal_strength: true,
            lambda_param: 0.02,
            decay: 0.03,
            cutout: 0,
            equity_floor: 0.25,
            ma_configs: vec![],
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_validate_schema_version_compatible() {
        let request = BatchRequest {
            batch_id: "test".to_string(),
            version: Some("1.0.0".to_string()),
            symbols: vec![],
            config: ATCConfig {
                robustness: crate::Robustness::Medium,
                weights: HashMap::new(),
                threshold: 0.5,
                min_signal: 0.1,
                use_signal_strength: true,
                lambda_param: 0.02,
                decay: 0.03,
                cutout: 0,
                equity_floor: 0.25,
                ma_configs: vec![],
            },
        };
        let result = validate_schema_version(&request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_schema_version_incompatible() {
        let request = BatchRequest {
            batch_id: "test".to_string(),
            version: Some("2.0.0".to_string()),
            symbols: vec![],
            config: ATCConfig {
                robustness: crate::Robustness::Medium,
                weights: HashMap::new(),
                threshold: 0.5,
                min_signal: 0.1,
                use_signal_strength: true,
                lambda_param: 0.02,
                decay: 0.03,
                cutout: 0,
                equity_floor: 0.25,
                ma_configs: vec![],
            },
        };
        let result = validate_schema_version(&request);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_batch_request() {
        let request = BatchRequest {
            batch_id: "test".to_string(),
            version: Some("1.0.0".to_string()),
            symbols: vec![SymbolData {
                symbol: "BTCUSDT".to_string(),
                timeframes: {
                    let mut tf = HashMap::new();
                    tf.insert("1h".to_string(), create_test_ohlcv());
                    tf
                },
            }],
            config: ATCConfig {
                robustness: crate::Robustness::Medium,
                weights: {
                    let mut w = HashMap::new();
                    w.insert("1h".to_string(), 1.0);
                    w
                },
                threshold: 0.5,
                min_signal: 0.1,
                use_signal_strength: true,
                lambda_param: 0.02,
                decay: 0.03,
                cutout: 0,
                equity_floor: 0.25,
                ma_configs: vec![MAConfig {
                    ma_type: crate::MAType::Ema,
                    length: 20,
                    weight: 1.0,
                }],
            },
        };
        let result = validate_batch_request(&request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_batch_request_includes_timeframe_in_error_context() {
        let mut invalid_ohlcv = create_test_ohlcv();
        invalid_ohlcv.open = vec![100.0, 101.0].into_boxed_slice();

        let request = BatchRequest {
            batch_id: "test".to_string(),
            version: Some("1.0.0".to_string()),
            symbols: vec![SymbolData {
                symbol: "BTCUSDT".to_string(),
                timeframes: {
                    let mut tf = HashMap::new();
                    tf.insert("1h".to_string(), invalid_ohlcv);
                    tf
                },
            }],
            config: ATCConfig {
                robustness: crate::Robustness::Medium,
                weights: {
                    let mut w = HashMap::new();
                    w.insert("1h".to_string(), 1.0);
                    w
                },
                threshold: 0.5,
                min_signal: 0.1,
                use_signal_strength: true,
                lambda_param: 0.02,
                decay: 0.03,
                cutout: 0,
                equity_floor: 0.25,
                ma_configs: vec![MAConfig {
                    ma_type: crate::MAType::Ema,
                    length: 20,
                    weight: 1.0,
                }],
            },
        };

        let result = validate_batch_request(&request);
        match result {
            Err(ValidationError::Ohlcv { field, .. }) => {
                assert!(field.contains("timeframe: 1h"));
            }
            _ => panic!("Expected OHLCV validation error with timeframe context"),
        }
    }

    #[test]
    fn test_validate_batch_request_exceeds_max_batch_size() {
        let symbols = (0..(MAX_BATCH_SIZE + 1))
            .map(|idx| SymbolData {
                symbol: format!("SYM{}", idx),
                timeframes: {
                    let mut tf = HashMap::new();
                    tf.insert("1h".to_string(), create_test_ohlcv());
                    tf
                },
            })
            .collect::<Vec<_>>();

        let request = BatchRequest {
            batch_id: "test".to_string(),
            version: Some("1.0.0".to_string()),
            symbols,
            config: ATCConfig {
                robustness: crate::Robustness::Medium,
                weights: {
                    let mut w = HashMap::new();
                    w.insert("1h".to_string(), 1.0);
                    w
                },
                threshold: 0.5,
                min_signal: 0.1,
                use_signal_strength: true,
                lambda_param: 0.02,
                decay: 0.03,
                cutout: 0,
                equity_floor: 0.25,
                ma_configs: vec![MAConfig {
                    ma_type: crate::MAType::Ema,
                    length: 20,
                    weight: 1.0,
                }],
            },
        };

        let result = validate_batch_request(&request);
        assert!(result.is_err());
    }
}
