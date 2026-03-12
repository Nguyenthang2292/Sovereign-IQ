//! Constants module for ATC Serverless
//!
//! This module centralizes all algorithm constants with documentation
//! explaining their purpose and rationale.

/// Default threshold for LONG/SHORT signal classification.
///
/// Signals with absolute value greater than this threshold are classified
/// as LONG or SHORT. Values below are NEUTRAL.
///
/// **Rationale**: 0.3 provides a balanced classification that reduces
/// noise while capturing significant trends. Based on backtesting on
/// cryptocurrency markets (BTC, ETH, etc.).
pub const DEFAULT_THRESHOLD: f64 = 0.3;

/// Minimum signal strength to consider for classification.
///
/// If the weighted score is below this value, the signal is forced to NEUTRAL.
/// This helps filter out weak signals.
///
/// **Rationale**: 0.0 means all signals above threshold are considered.
/// Increase to filter out more noise.
pub const DEFAULT_MIN_SIGNAL: f64 = 0.0;

/// The absolute minimum value allowed for min_signal configuration.
/// 0.0 is valid and means no minimum filter is applied.
pub const MIN_SIGNAL_VALUE: f64 = 0.0;

/// Lambda parameter for equity calculation.
///
/// Controls the exponential growth factor applied to returns over time.
/// Higher values make recent returns more important.
///
/// **Formula**: `exp(lambda * (bar_index - cutout))`
/// **Rationale**: 0.02 is the standard value from the original ATC algorithm.
/// It's scaled down by 1000 internally to 0.00002 for numerical stability.
pub const DEFAULT_LAMBDA_PARAM: f64 = 0.02;

/// Decay factor for equity weighting.
///
/// Controls how quickly past performance is forgotten in the equity curve.
/// Higher values mean faster decay (more weight on recent performance).
///
/// **Formula**: `equity[i] = equity[i-1] * decay_multiplier * (1 + return)`
/// **Rationale**: 0.03 means ~3% decay per bar, providing balanced
/// memory of past performance.
pub const DEFAULT_DECAY: f64 = 0.03;

/// Number of initial bars to cut out from calculation.
///
/// Bars before this index are excluded from equity calculation.
/// Useful for skipping unstable early data.
///
/// **Rationale**: 0 includes all bars. Use >0 to skip initial volatile period.
pub const DEFAULT_CUTOUT: usize = 0;

/// Minimum equity floor value.
///
/// Prevents numerical instability by clamping equity to this minimum.
/// Equity can grow very small (approaching 0) in losing streaks.
///
/// **Rationale**: 0.25 means equity never drops below 25% of starting value.
/// This prevents numerical underflow while allowing significant drawdowns.
pub const DEFAULT_EQUITY_FLOOR: f64 = 0.25;

/// Default robustness level for diflen calculation.
///
/// Determines the range of length variations around the base length.
///
/// - `Narrow`: ±1, ±2, ±3, ±4 (min length: 5)
/// - `Medium`: ±1, ±2, ±4, ±6 (min length: 7) - DEFAULT
/// - `Wide`: ±1, ±3, ±5, ±7 (min length: 8)
pub const DEFAULT_ROBUSTNESS: &str = "Medium";

/// Default length for MA calculations.
///
/// Used when no specific length is provided in MAConfig.
///
/// **Rationale**: 28 is derived from Fibonacci sequence (21+34)/2
/// and commonly used in trading systems.
pub const DEFAULT_MA_LENGTH: usize = 28;

/// Default weight for MA types in aggregation.
///
/// All MA types have equal weight by default.
///
/// **Rationale**: Equal weighting prevents bias toward any specific
/// MA type until backtesting suggests otherwise.
pub const DEFAULT_MA_WEIGHT: f64 = 1.0;

/// Maximum allowed MA length in validation.
///
/// Prevents unreasonably large windows that can degrade performance
/// and cause excessive memory usage.
pub const MAX_MA_LENGTH: usize = 10_000;

/// Minimum normalized value for bounded configuration fields.
///
/// Used for fields constrained to [0.0, 1.0].
pub const MIN_NORMALIZED_VALUE: f64 = 0.0;

/// Maximum normalized value for bounded configuration fields.
///
/// Used for fields constrained to [0.0, 1.0].
pub const MAX_NORMALIZED_VALUE: f64 = 1.0;

/// Tolerance for checking timeframe weight sum against 1.0.
pub const WEIGHT_SUM_TOLERANCE: f64 = 0.001;

/// Valid robustness names accepted by configuration parsing.
pub const VALID_ROBUSTNESS_LEVELS: [&str; 3] = ["Narrow", "Medium", "Wide"];

/// Starting equity for equity curve calculation.
///
/// Initial value used at the first bar (after cutout).
///
/// **Rationale**: 1.0 represents 100% of initial capital.
pub const STARTING_EQUITY: f64 = 1.0;

/// Default growth value before cutout index.
///
/// All bars before cutout use this value for the growth factor.
///
/// **Rationale**: 1.0 means no exponential adjustment before cutout.
pub const DEFAULT_GROWTH_VALUE: f64 = 1.0;

/// Signal value for LONG classification.
///
/// Returned when price > MA (uptrend).
pub const SIGNAL_LONG: f64 = 1.0;

/// Signal value for SHORT classification.
///
/// Returned when price < MA (downtrend).
pub const SIGNAL_SHORT: f64 = -1.0;

/// Signal value for NEUTRAL classification.
///
/// Returned when no clear trend or price == MA.
pub const SIGNAL_NEUTRAL: f64 = 0.0;

/// Maximum batch size for processing.
///
/// Limits the number of symbols per batch to keep serialized SQS payloads
/// under the 256KB hard limit and to avoid memory spikes.
///
/// **Rationale**: 1000 symbols gives safe headroom for JSON payload growth
/// (results + errors + metadata) while retaining high throughput per invocation.
pub const MAX_BATCH_SIZE: usize = 1000;

/// Maximum bars allowed per timeframe in request payload.
///
/// Prevents oversized requests from exhausting Lambda memory when callers send
/// extremely long OHLCV histories.
pub const MAX_BARS_PER_TIMEFRAME: usize = 1000;

/// Minimum price value allowed in validation.
///
/// Prices below this are considered invalid.
pub const MIN_PRICE: f64 = 0.0;

/// Maximum price value allowed in validation.
///
/// Prices above this are considered invalid (prevents data errors).
pub const MAX_PRICE: f64 = 1e12;

/// Minimum data length required for calculations.
///
/// Bars below this length cannot produce meaningful signals.
///
/// **Rationale**: Need at least 10 bars for any MA calculation
/// to produce stable results.
pub const MIN_DATA_LENGTH: usize = 10;

/// Minimum length required for Narrow robustness.
///
/// Base lengths below this cannot produce valid diflen variations.
pub const MIN_LENGTH_NARROW: usize = 5;

/// Minimum length required for Medium robustness.
///
/// Base lengths below this cannot produce valid diflen variations.
pub const MIN_LENGTH_MEDIUM: usize = 7;

/// Minimum length required for Wide robustness.
///
/// Base lengths below this cannot produce valid diflen variations.
pub const MIN_LENGTH_WIDE: usize = 8;

/// Lambda scaling factor.
///
/// Lambda parameter is divided by this to convert to internal representation.
///
/// **Rationale**: 1000 converts 0.02 to 0.00002 for numerical stability
/// in exponential calculations over many bars.
pub const LAMBDA_SCALE: f64 = 1000.0;

/// Decay scaling factor.
///
/// Decay is divided by this to convert to internal representation.
///
/// **Rationale**: 100 converts 0.03 to 0.0003 for numerical stability
/// in recursive equity calculations.
pub const DECAY_SCALE: f64 = 100.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values_are_valid() {
        assert!(DEFAULT_THRESHOLD > 0.0 && DEFAULT_THRESHOLD <= 1.0);
        assert!(DEFAULT_MIN_SIGNAL >= 0.0);
        assert!(DEFAULT_LAMBDA_PARAM > 0.0);
        assert!(DEFAULT_DECAY > 0.0 && DEFAULT_DECAY < 1.0);
        assert!(DEFAULT_EQUITY_FLOOR > 0.0 && DEFAULT_EQUITY_FLOOR <= 1.0);
    }

    #[test]
    fn test_signal_values() {
        assert_eq!(SIGNAL_LONG, 1.0);
        assert_eq!(SIGNAL_SHORT, -1.0);
        assert_eq!(SIGNAL_NEUTRAL, 0.0);
    }

    #[test]
    fn test_min_lengths() {
        assert!(MIN_LENGTH_NARROW < MIN_LENGTH_MEDIUM);
        assert!(MIN_LENGTH_MEDIUM < MIN_LENGTH_WIDE);
    }

    #[test]
    fn test_scale_factors() {
        assert_eq!(LAMBDA_SCALE, 1000.0);
        assert_eq!(DECAY_SCALE, 100.0);
    }

    #[test]
    fn test_equity_floor_bounds() {
        assert!(DEFAULT_EQUITY_FLOOR > 0.0, "Equity floor must be positive");
        assert!(
            DEFAULT_EQUITY_FLOOR <= 1.0,
            "Equity floor cannot exceed 1.0"
        );
    }

    #[test]
    fn test_threshold_bounds() {
        assert!(DEFAULT_THRESHOLD > 0.0, "Threshold must be positive");
        assert!(DEFAULT_THRESHOLD <= 1.0, "Threshold cannot exceed 1.0");
    }

    #[test]
    fn test_lambda_param_valid() {
        assert!(DEFAULT_LAMBDA_PARAM > 0.0, "Lambda must be positive");
        assert!(DEFAULT_LAMBDA_PARAM <= 1.0, "Lambda cannot exceed 1.0");
    }

    #[test]
    fn test_decay_valid() {
        assert!(DEFAULT_DECAY > 0.0, "Decay must be positive");
        assert!(DEFAULT_DECAY < 1.0, "Decay must be less than 1.0");
    }

    #[test]
    fn test_ma_length_reasonable() {
        assert!(DEFAULT_MA_LENGTH >= 5, "MA length should be at least 5");
        assert!(DEFAULT_MA_LENGTH <= 500, "MA length should be reasonable");
    }

    #[test]
    fn test_scaling_produces_small_values() {
        let lambda_scaled = DEFAULT_LAMBDA_PARAM / LAMBDA_SCALE;
        let decay_scaled = DEFAULT_DECAY / DECAY_SCALE;

        assert!(lambda_scaled < 0.001, "Scaled lambda should be very small");
        assert!(decay_scaled < 0.001, "Scaled decay should be very small");
    }
}
