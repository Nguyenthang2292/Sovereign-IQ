use ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Signal state representation for type safety and clarity.
///
/// This enum provides a type-safe way to represent the three possible signal states:
/// - Neutral (0): No active signal
/// - Bullish (1): Long/buy signal
/// - Bearish (-1): Short/sell signal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
enum SignalState {
    Neutral = 0,
    Bullish = 1,
    Bearish = -1,
}

impl SignalState {
    /// Convert i8 value to SignalState.
    #[allow(dead_code)]
    #[inline(always)]
    fn from_i8(value: i8) -> Self {
        match value {
            1 => SignalState::Bullish,
            -1 => SignalState::Bearish,
            _ => SignalState::Neutral,
        }
    }

    /// Convert SignalState to i8 value.
    #[inline(always)]
    fn as_i8(self) -> i8 {
        self as i8
    }
}

/// Update signal state based on crossover/crossunder events.
///
/// This function implements the state transition logic:
/// - If up (bullish crossover): transition to Bullish
/// - Else if down (bearish crossunder): transition to Bearish
/// - Else: maintain current state (persistence)
///
/// The inline hint encourages the compiler to optimize this hot path.
#[inline(always)]
fn update_signal_state(current: SignalState, up: bool, down: bool) -> SignalState {
    if up {
        SignalState::Bullish
    } else if down {
        SignalState::Bearish
    } else {
        current
    }
}

/// Core signal persistence processing using enum-based state machine.
///
/// This implementation uses an enum-based state machine for better type safety
/// and code clarity while maintaining high performance through:
/// - Iterator-based approach for potential SIMD optimization
/// - Inline hints for hot path optimization
/// - Minimal allocations (single output array)
///
/// # Arguments
///
/// * `up_arr` - Boolean array view of bullish crossover events
/// * `down_arr` - Boolean array view of bearish crossunder events
///
/// # Returns
///
/// Array1<i8> containing persistent signals (1, -1, or 0)
pub fn process_signal_persistence_internal(
    up_arr: ArrayView1<bool>,
    down_arr: ArrayView1<bool>,
) -> Array1<i8> {
    let n = up_arr.len();
    let mut out = Array1::<i8>::zeros(n);
    let mut current_state = SignalState::Neutral;

    // Use iterator-based approach for better LLVM optimization
    for (i, (&up, &down)) in up_arr.iter().zip(down_arr.iter()).enumerate() {
        current_state = update_signal_state(current_state, up, down);
        out[i] = current_state.as_i8();
    }
    out
}

/// Apply signal persistence logic (state tracking).
///
/// sig = 0
/// if up: sig = 1
/// elif down: sig = -1
/// else: sig = sig[prev]
///
/// # Arguments
///
/// * `up` - Boolean array of bullish crossover events
/// * `down` - Boolean array of bearish crossunder events
///
/// # Returns
///
/// PyArray1<i8> containing persistent signals (1, -1, or last state)
///
/// # Example
///
/// ```python
/// import numpy as np
/// from atc_rust import process_signal_persistence_rust
///
/// up = np.array([True, False, False, False], dtype=bool)
/// down = np.array([False, False, True, False], dtype=bool)
/// signals = process_signal_persistence_rust(up, down)
/// # Result: [1, 1, -1, -1]
/// ```
#[pyfunction]
pub fn process_signal_persistence_rust<'py>(
    _py: Python<'py>,
    up: PyReadonlyArray1<'py, bool>,
    down: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i8>> {
    let up_arr = up.as_array();
    let down_arr = down.as_array();
    let out = process_signal_persistence_internal(up_arr, down_arr);
    PyArray1::from_array(_py, &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_signal_state_enum() {
        // Test enum to i8 conversion
        assert_eq!(SignalState::Neutral.as_i8(), 0);
        assert_eq!(SignalState::Bullish.as_i8(), 1);
        assert_eq!(SignalState::Bearish.as_i8(), -1);

        // Test i8 to enum conversion
        assert_eq!(SignalState::from_i8(0), SignalState::Neutral);
        assert_eq!(SignalState::from_i8(1), SignalState::Bullish);
        assert_eq!(SignalState::from_i8(-1), SignalState::Bearish);

        // Test invalid values default to Neutral
        assert_eq!(SignalState::from_i8(99), SignalState::Neutral);
        assert_eq!(SignalState::from_i8(-99), SignalState::Neutral);
    }

    #[test]
    fn test_state_transitions() {
        let neutral = SignalState::Neutral;

        // Bullish transition
        assert_eq!(
            update_signal_state(neutral, true, false),
            SignalState::Bullish
        );

        // Bearish transition
        assert_eq!(
            update_signal_state(neutral, false, true),
            SignalState::Bearish
        );

        // Persistence (no change)
        assert_eq!(
            update_signal_state(SignalState::Bullish, false, false),
            SignalState::Bullish
        );
        assert_eq!(
            update_signal_state(SignalState::Bearish, false, false),
            SignalState::Bearish
        );

        // Override: bullish overrides bearish
        assert_eq!(
            update_signal_state(SignalState::Bearish, true, false),
            SignalState::Bullish
        );
        assert_eq!(
            update_signal_state(SignalState::Bullish, false, true),
            SignalState::Bearish
        );

        // Simultaneous up and down (up takes precedence)
        assert_eq!(
            update_signal_state(neutral, true, true),
            SignalState::Bullish
        );
    }

    #[test]
    fn test_persistence_logic() {
        let up = array![true, false, false, false];
        let down = array![false, false, true, false];
        let out = process_signal_persistence_internal(up.view(), down.view());

        assert_eq!(out[0], 1); // Bullish signal
        assert_eq!(out[1], 1); // Persists
        assert_eq!(out[2], -1); // Bearish signal
        assert_eq!(out[3], -1); // Persists
    }

    #[test]
    fn test_persistence_with_enum() {
        // Same test as above but verifying enum-based implementation
        let up = array![true, false, false, false];
        let down = array![false, false, true, false];
        let out = process_signal_persistence_internal(up.view(), down.view());

        assert_eq!(out[0], SignalState::Bullish.as_i8());
        assert_eq!(out[1], SignalState::Bullish.as_i8());
        assert_eq!(out[2], SignalState::Bearish.as_i8());
        assert_eq!(out[3], SignalState::Bearish.as_i8());
    }

    #[test]
    fn test_edge_cases() {
        // Empty arrays
        let up = array![];
        let down = array![];
        let out = process_signal_persistence_internal(up.view(), down.view());
        assert_eq!(out.len(), 0);

        // Single element - bullish
        let up = array![true];
        let down = array![false];
        let out = process_signal_persistence_internal(up.view(), down.view());
        assert_eq!(out[0], 1);

        // Single element - bearish
        let up = array![false];
        let down = array![true];
        let out = process_signal_persistence_internal(up.view(), down.view());
        assert_eq!(out[0], -1);

        // Single element - neutral
        let up = array![false];
        let down = array![false];
        let out = process_signal_persistence_internal(up.view(), down.view());
        assert_eq!(out[0], 0);

        // Simultaneous up and down (up takes precedence)
        let up = array![true];
        let down = array![true];
        let out = process_signal_persistence_internal(up.view(), down.view());
        assert_eq!(out[0], 1);
    }

    #[test]
    fn test_alternating_signals() {
        // Test rapid signal changes
        let up = array![true, false, false, true, false];
        let down = array![false, true, false, false, true];
        let out = process_signal_persistence_internal(up.view(), down.view());

        assert_eq!(out[0], 1); // Bullish
        assert_eq!(out[1], -1); // Bearish
        assert_eq!(out[2], -1); // Persists
        assert_eq!(out[3], 1); // Bullish
        assert_eq!(out[4], -1); // Bearish
    }

    #[test]
    fn test_large_array_performance() {
        // Test with large array to verify no performance regression
        let n = 10000;
        let up = Array1::<bool>::from_elem(n, false);
        let down = Array1::<bool>::from_elem(n, false);

        let out = process_signal_persistence_internal(up.view(), down.view());
        assert_eq!(out.len(), n);
        assert!(out.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_long_persistence() {
        // Test long periods of signal persistence
        let n = 1000;
        let mut up = Array1::<bool>::from_elem(n, false);
        let mut down = Array1::<bool>::from_elem(n, false);

        // Single bullish signal at start
        up[0] = true;

        let out = process_signal_persistence_internal(up.view(), down.view());

        // All values should be 1 (bullish persists)
        assert!(out.iter().all(|&x| x == 1));

        // Now add bearish signal in middle
        down[500] = true;
        let out = process_signal_persistence_internal(up.view(), down.view());

        // First half bullish, second half bearish
        assert!(out.slice(ndarray::s![0..500]).iter().all(|&x| x == 1));
        assert!(out.slice(ndarray::s![500..]).iter().all(|&x| x == -1));
    }
}
