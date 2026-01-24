"""
Integration tests for signal persistence Rust implementation.

This test suite verifies:
1. Correctness: Rust implementation produces same results as Numba reference
2. Edge cases: Empty arrays, single elements, simultaneous signals
3. Performance: Large arrays process correctly
"""

import numpy as np
import pytest

try:
    from atc_rust import process_signal_persistence_rust

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    pytest.skip("Rust extensions not available", allow_module_level=True)

from modules.adaptive_trend_LTS.core.signal_detection.generate_signal import _apply_signal_persistence


class TestSignalPersistenceCorrectness:
    """Test correctness against Numba reference implementation."""

    def test_basic_persistence(self):
        """Test basic signal persistence logic."""
        up = np.array([True, False, False, False], dtype=bool)
        down = np.array([False, False, True, False], dtype=bool)

        rust_result = process_signal_persistence_rust(up, down)

        numba_result = np.zeros(len(up), dtype=np.int8)
        _apply_signal_persistence(up, down, numba_result)

        np.testing.assert_array_equal(rust_result, numba_result)
        assert rust_result[0] == 1  # Bullish
        assert rust_result[1] == 1  # Persists
        assert rust_result[2] == -1  # Bearish
        assert rust_result[3] == -1  # Persists

    def test_alternating_signals(self):
        """Test rapid signal changes."""
        up = np.array([True, False, False, True, False], dtype=bool)
        down = np.array([False, True, False, False, True], dtype=bool)

        rust_result = process_signal_persistence_rust(up, down)

        numba_result = np.zeros(len(up), dtype=np.int8)
        _apply_signal_persistence(up, down, numba_result)

        np.testing.assert_array_equal(rust_result, numba_result)

    def test_large_array(self):
        """Test with large array to verify performance and correctness."""
        n = 10000
        np.random.seed(42)
        up = np.random.choice([True, False], size=n, p=[0.05, 0.95])
        down = np.random.choice([True, False], size=n, p=[0.05, 0.95])

        rust_result = process_signal_persistence_rust(up, down)

        numba_result = np.zeros(n, dtype=np.int8)
        _apply_signal_persistence(up, down, numba_result)

        np.testing.assert_array_equal(rust_result, numba_result)

    def test_long_persistence(self):
        """Test long periods of signal persistence."""
        n = 1000
        up = np.zeros(n, dtype=bool)
        down = np.zeros(n, dtype=bool)

        # Single bullish signal at start
        up[0] = True

        rust_result = process_signal_persistence_rust(up, down)

        # All values should be 1 (bullish persists)
        assert np.all(rust_result == 1)

        # Add bearish signal in middle
        down[500] = True
        rust_result = process_signal_persistence_rust(up, down)

        # First half bullish, second half bearish
        assert np.all(rust_result[:500] == 1)
        assert np.all(rust_result[500:] == -1)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        up = np.array([], dtype=bool)
        down = np.array([], dtype=bool)

        rust_result = process_signal_persistence_rust(up, down)

        assert len(rust_result) == 0
        assert rust_result.dtype == np.int8

    def test_single_element_bullish(self):
        """Test single element - bullish signal."""
        up = np.array([True], dtype=bool)
        down = np.array([False], dtype=bool)

        rust_result = process_signal_persistence_rust(up, down)

        assert len(rust_result) == 1
        assert rust_result[0] == 1

    def test_single_element_bearish(self):
        """Test single element - bearish signal."""
        up = np.array([False], dtype=bool)
        down = np.array([True], dtype=bool)

        rust_result = process_signal_persistence_rust(up, down)

        assert len(rust_result) == 1
        assert rust_result[0] == -1

    def test_single_element_neutral(self):
        """Test single element - neutral (no signal)."""
        up = np.array([False], dtype=bool)
        down = np.array([False], dtype=bool)

        rust_result = process_signal_persistence_rust(up, down)

        assert len(rust_result) == 1
        assert rust_result[0] == 0

    def test_simultaneous_up_down(self):
        """Test simultaneous up and down (up takes precedence)."""
        up = np.array([True, True, False], dtype=bool)
        down = np.array([True, False, True], dtype=bool)

        rust_result = process_signal_persistence_rust(up, down)

        assert rust_result[0] == 1  # Up takes precedence
        assert rust_result[1] == 1  # Bullish
        assert rust_result[2] == -1  # Bearish

    def test_all_neutral(self):
        """Test array with no signals."""
        n = 100
        up = np.zeros(n, dtype=bool)
        down = np.zeros(n, dtype=bool)

        rust_result = process_signal_persistence_rust(up, down)

        assert np.all(rust_result == 0)

    def test_all_bullish(self):
        """Test array with all bullish signals."""
        n = 100
        up = np.ones(n, dtype=bool)
        down = np.zeros(n, dtype=bool)

        rust_result = process_signal_persistence_rust(up, down)

        assert np.all(rust_result == 1)

    def test_all_bearish(self):
        """Test array with all bearish signals."""
        n = 100
        up = np.zeros(n, dtype=bool)
        down = np.ones(n, dtype=bool)

        rust_result = process_signal_persistence_rust(up, down)

        assert np.all(rust_result == -1)


class TestPerformance:
    """Performance comparison tests."""

    @pytest.mark.performance
    def test_performance_comparison(self):
        """Compare Rust vs Numba performance."""
        import time

        n = 10000
        iterations = 1000

        # Generate test data
        np.random.seed(42)
        up = np.random.choice([True, False], size=n, p=[0.05, 0.95])
        down = np.random.choice([True, False], size=n, p=[0.05, 0.95])

        # Benchmark Rust
        start = time.perf_counter()
        for _ in range(iterations):
            rust_result = process_signal_persistence_rust(up, down)
        rust_time = time.perf_counter() - start

        # Benchmark Numba
        numba_result = np.zeros(n, dtype=np.int8)
        start = time.perf_counter()
        for _ in range(iterations):
            _apply_signal_persistence(up, down, numba_result)
        numba_time = time.perf_counter() - start

        speedup = numba_time / rust_time
        print(f"\nRust: {rust_time:.4f}s")
        print(f"Numba: {numba_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Verify results match
        np.testing.assert_array_equal(rust_result, numba_result)

        # Expect at least 2x speedup (current target from phase3_task.md)
        assert speedup >= 2.0, f"Expected 2x+ speedup, got {speedup:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
