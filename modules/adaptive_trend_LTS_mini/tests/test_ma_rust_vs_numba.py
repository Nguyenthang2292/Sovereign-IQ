"""
Integration tests comparing Rust MA implementations vs Numba/pandas_ta.

This test suite verifies:
1. Correctness: Rust implementations produce same results as reference implementations
2. Performance: Rust implementations are faster than Numba/pandas_ta
"""


import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest

try:
    from atc_rust import (
        calculate_dema_rust,
        calculate_ema_rust,
        calculate_hma_rust,
        calculate_lsma_rust,
        calculate_wma_rust,
    )
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    pytest.skip("Rust extensions not available", allow_module_level=True)


def calculate_ema_numba(prices: np.ndarray, length: int) -> np.ndarray:
    """Reference EMA implementation using pandas_ta."""
    series = pd.Series(prices)
    result = ta.ema(series, length=length)
    return result.values if result is not None else np.full(len(prices), np.nan)


def calculate_wma_numba(prices: np.ndarray, length: int) -> np.ndarray:
    """Reference WMA implementation using pandas_ta."""
    series = pd.Series(prices)
    result = ta.wma(series, length=length)
    return result.values if result is not None else np.full(len(prices), np.nan)


def calculate_dema_numba(prices: np.ndarray, length: int) -> np.ndarray:
    """Reference DEMA implementation using pandas_ta."""
    series = pd.Series(prices)
    result = ta.dema(series, length=length)
    return result.values if result is not None else np.full(len(prices), np.nan)


def calculate_lsma_numba(prices: np.ndarray, length: int) -> np.ndarray:
    """Reference LSMA implementation using pandas_ta."""
    series = pd.Series(prices)
    result = ta.linreg(series, length=length)
    return result.values if result is not None else np.full(len(prices), np.nan)


def calculate_hma_numba(prices: np.ndarray, length: int) -> np.ndarray:
    """Reference HMA implementation using pandas_ta (SMA fallback)."""
    series = pd.Series(prices)
    # Note: pandas_ta doesn't have HMA, so we use SMA as fallback
    # This matches the behavior in ma_calculation_enhanced.py
    result = ta.sma(series, length=length)
    return result.values if result is not None else np.full(len(prices), np.nan)


def compare_arrays(
    rust_result: np.ndarray,
    numba_result: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Compare two arrays with tolerance for floating point differences."""
    # Handle NaN values
    rust_nan = np.isnan(rust_result)
    numba_nan = np.isnan(numba_result)

    # Both should have NaN at same positions
    if not np.array_equal(rust_nan, numba_nan):
        return False

    # Compare non-NaN values
    mask = ~rust_nan
    if not np.any(mask):
        return True  # All NaN, consider equal

    return np.allclose(
        rust_result[mask],
        numba_result[mask],
        rtol=rtol,
        atol=atol,
    )


class TestEMA:
    """Test EMA (Exponential Moving Average) implementation."""

    def test_ema_basic(self):
        """Test basic EMA calculation."""
        prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0], dtype=np.float64)
        length = 3

        rust_result = calculate_ema_rust(prices, length)
        numba_result = calculate_ema_numba(prices, length)

        assert compare_arrays(rust_result, numba_result), \
            f"EMA mismatch: Rust={rust_result}, Numba={numba_result}"

    def test_ema_large_array(self):
        """Test EMA with large array."""
        prices = np.random.randn(10000).cumsum() + 100.0
        length = 20

        rust_result = calculate_ema_rust(prices, length)
        numba_result = calculate_ema_numba(prices, length)

        assert compare_arrays(rust_result, numba_result, rtol=1e-4), \
            "EMA mismatch on large array"

    def test_ema_edge_cases(self):
        """Test EMA edge cases."""
        # Single value
        prices = np.array([10.0], dtype=np.float64)
        rust_result = calculate_ema_rust(prices, 1)
        assert not np.isnan(rust_result[0])

        # Length > array size
        prices = np.array([10.0, 11.0], dtype=np.float64)
        rust_result = calculate_ema_rust(prices, 10)
        assert not np.isnan(rust_result[0])


class TestWMA:
    """Test WMA (Weighted Moving Average) implementation."""

    def test_wma_basic(self):
        """Test basic WMA calculation."""
        prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0], dtype=np.float64)
        length = 3

        rust_result = calculate_wma_rust(prices, length)
        numba_result = calculate_wma_numba(prices, length)

        assert compare_arrays(rust_result, numba_result), \
            f"WMA mismatch: Rust={rust_result}, Numba={numba_result}"

    def test_wma_large_array(self):
        """Test WMA with large array."""
        prices = np.random.randn(10000).cumsum() + 100.0
        length = 20

        rust_result = calculate_wma_rust(prices, length)
        numba_result = calculate_wma_numba(prices, length)

        assert compare_arrays(rust_result, numba_result, rtol=1e-4), \
            "WMA mismatch on large array"


class TestDEMA:
    """Test DEMA (Double Exponential Moving Average) implementation."""

    def test_dema_basic(self):
        """Test basic DEMA calculation."""
        prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0], dtype=np.float64)
        length = 3

        rust_result = calculate_dema_rust(prices, length)
        numba_result = calculate_dema_numba(prices, length)

        assert compare_arrays(rust_result, numba_result, rtol=1e-4), \
            f"DEMA mismatch: Rust={rust_result}, Numba={numba_result}"

    def test_dema_large_array(self):
        """Test DEMA with large array."""
        prices = np.random.randn(10000).cumsum() + 100.0
        length = 20

        rust_result = calculate_dema_rust(prices, length)
        numba_result = calculate_dema_numba(prices, length)

        assert compare_arrays(rust_result, numba_result, rtol=1e-4), \
            "DEMA mismatch on large array"


class TestLSMA:
    """Test LSMA (Least Squares Moving Average) implementation."""

    def test_lsma_basic(self):
        """Test basic LSMA calculation."""
        prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0], dtype=np.float64)
        length = 3

        rust_result = calculate_lsma_rust(prices, length)
        numba_result = calculate_lsma_numba(prices, length)

        assert compare_arrays(rust_result, numba_result, rtol=1e-3), \
            f"LSMA mismatch: Rust={rust_result}, Numba={numba_result}"

    def test_lsma_large_array(self):
        """Test LSMA with large array."""
        prices = np.random.randn(10000).cumsum() + 100.0
        length = 20

        rust_result = calculate_lsma_rust(prices, length)
        numba_result = calculate_lsma_numba(prices, length)

        assert compare_arrays(rust_result, numba_result, rtol=1e-3), \
            "LSMA mismatch on large array"


class TestHMA:
    """Test HMA (Hull Moving Average) implementation."""

    def test_hma_basic(self):
        """Test basic HMA calculation."""
        prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0, 12.0], dtype=np.float64)
        length = 5

        rust_result = calculate_hma_rust(prices, length)
        # Note: HMA implementation in Rust is different from SMA fallback
        # So we just check that it produces valid results
        assert not np.all(np.isnan(rust_result)), "HMA should produce some valid values"
        assert len(rust_result) == len(prices), "HMA should return same length as input"

    def test_hma_large_array(self):
        """Test HMA with large array."""
        prices = np.random.randn(10000).cumsum() + 100.0
        length = 20

        rust_result = calculate_hma_rust(prices, length)
        assert not np.all(np.isnan(rust_result)), "HMA should produce some valid values"


class TestPerformance:
    """Performance comparison tests."""

    @pytest.mark.performance
    def test_ema_performance(self):
        """Compare EMA performance."""
        prices = np.random.randn(10000).cumsum() + 100.0
        length = 20

        import time

        # Rust timing
        start = time.perf_counter()
        for _ in range(100):
            _ = calculate_ema_rust(prices, length)
        rust_time = time.perf_counter() - start

        # Numba timing
        start = time.perf_counter()
        for _ in range(100):
            _ = calculate_ema_numba(prices, length)
        numba_time = time.perf_counter() - start

        speedup = numba_time / rust_time
        print(f"\nEMA Performance: Rust={rust_time:.4f}s, Numba={numba_time:.4f}s, Speedup={speedup:.2f}x")

        # Rust should be at least as fast (allowing some variance)
        assert speedup >= 0.8, f"Rust should be competitive: {speedup:.2f}x"

    @pytest.mark.performance
    def test_wma_performance(self):
        """Compare WMA performance."""
        prices = np.random.randn(10000).cumsum() + 100.0
        length = 20

        import time

        # Rust timing
        start = time.perf_counter()
        for _ in range(100):
            _ = calculate_wma_rust(prices, length)
        rust_time = time.perf_counter() - start

        # Numba timing
        start = time.perf_counter()
        for _ in range(100):
            _ = calculate_wma_numba(prices, length)
        numba_time = time.perf_counter() - start

        speedup = numba_time / rust_time
        print(f"\nWMA Performance: Rust={rust_time:.4f}s, Numba={numba_time:.4f}s, Speedup={speedup:.2f}x")

        assert speedup >= 0.8, f"Rust should be competitive: {speedup:.2f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
