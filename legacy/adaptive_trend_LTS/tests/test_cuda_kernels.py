"""
Unit tests for CUDA kernel implementations.

Tests numerical accuracy of CUDA kernels against CPU Rust implementations.
"""

import numpy as np
import pytest

try:
    from atc_rust import (
        calculate_and_classify_cuda,
        # Signal
        calculate_average_signal_cuda,
        # MA
        calculate_ema_cuda,
        calculate_ema_rust,
        # Equity
        calculate_equity_cuda,
        calculate_equity_rust,
        calculate_hma_cuda,
        calculate_hma_rust,
        calculate_kama_cuda,
        calculate_kama_rust,
        calculate_wma_cuda,
        calculate_wma_rust,
        classify_trend_cuda,
    )

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extensions not available")
class TestEquityCUDA:
    """Test equity calculation CUDA kernel."""

    def test_equity_cuda_vs_cpu_basic(self):
        """Test basic equity calculation."""
        n = 1000
        r_values = np.random.randn(n) * 0.02  # 2% returns
        sig_prev = np.random.choice([-1.0, 0.0, 1.0], size=n)

        cpu_result = calculate_equity_rust(r_values, sig_prev, starting_equity=100.0, decay_multiplier=0.97, cutout=0)

        cuda_result = calculate_equity_cuda(r_values, sig_prev, starting_equity=100.0, decay_multiplier=0.97, cutout=0)

        np.testing.assert_allclose(cpu_result, cuda_result, rtol=1e-6, atol=1e-6)

    def test_equity_cuda_with_cutout(self):
        """Test equity with cutout period."""
        n = 500
        cutout = 50
        r_values = np.random.randn(n) * 0.01
        sig_prev = np.random.choice([-1.0, 1.0], size=n)

        cpu_result = calculate_equity_rust(r_values, sig_prev, 100.0, 0.97, cutout)
        cuda_result = calculate_equity_cuda(r_values, sig_prev, 100.0, 0.97, cutout)

        # First cutout values should be NaN
        assert np.all(np.isnan(cpu_result[:cutout]))
        assert np.all(np.isnan(cuda_result[:cutout]))

        # Rest should match
        np.testing.assert_allclose(cpu_result[cutout:], cuda_result[cutout:], rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extensions not available")
class TestMACUDA:
    """Test moving average CUDA kernels."""

    def test_ema_cuda_vs_cpu(self):
        """Test EMA calculation."""
        prices = np.random.randn(1000).cumsum() + 100
        length = 20

        cpu_result = calculate_ema_rust(prices, length)
        cuda_result = calculate_ema_cuda(prices, length)

        np.testing.assert_allclose(cpu_result, cuda_result, rtol=1e-6, atol=1e-6)

    def test_kama_cuda_vs_cpu(self):
        """Test KAMA calculation."""
        prices = np.random.randn(1000).cumsum() + 100
        length = 20

        cpu_result = calculate_kama_rust(prices, length)
        cuda_result = calculate_kama_cuda(prices, length)

        np.testing.assert_allclose(cpu_result, cuda_result, rtol=1e-6, atol=1e-6)

    def test_wma_cuda_vs_cpu(self):
        """Test WMA calculation."""
        prices = np.random.randn(1000).cumsum() + 100
        length = 20

        cpu_result = calculate_wma_rust(prices, length)
        cuda_result = calculate_wma_cuda(prices, length)

        np.testing.assert_allclose(cpu_result, cuda_result, rtol=1e-6, atol=1e-6)

    def test_hma_cuda_vs_cpu(self):
        """Test HMA calculation."""
        prices = np.random.randn(1000).cumsum() + 100
        length = 20

        cpu_result = calculate_hma_rust(prices, length)
        cuda_result = calculate_hma_cuda(prices, length)

        np.testing.assert_allclose(cpu_result, cuda_result, rtol=1e-6, atol=1e-6)

    def test_ma_with_different_sizes(self):
        """Test MA calculations with various array sizes."""
        lengths = [10, 50, 100]
        sizes = [100, 500, 2000]

        for size in sizes:
            prices = np.random.randn(size).cumsum() + 100
            for length in lengths:
                if length < size:
                    cpu_ema = calculate_ema_rust(prices, length)
                    cuda_ema = calculate_ema_cuda(prices, length)
                    np.testing.assert_allclose(cpu_ema, cuda_ema, rtol=1e-6)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extensions not available")
class TestSignalCUDA:
    """Test signal classification CUDA kernels."""

    def test_weighted_average_signal(self):
        """Test weighted average signal calculation."""
        n_mas = 5
        n_bars = 1000

        signals = np.random.randn(n_mas, n_bars)
        equities = np.abs(np.random.randn(n_mas, n_bars)) + 1.0

        avg_signal = calculate_average_signal_cuda(signals, equities, long_threshold=0.5, short_threshold=-0.5)

        assert avg_signal.shape == (n_bars,)
        assert np.all(np.isfinite(avg_signal))
        assert np.all(avg_signal >= -1.0) and np.all(avg_signal <= 1.0)

    def test_classify_trend(self):
        """Test trend classification."""
        signals = np.array([0.8, -0.7, 0.2, -0.3, 0.0])

        trends = classify_trend_cuda(signals, 0.5, -0.5)

        expected = np.array([1, -1, 0, 0, 0], dtype=np.int32)
        np.testing.assert_array_equal(trends, expected)

    def test_fused_average_and_classify(self):
        """Test fused weighted average + classification."""
        n_mas = 3
        n_bars = 500

        signals = np.random.randn(n_mas, n_bars)
        equities = np.abs(np.random.randn(n_mas, n_bars)) + 1.0

        avg_signal, trends = calculate_and_classify_cuda(signals, equities, 0.5, -0.5)

        # Verify shapes
        assert avg_signal.shape == (n_bars,)
        assert trends.shape == (n_bars,)

        # Verify trends match signals
        for i in range(n_bars):
            if avg_signal[i] > 0.5:
                assert trends[i] == 1
            elif avg_signal[i] < -0.5:
                assert trends[i] == -1
            else:
                assert trends[i] == 0

    def test_signal_with_nan_handling(self):
        """Test signal calculation with NaN values."""
        n_mas = 3
        n_bars = 100

        signals = np.random.randn(n_mas, n_bars)
        signals[0, :10] = np.nan  # Add some NaNs

        equities = np.ones((n_mas, n_bars))

        avg_signal = calculate_average_signal_cuda(signals, equities, 0.5, -0.5)

        # Should handle NaNs gracefully
        assert np.all(np.isfinite(avg_signal))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA extensions not available")
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_arrays(self):
        """Test with minimal data."""
        prices = np.array([100.0, 101.0, 102.0])

        # Should not crash
        result = calculate_ema_cuda(prices, 2)
        assert len(result) == 3

    def test_all_nan_prices(self):
        """Test with all NaN prices."""
        prices = np.full(100, np.nan)

        result = calculate_ema_cuda(prices, 10)
        assert np.all(np.isnan(result))

    def test_zero_equity(self):
        """Test signal calculation with zero equity."""
        signals = np.random.randn(3, 100)
        equities = np.zeros((3, 100))

        avg_signal = calculate_average_signal_cuda(signals, equities, 0.5, -0.5)

        # Should return zeros when all equities are zero
        assert np.all(avg_signal == 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
