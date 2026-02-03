"""
Tests for GPU Kernels and Signal Processing (Phase 2, Section 7.1).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import GPU modules (safe import)
try:
    import cupy as cp

    from modules.adaptive_trend_enhance.core.compute_moving_averages._gpu import (
        _HAS_CUPY,
        _calculate_lsma_gpu_optimized,
        _calculate_wma_gpu_optimized,
        calculate_batch_ema_gpu,
    )
    from modules.adaptive_trend_enhance.core.process_layer1._gpu_signals import cut_signal_gpu, trend_sign_gpu
except ImportError:
    _HAS_CUPY = False
    cp = None


# CPU Reference implementations for comparison
def calculate_wma_cpu(prices: np.ndarray, length: int) -> np.ndarray:
    """CPU Reference WMA."""
    res = np.full_like(prices, np.nan, dtype=np.float64)
    weights = np.arange(1, length + 1)
    w_sum = weights.sum()

    for i in range(length - 1, len(prices)):
        window = prices[i - length + 1 : i + 1]
        res[i] = np.sum(window * weights) / w_sum
    return res


def calculate_lsma_cpu(prices: np.ndarray, length: int) -> np.ndarray:
    """CPU Reference LSMA (Linear Regression)."""
    res = np.full_like(prices, np.nan, dtype=np.float64)
    x = np.arange(1, length + 1)
    sum_x = x.sum()
    sum_xx = (x**2).sum()

    for i in range(length - 1, len(prices)):
        y = prices[i - length + 1 : i + 1]
        sum_y = y.sum()
        sum_xy = (x * y).sum()

        n = length
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2)
        intercept = (sum_y - slope * sum_x) / n

        # LSMA is value at the END of window (x = length)
        res[i] = slope * length + intercept
    return res


@pytest.fixture
def gpu_test_data():
    """Create synthetic data for GPU tests."""
    size = 1000
    np.random.seed(42)  # Deterministic
    prices = np.random.uniform(100, 200, size).astype(np.float64)

    # Warmup GPU if available
    if _HAS_CUPY:
        cp.array([1.0])

    return {"size": size, "prices": prices}


@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not available or GPU not detected")
def test_wma_correctness(gpu_test_data):
    """Verify WMA GPU kernel against CPU implementation."""
    length = 20
    prices = gpu_test_data["prices"]
    prices_gpu = cp.asarray(prices)

    # Run GPU Kernel
    gpu_out = _calculate_wma_gpu_optimized(prices_gpu, length)
    gpu_res = cp.asnumpy(gpu_out)

    # Run CPU Reference
    cpu_res = calculate_wma_cpu(prices, length)

    # Compare (ignoring NaNs at start)
    # Use close tolerance due to float differences (GPU uses double, CPU double)
    np.testing.assert_allclose(gpu_res[length:], cpu_res[length:], rtol=1e-10, atol=1e-10)

    print("\nWMA Correctness: PASSED")


@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not available or GPU not detected")
def test_lsma_correctness(gpu_test_data):
    """Verify LSMA GPU kernel against CPU implementation."""
    length = 20
    prices = gpu_test_data["prices"]
    prices_gpu = cp.asarray(prices)

    gpu_out = _calculate_lsma_gpu_optimized(prices_gpu, length)
    gpu_res = cp.asnumpy(gpu_out)

    cpu_res = calculate_lsma_cpu(prices, length)

    # LSMA involves more ops, maybe slightly higher error
    np.testing.assert_allclose(gpu_res[length:], cpu_res[length:], rtol=1e-8, atol=1e-8)

    print("LSMA Correctness: PASSED")


@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not available or GPU not detected")
def test_batch_ema_correctness(gpu_test_data):
    """Verify Batch EMA GPU kernel."""
    # Create a batch of symbols
    num_symbols = 5
    length = 10
    size = gpu_test_data["size"]
    prices_batch = np.random.uniform(100, 200, (num_symbols, size)).astype(np.float64)

    prices_gpu = cp.asarray(prices_batch)

    # Run Batch GPU
    gpu_out = calculate_batch_ema_gpu(prices_gpu, length)
    gpu_res = cp.asnumpy(gpu_out)

    # Run Sequential CPU loop for verification
    alpha = 2.0 / (length + 1.0)
    cpu_res = np.zeros_like(prices_batch)

    for s in range(num_symbols):
        # Simple EMA loop
        ema = np.zeros(size)
        ema[0] = prices_batch[s, 0]
        for i in range(1, size):
            ema[i] = alpha * prices_batch[s, i] + (1 - alpha) * ema[i - 1]
        cpu_res[s] = ema

    np.testing.assert_allclose(gpu_res, cpu_res, rtol=1e-8, atol=1e-8)

    print("Batch EMA Correctness: PASSED")


@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not available or GPU not detected")
def test_gpu_signals():
    """Verify cut_signal and trend_sign on GPU."""
    # Data: -1.0 to 1.0
    data = np.linspace(-1.0, 1.0, 100)
    data_gpu = cp.asarray(data)

    # Test cut_signal
    # L=0.5, S=-0.5
    cut_gpu = cut_signal_gpu(data_gpu, long_threshold=0.5, short_threshold=-0.5)
    cut_res = cp.asnumpy(cut_gpu)

    expected = np.zeros_like(data)
    expected[data > 0.5] = 1
    expected[data < -0.5] = -1

    np.testing.assert_equal(cut_res, expected)

    # Test trend_sign
    trend_gpu = trend_sign_gpu(data_gpu)
    trend_res = cp.asnumpy(trend_gpu)

    expected_trend = np.sign(data)
    # Fix 0 sign behavior if needed (numpy sign(0) is 0)

    np.testing.assert_equal(trend_res, expected_trend)

    print("GPU Signals: PASSED")
