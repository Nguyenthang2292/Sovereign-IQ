"""
Benchmark for SIMD/Numba optimizations.
"""

import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from modules.adaptive_trend_enhance.core.compute_moving_averages._numba_cores import (
    _calculate_lsma_core,
    _calculate_wma_core,
)


def benchmark_wma(size=10000, length=20, runs=100):
    prices = np.random.randn(size).astype(np.float64)
    # Ensure compile
    _calculate_wma_core(prices, length)

    start = time.time()
    for _ in range(runs):
        _calculate_wma_core(prices, length)
    end = time.time()

    avg = (end - start) / runs
    print(f"WMA (size={size}, len={length}): {avg * 1000:.4f} ms per run")
    return avg


def benchmark_lsma(size=10000, length=20, runs=100):
    prices = np.random.randn(size).astype(np.float64)
    _calculate_lsma_core(prices, length)

    start = time.time()
    for _ in range(runs):
        _calculate_lsma_core(prices, length)
    end = time.time()

    avg = (end - start) / runs
    print(f"LSMA (size={size}, len={length}): {avg * 1000:.4f} ms per run")
    return avg


if __name__ == "__main__":
    print("Benchmarking Numba Cores (Parallel Enabled)...")

    # Small size
    benchmark_wma(size=1000, length=20, runs=1000)
    benchmark_lsma(size=1000, length=20, runs=1000)

    # Medium size
    benchmark_wma(size=10000, length=50, runs=100)
    benchmark_lsma(size=10000, length=50, runs=100)

    # Large size (Parallel shine here)
    benchmark_wma(size=100000, length=100, runs=10)
    benchmark_lsma(size=100000, length=100, runs=10)

    print("Benchmark Complete")
