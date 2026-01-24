from __future__ import annotations

import math
from typing import Optional

import numpy as np

from modules.common.ui.logging import log_debug, log_error

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:  # pragma: no cover
    _HAS_CUPY = False


# =============================================================================
# OPTIMIZED CUDA KERNELS
# =============================================================================

# WMA Kernel: Weighted Moving Average
# Calculates: sum(price[i] * weight[k]) / sum(weights)
# where weight[k] = k + 1
_WMA_KERNEL_CODE = r"""
extern "C" __global__
void wma_kernel(const double* prices, double* output, int n_bars, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n_bars) return;

    // For WMA, we need 'length' previous bars.
    // If not enough history, output NaN.
    if (i < length - 1) {
        output[i] = nan("");
        return;
    }

    double sum = 0.0;
    double weight_sum = (double)length * (length + 1) / 2.0;

    // Weights: 1, 2, ..., length
    // Applied to window: price[i-length+1] ... price[i]
    // price[i] gets weight 'length'
    // price[i-length+1] gets weight 1

    for (int k = 0; k < length; k++) {
        // Index in prices array: i - k
        // Weight: length - k
        int idx = i - k;
        double w = (double)(length - k);
        sum += prices[idx] * w;
    }

    output[i] = sum / weight_sum;
}
"""

# LSMA Kernel: Least Squares Moving Average
# Calculates linear regression y = mx + b for the window, then projects to current bar.
# Optimized formulation for fixed length window.
_LSMA_KERNEL_CODE = r"""
extern "C" __global__
void lsma_kernel(const double* prices, double* output, int n_bars, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n_bars) return;

    if (i < length - 1) {
        output[i] = nan("");
        return;
    }

    // Linear Regression on window of size 'length'
    // We want the value at the END of the regression line (at current bar 'i')

    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xy = 0.0;
    double sum_xx = 0.0;

    // X coordinates: 1 to length
    // Y coordinates: prices in window

    // Pre-calculate fixed sums for x and x^2 if possible, but here we do it inline for simplicity
    // sum_x = length * (length + 1) / 2
    // sum_xx = length * (length + 1) * (2*length + 1) / 6

    // Optimization: x is just 1, 2, ..., length relative to the window start

    for (int k = 0; k < length; k++) {
        // Current window index: i - length + 1 + k
        // But for regression X, let's use 1..length
        double x = (double)(k + 1);
        double y = prices[i - length + 1 + k];

        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    double n = (double)length;
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    double intercept = (sum_y - slope * sum_x) / n;

    // Value at current bar (x = length)
    output[i] = slope * n + intercept;
}
"""


def _get_kernel(name: str, code: str):
    """Compile or retrieve cached RawKernel."""
    return cp.RawKernel(code, name)


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================


def _calculate_ma_gpu(prices: np.ndarray, length: int, ma_type: str) -> Optional[np.ndarray]:
    """Calculate Moving Average on GPU."""
    if not _HAS_CUPY:
        return None

    try:
        # Move inputs to GPU
        # Check if already on GPU
        if isinstance(prices, cp.ndarray):  # Is CuPy array
            prices_gpu = prices
        else:
            prices_gpu = cp.asarray(prices, dtype=cp.float64)

        if ma_type == "EMA":
            result_gpu = _calculate_ema_gpu_loop(prices_gpu, length)
        elif ma_type == "HMA":
            result_gpu = _calculate_hma_gpu(prices_gpu, length)
        elif ma_type == "WMA":
            result_gpu = _calculate_wma_gpu_optimized(prices_gpu, length)
        elif ma_type == "DEMA":
            result_gpu = _calculate_dema_gpu(prices_gpu, length)
        elif ma_type == "LSMA":
            result_gpu = _calculate_lsma_gpu_optimized(prices_gpu, length)
        else:
            return None

        if result_gpu is None:
            return None

        # Return as numpy array if input was numpy
        if isinstance(prices, cp.ndarray):
            return result_gpu
        else:
            return cp.asnumpy(result_gpu)

    except Exception as e:
        log_debug(f"GPU calculation failed for {ma_type}: {e}")
        return None


def _calculate_ema_gpu_loop(prices_gpu: cp.ndarray, length: int) -> cp.ndarray:
    """Calculate EMA using simple loop (recursive dependency prevents easy parallelization)."""
    # NOTE: For recursive EMAs, a sequential scan on GPU is often SLOWER than CPU due to global memory latency.
    # However, for consistency when data is already on GPU, we provide it.
    # Best optimization for EMA is BATCH processing (many symbols in parallel).

    n = len(prices_gpu)
    alpha = 2.0 / (length + 1.0)

    # We can use CuPy ElementwiseKernel for non-recursive parts, but EMA is inherently recursive.
    # Fallback to single-thread GPU kernel.
    # Using a Python loop on CuPy array indexing is VERY SLOW due to kernel launch overhead per element.

    code = r"""
    extern "C" __global__
    void ema_recursive_kernel(const double* prices, double* output, int n, double alpha) {
        // Single thread execution for recursive dependency
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            output[0] = prices[0];
            for (int i = 1; i < n; i++) {
                // Simple EMA:
                output[i] = alpha * prices[i] + (1.0 - alpha) * output[i-1];
            }
        }
    }
    """
    kernel = cp.RawKernel(code, "ema_recursive_kernel")
    output = cp.empty_like(prices_gpu)

    # Launch with 1 block, 1 thread (Sequential GPU)
    # This is SLOW (approx same as CPU) but keeps data on GPU.
    # Use numpy types for scalars to ensure correct kernel argument mapping
    kernel((1,), (1,), (prices_gpu, output, np.int32(n), np.float64(alpha)))
    return output


def _calculate_wma_gpu_optimized(prices_gpu: cp.ndarray, length: int) -> cp.ndarray:
    """Calculate WMA using custom optimized CUDA kernel."""
    n_bars = len(prices_gpu)
    output = cp.full(n_bars, cp.nan, dtype=cp.float64)

    kernel = _get_kernel("wma_kernel", _WMA_KERNEL_CODE)

    # Grid calculation
    threads_per_block = 256
    blocks_per_grid = (n_bars + threads_per_block - 1) // threads_per_block

    kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (prices_gpu, output, np.int32(n_bars), np.int32(length)),
    )

    return output


def _calculate_lsma_gpu_optimized(prices_gpu: cp.ndarray, length: int) -> cp.ndarray:
    """Calculate LSMA using custom optimized CUDA kernel."""
    n_bars = len(prices_gpu)
    output = cp.full(n_bars, cp.nan, dtype=cp.float64)

    kernel = _get_kernel("lsma_kernel", _LSMA_KERNEL_CODE)

    threads_per_block = 256
    blocks_per_grid = (n_bars + threads_per_block - 1) // threads_per_block

    kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (prices_gpu, output, np.int32(n_bars), np.int32(length)),
    )

    return output


def _calculate_hma_gpu(prices_gpu: cp.ndarray, length: int) -> cp.ndarray:
    """Calculate HMA on GPU."""
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)

    half_len = int(length / 2)
    sqrt_len = int(math.sqrt(length))

    wma_n = _calculate_wma_gpu_optimized(prices_gpu, length)
    wma_half = _calculate_wma_gpu_optimized(prices_gpu, half_len)

    # 2 * WMA(n/2) - WMA(n)
    # Elementwise operation is fast in CuPy
    raw_hma = 2.0 * wma_half - wma_n

    return _calculate_wma_gpu_optimized(raw_hma, sqrt_len)


def _calculate_dema_gpu(prices_gpu: cp.ndarray, length: int) -> cp.ndarray:
    """Calculate DEMA on GPU."""
    ema1 = _calculate_ema_gpu_loop(prices_gpu, length)
    ema2 = _calculate_ema_gpu_loop(ema1, length)
    return 2.0 * ema1 - ema2


# Batch EMA Kernel: One thread per symbol (or block)
# Input shape: (num_symbols, n_bars) flattened
# Output shape: same
_BATCH_EMA_KERNEL_CODE = r"""
extern "C" __global__
void batch_ema_kernel(const double* prices, double* output, int num_symbols, int n_bars, double alpha) {
    // One thread per symbol
    int symbol_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (symbol_idx >= num_symbols) return;

    // Calculate offset for this symbol
    int offset = symbol_idx * n_bars;

    // Pointer to start of this symbol's data
    const double* symbol_prices = prices + offset;
    double* symbol_output = output + offset;

    // Compute EMA sequentially for this symbol
    // Check for NaN at start if needed, but assuming clean data or handle in loop
    
    // Initialize first value
    symbol_output[0] = symbol_prices[0];
    
    // Iterate through bars
    for (int i = 1; i < n_bars; i++) {
        // Equivalent to: ema = alpha * price + (1-alpha) * prev
        // Using fma (fused multiply-add) for potential precision/speed
        // symbol_output[i] = alpha * symbol_prices[i] + (1.0 - alpha) * symbol_output[i - 1];
        
        double prev = symbol_output[i - 1];
        double curr = symbol_prices[i];
        
        // Handle NaNs: if prev is NaN, restart from current? Or propagate?
        // Standard EMA behavior: if input is NaN, output is usually NaN unless handled.
        // If prev is NaN, we treat current as new seed if current is not NaN.
        
        if (isnan(prev)) {
            symbol_output[i] = curr;
        } else if (isnan(curr)) {
            symbol_output[i] = prev; // Hold optimization? Or NaN? Standard is often hold or NaN. Let's start with NaN to match pandas ewm.
            // Actually pandas ewm ignores NaNs by default. 
            // Simple approach: standard formula.
             symbol_output[i] = alpha * curr + (1.0 - alpha) * prev;
        } else {
             symbol_output[i] = alpha * curr + (1.0 - alpha) * prev;
        }
    }
}
"""


def calculate_batch_ema_gpu(prices_batch: cp.ndarray, length: int) -> cp.ndarray:
    """
    Calculate EMA for a batch of symbols in parallel on GPU.

    Args:
        prices_batch: CuPy array of shape (num_symbols, n_bars).
        length: EMA length.

    Returns:
        CuPy array of same shape with EMA values.
    """
    if not _HAS_CUPY:
        return None

    # Ensure input is 2D
    if prices_batch.ndim != 2:
        log_error(f"calculate_batch_ema_gpu requires 2D array, got {prices_batch.ndim}D")
        return None

    num_symbols, n_bars = prices_batch.shape

    # Allocate output
    output = cp.empty_like(prices_batch, dtype=cp.float64)

    alpha = 2.0 / (length + 1.0)

    kernel = _get_kernel("batch_ema_kernel", _BATCH_EMA_KERNEL_CODE)

    # Grid: One thread per symbol
    threads_per_block = 256
    blocks_per_grid = (num_symbols + threads_per_block - 1) // threads_per_block

    kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (prices_batch, output, np.int32(num_symbols), np.int32(n_bars), np.float64(alpha)),
    )

    return output


__all__ = [
    "_HAS_CUPY",
    "_calculate_ma_gpu",
    "_calculate_ema_gpu_loop",
    "_calculate_wma_gpu_optimized",
    "_calculate_dema_gpu",
    "_calculate_lsma_gpu_optimized",
]
