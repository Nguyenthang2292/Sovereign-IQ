"""
GPU-accelerated signal processing logic.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Union

import numpy as np

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

logger = logging.getLogger(__name__)


# CUDA Kernel for Signal Persistence (Forward Fill 0s)
_persistence_kernel_source = r"""
extern "C" __global__
void persistence_kernel(
    const double* raw_sig,  // (n_signals, n_bars)
    double* out_sig,        // (n_signals, n_bars)
    int n_signals,
    int n_bars
) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= n_signals) return;

    int row_offset = id * n_bars;
    double prev = 0.0;

    for (int t = 0; t < n_bars; t++) {
        int idx = row_offset + t;
        double val = raw_sig[idx];

        // If not 0 and not NaN, update prev
        // Using val!=0.0 comparison which is safe for exact 0 int->float
        bool is_nan = (val != val);

        if (!is_nan && val != 0.0) {
            prev = val;
        }

        out_sig[idx] = prev;
    }
}
"""


def _get_persistence_kernel():
    if not _HAS_CUPY:
        return None
    return cp.RawKernel(_persistence_kernel_source, "persistence_kernel")


def rate_of_change_gpu(prices_gpu: cp.ndarray, length: int = 1) -> cp.ndarray:
    """
    Calculate Rate of Change (Percentage Change) on GPU.
    R = (Price - PrevPrice) / PrevPrice

    Args:
        prices_gpu: (num_signals, n_bars) or (n_bars,)
        length: offset

    Returns:
        CuPy array same shape as input
    """
    if not _HAS_CUPY:
        return None

    prices_gpu = cp.asarray(prices_gpu)
    if prices_gpu.ndim == 1:
        prices_gpu = prices_gpu.reshape(1, -1)

    n_rows, n_cols = prices_gpu.shape

    if n_cols <= length:
        return cp.full_like(prices_gpu, cp.nan)

    # Vectorized calculation
    # p[t], p[t-1]
    curr = prices_gpu[:, length:]
    prev = prices_gpu[:, :-length]

    # Avoid div by zero
    with cp.errstate(divide="ignore", invalid="ignore"):
        roc = (curr - prev) / prev

    # Replace Inf with NaN if needed, or handle
    roc = cp.nan_to_num(roc, nan=0.0, posinf=0.0, neginf=0.0)

    # Pad beginning
    pad = cp.zeros((n_rows, length), dtype=prices_gpu.dtype)
    # Usually RoC first value is NaN or 0?
    # Pine Script change() returns NaN at start?
    # Let's fill 0.0 to be safe for subsequent calcs, or NaN if consistent.
    # The CPU version returns NaN at start? Pandas pct_change returns NaN.
    # However, equity calculation handles NaNs by treating a=0. So NaN is fine.
    pad[:] = cp.nan

    return cp.hstack([pad, roc])


def generate_signal_from_ma_gpu(prices_gpu: cp.ndarray, ma_gpu: cp.ndarray) -> cp.ndarray:
    """
    Generate persistent signals on GPU.
    Signal = 1 if Price > MA, -1 if Price < MA.
    Persist previous signal if cross hasn't reversed (implied by Sign logic).
    Actually, standard Sign logic IS persistent if we ignore noise.
    But strictly, we want "Crossover=1, Crossunder=-1, else hold".

    If 'hold' means 'keep previous 1/-1', then:
    If Price > MA -> 1
    If Price < MA -> -1
    If Price == MA -> Hold previous?

    Standard `cp.sign(p - ma)` returns 0 if equal.
    So we just need to ffill the 0s.
    """
    if not _HAS_CUPY:
        return None

    # Diff
    diff = prices_gpu - ma_gpu

    # Sign (1, -1, 0)
    raw_sig = cp.sign(diff)

    # Kernel for persistence (ffill 0s)
    kernel = _get_persistence_kernel()

    n_signals, n_bars = raw_sig.shape
    out_sig = cp.empty_like(raw_sig)

    block_size = 128
    grid_size = (n_signals + block_size - 1) // block_size

    kernel((grid_size,), (block_size,), (raw_sig, out_sig, n_signals, n_bars))

    return out_sig


def cut_signal_gpu(
    signal_gpu: Union[cp.ndarray, np.ndarray],
    threshold: float = 0.49,
    long_threshold: float = None,
    short_threshold: float = None,
    cutout: int = 0,
) -> Optional[cp.ndarray]:
    """
    Discretize continuous signal into {-1, 0, 1} on GPU.

    Args:
        signal_gpu: Signal array (CuPy or NumPy).
        threshold: Default threshold.
        long_threshold: Long threshold.
        short_threshold: Short threshold.
        cutout: Number of bars to zero out at start.

    Returns:
        CuPy array with discrete values.
    """
    if not _HAS_CUPY:
        return None

    if long_threshold is None:
        long_threshold = threshold
    if short_threshold is None:
        short_threshold = -threshold

    # Ensure input is on GPU
    if not hasattr(signal_gpu, "device"):
        signal_gpu = cp.asarray(signal_gpu)

    # Use vectorized CuPy operations
    # c = 1 if x > long, -1 if x < short, else 0

    # We use cp.where(condition, x, y)
    # Nested where: where(> L, 1, where(< S, -1, 0))

    # Check for NaNs: NaNs in comparisons return False, so they become 0 naturally in this logic.
    output = cp.where(signal_gpu > long_threshold, 1, cp.where(signal_gpu < short_threshold, -1, 0)).astype(cp.int8)

    # Apply cutout
    if cutout > 0 and cutout < signal_gpu.shape[-1]:
        # Slice assignment on GPU
        # If signal_gpu is 1D:
        if output.ndim == 1:
            output[:cutout] = 0
        # If signal_gpu is 2D (batch):
        elif output.ndim == 2:
            output[:, :cutout] = 0

    return output


def trend_sign_gpu(signal_gpu: Union[cp.ndarray, np.ndarray]) -> Optional[cp.ndarray]:
    """
    Calculate trend sign on GPU (sign(x)).

    Args:
        signal_gpu: Signal array.

    Returns:
        CuPy array with sign values {-1, 0, 1}.
    """
    if not _HAS_CUPY:
        return None

    if not hasattr(signal_gpu, "device"):
        signal_gpu = cp.asarray(signal_gpu)

    # cp.sign returns -1, 0, 1 for float inputs (and deals with NaNs usually by returning NaN)
    # We want strict {-1, 0, 1}, casting NaNs to 0?

    # cp.sign implementation:
    # x > 0 -> 1
    # x == 0 -> 0
    # x < 0 -> -1
    # NaN -> NaN

    res = cp.sign(signal_gpu)

    # Handle NaNs -> 0
    res = cp.nan_to_num(res, nan=0.0)

    return res.astype(cp.int8)


def weighted_signal_gpu(
    signals: Dict[str, cp.ndarray],
    equities: Dict[str, cp.ndarray],
    weights: Dict[str, float] = None,
) -> Optional[cp.ndarray]:
    """
    Calculate weighted average signal on GPU.

    Formula: Sum(signal * equity * weight) / Sum(equity * weight)

    Args:
        signals: Dict of {ma_type: signal_array_gpu}
        equities: Dict of {ma_type: equity_array_gpu}
        weights: Optional dict of {ma_type: initial_weight} (default 1.0)

    Returns:
        Weighted signal array (CuPy).
    """
    if not _HAS_CUPY:  # or not signals or not equities: # Dictionary checks are separate
        return None

    if not signals or not equities:
        return None

    # Assume all arrays have same shape (1D or 2D batch)
    first_key = next(iter(signals))
    shape = signals[first_key].shape

    numerator = cp.zeros(shape, dtype=cp.float64)
    denominator = cp.zeros(shape, dtype=cp.float64)

    for ma_type, signal in signals.items():
        if ma_type not in equities:
            continue

        equity = equities[ma_type]
        w = weights.get(ma_type, 1.0) if weights else 1.0

        # Accumulate: num += sig * eq * w
        # We can use elementwise operations
        term = signal * equity
        if w != 1.0:
            term *= w

        numerator += term

        # Accumulate denominator: den += eq * w
        eq_term = equity
        if w != 1.0:
            eq_term = eq_term * w

        denominator += eq_term

    # Safe division
    # Avoid division by zero (replace with 0)

    result = cp.zeros_like(numerator)
    mask = denominator != 0

    # Only divide where denominator != 0
    result = cp.where(mask, numerator / denominator, 0.0)

    return result
