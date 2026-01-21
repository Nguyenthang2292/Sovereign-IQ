"""
GPU-accelerated equity calculation kernels.
"""

from __future__ import annotations

import logging

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

logger = logging.getLogger(__name__)

# CUDA Kernel for Equity Calculation (Scan)
# Each thread handles one signal (one row) and iterates over time.
# This exploits the parallelism across multiple signals/symbols.
_equity_scan_kernel_source = r"""
extern "C" __global__
void equity_scan_kernel(
    const double* sig_prev,      // (N_signals, N_bars)
    const double* r_values,      // (N_signals, N_bars) - Pre-expanded R to match signals
    const double* starting_equity, // (N_signals,)
    double* out_equity,          // (N_signals, N_bars)
    int n_signals,
    int n_bars,
    double decay_multiplier,
    int cutout
) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id >= n_signals) return;

    double prev_e = starting_equity[id];

    // Stride for this signal's row
    int row_offset = id * n_bars;

    for (int t = 0; t < n_bars; t++) {
        int idx = row_offset + t;

        if (t < cutout) {
            out_equity[idx] = NAN;
            continue;
        }

        double s_val = sig_prev[idx];
        double r_val = r_values[idx];

        // Logic from _calculate_equity_core
        // a = 0 if nan or 0
        // a = r if s > 0
        // a = -r if s < 0

        double a = 0.0;

        // Check NaNs (standard way in CUDA? usually strict comparison fails)
        // isnan() is available in CUDA C++ but this is C interface?
        // simple hack: if (s!=s) is strict nan check.

        bool s_nan = (s_val != s_val);
        bool r_nan = (r_val != r_val);

        if (s_nan || r_nan || s_val == 0.0) {
            a = 0.0;
        } else if (s_val > 0.0) {
            a = r_val;
        } else {
            a = -r_val;
        }

        // Calculate current equity
        // If prev_e is nan (initially nan?), treat as start?
        // Our prev_e starts as starting_equity[id], so valid.

        double e_curr = (prev_e * decay_multiplier) * (1.0 + a);

        // Floor
        if (e_curr < 0.25) {
            e_curr = 0.25;
        }

        out_equity[idx] = e_curr;
        prev_e = e_curr;
    }
}
"""


def _get_equity_kernel():
    if not _HAS_CUPY:
        return None
    return cp.RawKernel(_equity_scan_kernel_source, "equity_scan_kernel")


def calculate_equity_gpu(
    sig_prev: cp.ndarray,
    r_values: cp.ndarray,
    starting_equities: cp.ndarray,
    decay_multiplier: float,
    cutout: int,
) -> cp.ndarray:
    """
    Calculate equity curves on GPU.

    Args:
        sig_prev: Signal array shifted by 1 (N_signals, N_bars)
        r_values: Returns array (Must be same shape as sig_prev, broadcast if needed)
        starting_equities: (N_signals,)
        decay_multiplier: float
        cutout: int

    Returns:
        Equity array (N_signals, N_bars)
    """
    if not _HAS_CUPY:
        return None

    kernel = _get_equity_kernel()
    if kernel is None:
        return None

    n_signals, n_bars = sig_prev.shape

    # Ensure inputs are contiguous C arrays
    sig_prev = cp.ascontiguousarray(sig_prev, dtype=cp.float64)
    r_values = cp.ascontiguousarray(r_values, dtype=cp.float64)
    starting_equities = cp.ascontiguousarray(starting_equities, dtype=cp.float64)

    out_equity = cp.zeros((n_signals, n_bars), dtype=cp.float64)

    # Grid config
    block_size = 128
    grid_size = (n_signals + block_size - 1) // block_size

    kernel(
        (grid_size,),
        (block_size,),
        (sig_prev, r_values, starting_equities, out_equity, n_signals, n_bars, float(decay_multiplier), cutout),
    )

    return out_equity
