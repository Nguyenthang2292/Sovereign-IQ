"""
Phase 4 - Task 2.1.2: Python wrapper for equity CUDA kernel.

Compiles equity_kernel.cu via PyCUDA, provides calculate_equity_cuda(...)
with CPU↔GPU transfer and error handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule

    _PYCUDA_AVAILABLE = True
except ImportError:
    _PYCUDA_AVAILABLE = False
    drv = None
    SourceModule = None  # type: ignore[misc, assignment]

_EQ_KERNEL_SRC: Optional[str] = None
_EQ_MODULE: Optional["object"] = None  # SourceModule


def _load_kernel_source() -> str:
    global _EQ_KERNEL_SRC
    if _EQ_KERNEL_SRC is not None:
        return _EQ_KERNEL_SRC
    p = Path(__file__).resolve().parent / "equity_kernel.cu"
    _EQ_KERNEL_SRC = p.read_text(encoding="utf-8")
    return _EQ_KERNEL_SRC


def _get_equity_module():
    """Compile and cache the equity kernel module."""
    global _EQ_MODULE
    if _EQ_MODULE is not None:
        return _EQ_MODULE
    drv.init()
    src = _load_kernel_source()
    base = Path(__file__).resolve().parent
    _cache = base / ".pycuda_cache"
    _cache.mkdir(parents=True, exist_ok=True)
    _EQ_MODULE = SourceModule(src, cache_dir=str(_cache))
    return _EQ_MODULE


def calculate_equity_cuda(
    r_values: np.ndarray,
    sig_prev: np.ndarray,
    starting_equity: float,
    decay_multiplier: float,
    cutout: int,
) -> np.ndarray:
    """Run equity calculation on GPU via custom CUDA kernel.

    Matches logic in rust_extensions/src/equity.rs and compute_equity/core.py.
    Requires PyCUDA and a CUDA-capable GPU.

    Args:
        r_values: Return values (n,) float64.
        sig_prev: Previous signals (n,) float64.
        starting_equity: Initial equity.
        decay_multiplier: Decay factor (1 - De).
        cutout: Bars to skip at start; output[:cutout] = NaN.

    Returns:
        Equity array (n,) float64. NaN for indices < cutout.

    Raises:
        RuntimeError: If PyCUDA unavailable or kernel launch fails.
    """
    if not _PYCUDA_AVAILABLE or drv is None or SourceModule is None:
        raise RuntimeError("PyCUDA is required for calculate_equity_cuda. Install with: pip install pycuda")

    n = int(len(r_values))
    if n != len(sig_prev):
        raise ValueError("r_values and sig_prev must have same length")

    r = np.ascontiguousarray(r_values, dtype=np.float64)
    s = np.ascontiguousarray(sig_prev, dtype=np.float64)
    out = np.full(n, np.nan, dtype=np.float64)

    mod = _get_equity_module()
    kernel = mod.get_function("equity_kernel")

    kernel(
        drv.In(r),
        drv.In(s),
        drv.Out(out),
        np.float64(starting_equity),
        np.float64(decay_multiplier),
        np.int32(cutout),
        np.int32(n),
        block=(1, 1, 1),
        grid=(1, 1),
    )
    return out
