"""GPU backend for Phase 4 CUDA kernels (equity, MA, signals).

Uses PyCUDA to compile and launch custom CUDA kernels. Fallback to CuPy/NumPy
when CUDA is unavailable.
"""

from __future__ import annotations

__all__ = ["calculate_equity_cuda", "EQUITY_CUDA_AVAILABLE"]

try:
    from .equity_cuda import calculate_equity_cuda
    EQUITY_CUDA_AVAILABLE = True
except ImportError:
    calculate_equity_cuda = None  # type: ignore[misc, assignment]
    EQUITY_CUDA_AVAILABLE = False
