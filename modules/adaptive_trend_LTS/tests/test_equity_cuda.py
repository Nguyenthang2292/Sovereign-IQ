"""
Phase 4 – Task 2.1.2: Unit tests for equity CUDA kernel.

Compares calculate_equity_cuda vs _calculate_equity_core (Numba) reference.
Skips if PyCUDA/CUDA unavailable or kernel compile fails (e.g. PermissionError on Temp).
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.adaptive_trend_LTS.core.compute_equity import _calculate_equity_core

try:
    from modules.adaptive_trend_LTS.core.gpu_backend import (
        EQUITY_CUDA_AVAILABLE,
        calculate_equity_cuda,
    )
except ImportError:
    EQUITY_CUDA_AVAILABLE = False
    calculate_equity_cuda = None


def _run_cuda(r: np.ndarray, sig: np.ndarray, start: float, decay: float, cutout: int):
    """Run CUDA kernel; return None if compile/launch fails."""
    if not EQUITY_CUDA_AVAILABLE or calculate_equity_cuda is None:
        return None
    try:
        return calculate_equity_cuda(r, sig, start, decay, cutout)
    except (RuntimeError, OSError, PermissionError) as e:
        pytest.skip(f"Equity CUDA unavailable: {e}")


@pytest.mark.skipif(not EQUITY_CUDA_AVAILABLE, reason="PyCUDA / equity CUDA not available")
class TestEquityCudaCorrectness:
    """Correctness of equity CUDA kernel vs Numba reference."""

    def test_basic_equity(self):
        """Match Rust/core.py test: r=[0.01,0.02,-0.01], sig=[1,1,-1], decay=1, cutout=0."""
        r = np.array([0.01, 0.02, -0.01], dtype=np.float64)
        sig = np.array([1.0, 1.0, -1.0], dtype=np.float64)
        ref = _calculate_equity_core(r, sig, 100.0, 1.0, 0)
        out = _run_cuda(r, sig, 100.0, 1.0, 0)
        if out is None:
            pytest.skip("CUDA kernel compile/launch failed")
        np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-9)

    def test_cutout(self):
        """First cutout elements are NaN; rest match reference."""
        r = np.array([0.01, 0.02, -0.01, 0.03], dtype=np.float64)
        sig = np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float64)
        cutout = 2
        ref = _calculate_equity_core(r, sig, 100.0, 1.0, cutout)
        out = _run_cuda(r, sig, 100.0, 1.0, cutout)
        if out is None:
            pytest.skip("CUDA kernel compile/launch failed")
        assert np.isnan(out[:cutout]).all()
        np.testing.assert_allclose(out[cutout:], ref[cutout:], rtol=1e-9, atol=1e-9)

    def test_decay(self):
        """With decay < 1, values differ from decay=1; still match reference."""
        r = np.array([0.01, 0.02, -0.01, 0.02], dtype=np.float64)
        sig = np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float64)
        decay = 0.97
        ref = _calculate_equity_core(r, sig, 100.0, decay, 0)
        out = _run_cuda(r, sig, 100.0, decay, 0)
        if out is None:
            pytest.skip("CUDA kernel compile/launch failed")
        np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-9)

    def test_larger_length(self):
        """Larger n (e.g. 1000) still matches reference."""
        n = 1000
        np.random.seed(42)
        r = np.random.randn(n).astype(np.float64) * 0.01
        sig = np.where(np.random.rand(n) > 0.5, 1.0, -1.0).astype(np.float64)
        ref = _calculate_equity_core(r, sig, 100.0, 0.97, 10)
        out = _run_cuda(r, sig, 100.0, 0.97, 10)
        if out is None:
            pytest.skip("CUDA kernel compile/launch failed")
        np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-9)
