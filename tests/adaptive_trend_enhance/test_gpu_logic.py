import os
import sys

import numpy as np
import pytest

# Add project root to path (adjusted for depth of tests/adaptive_trend_enhance/test_gpu_logic.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

from modules.adaptive_trend_enhance.core.process_layer1._gpu_equity import calculate_equity_gpu
from modules.adaptive_trend_enhance.core.process_layer1._gpu_signals import (
    generate_signal_from_ma_gpu,
    rate_of_change_gpu,
)


@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy not installed or no GPU")
class TestGPULogic:
    def test_rate_of_change_gpu(self):
        # [10, 11, 12.1]
        # R = (11-10)/10 = 0.1
        # R = (12.1-11)/11 = 0.1
        prices = cp.array([[10.0, 11.0, 12.1]])
        r_gpu = rate_of_change_gpu(prices, length=1)
        r_cpu = r_gpu.get()

        # r_cpu[0,0] should be nan or 0 depending on implementation.
        # My impl returns nan at start? Let's check impl.
        # impl: pad[:] = cp.nan. So it is nan.
        assert np.isnan(r_cpu[0, 0])
        assert np.isclose(r_cpu[0, 1], 0.1)
        assert np.isclose(r_cpu[0, 2], 0.1)

    def test_generate_signal_persistence(self):
        # Price: [10, 12, 11, 9, 10]
        # MA:    [10, 10, 10, 10, 10]
        # Diff:  [0, 2, 1, -1, 0]
        # RawSig:[0, 1, 1, -1, 0]
        # Persist:[0, 1, 1, -1, -1] -> 0s should hold previous

        prices = cp.array([[10, 12, 11, 9, 10]], dtype=cp.float64)
        ma = cp.full_like(prices, 10.0)

        sig_gpu = generate_signal_from_ma_gpu(prices, ma)
        sig_cpu = sig_gpu.get()

        expected = np.array([[0, 1, 1, -1, -1]])
        np.testing.assert_array_equal(sig_cpu, expected)

    def test_equity_calculation(self):
        # Simple Scenario:
        # Start = 1.0
        # Sig (shifted): [0, 1, 1, -1]
        # R: [0, 0.1, -0.1, 0.1]
        # Decay: 1.0 (No decay)
        # Cutout: 0

        # t=0: nan (or handled) -> e=1.0?
        #   kernel: if t<cutout -> nan. If cutout=0, t=0 processes.
        #   t=0: sig=0 -> a=0 -> e = 1.0 * 1 * 1 = 1.0
        # t=1: sig=1 -> a=R[1]=0.1 -> e = 1.0 * (1.1) = 1.1
        # t=2: sig=1 -> a=R[2]=-0.1 -> e = 1.1 * (0.9) = 0.99
        # t=3: sig=-1 -> a=-R[3]=-0.1 -> e = 0.99 * (0.9) = 0.891

        sig = cp.array([[0, 1, 1, -1]], dtype=cp.float64)
        r = cp.array([[0, 0.1, -0.1, 0.1]], dtype=cp.float64)
        start = cp.array([1.0], dtype=cp.float64)

        eq_gpu = calculate_equity_gpu(sig, r, start, decay_multiplier=1.0, cutout=0)
        eq_cpu = eq_gpu.get()

        assert np.isclose(eq_cpu[0, 0], 1.0)
        assert np.isclose(eq_cpu[0, 1], 1.1)
        assert np.isclose(eq_cpu[0, 2], 0.99)
        assert np.isclose(eq_cpu[0, 3], 0.891)
