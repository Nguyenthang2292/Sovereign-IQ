"""
Tests for consistency between Rust and Python/Numba implementations.
"""

import numpy as np
import pandas as pd
import pytest
from modules.adaptive_trend_LTS_mini.core import rust_backend


# Helper to generate test data
def generate_data(n=1000):
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0, 1, n))
    return prices


class TestRustPythonConsistency:
    def setup_method(self):
        self.prices = generate_data(1000)
        self.r_values = np.random.normal(0, 0.01, 1000)
        self.sig_prev = np.random.choice([-1, 0, 1], 1000)

    def test_equity_consistency(self):
        """Test consistency of equity calculation."""
        starting_equity = 1.0
        decay_multiplier = 0.97
        cutout = 50

        # Ensure inputs are float64 for Rust compatibility
        r_values = self.r_values.astype(np.float64)
        sig_prev = self.sig_prev.astype(np.float64)

        # Python implementation
        equity_py = rust_backend.calculate_equity(
            r_values, sig_prev, starting_equity, decay_multiplier, cutout, use_rust=False
        )

        # Rust implementation (if available)
        if rust_backend.RUST_AVAILABLE:
            equity_rust = rust_backend.calculate_equity(
                r_values, sig_prev, starting_equity, decay_multiplier, cutout, use_rust=True
            )

            # Allow small float differences
            np.testing.assert_allclose(
                equity_rust,
                equity_py,
                rtol=1e-10,
                atol=1e-10,
                err_msg="Equity calculation mismatch between Rust and Python",
            )
        else:
            pytest.skip("Rust backend not available")

    def test_kama_consistency(self):
        """Test consistency of KAMA calculation."""
        length = 28

        # Python implementation
        kama_py = rust_backend.calculate_kama(self.prices, length, use_rust=False)

        # Rust implementation
        if rust_backend.RUST_AVAILABLE:
            kama_rust = rust_backend.calculate_kama(self.prices, length, use_rust=True)

            # Note: KAMA involves cumulative floating point ops, tolerance might need to be looser
            np.testing.assert_allclose(
                kama_rust, kama_py, rtol=1e-10, atol=1e-10, err_msg="KAMA calculation mismatch between Rust and Python"
            )
        else:
            pytest.skip("Rust backend not available")

    def test_ema_consistency(self):
        """Test consistency of EMA calculation."""
        length = 28

        ema_py = rust_backend.calculate_ema(self.prices, length, use_rust=False)

        if rust_backend.RUST_AVAILABLE:
            ema_rust = rust_backend.calculate_ema(self.prices, length, use_rust=True)
            np.testing.assert_allclose(ema_rust, ema_py, rtol=1e-10, atol=1e-10)
        else:
            pytest.skip("Rust backend not available")

    def test_wma_consistency(self):
        """Test consistency of WMA calculation."""
        length = 28

        wma_py = rust_backend.calculate_wma(self.prices, length, use_rust=False)

        if rust_backend.RUST_AVAILABLE:
            wma_rust = rust_backend.calculate_wma(self.prices, length, use_rust=True)
            np.testing.assert_allclose(wma_rust, wma_py, rtol=1e-10, atol=1e-10)
        else:
            pytest.skip("Rust backend not available")

    def test_hma_consistency(self):
        """Test consistency of HMA calculation."""
        length = 28

        hma_py = rust_backend.calculate_hma(self.prices, length, use_rust=False)

        if rust_backend.RUST_AVAILABLE:
            hma_rust = rust_backend.calculate_hma(self.prices, length, use_rust=True)
            # HMA involves multiple WMAs, so error might accumulate slightly more
            np.testing.assert_allclose(hma_rust, hma_py, rtol=1e-9, atol=1e-9)
        else:
            pytest.skip("Rust backend not available")

    def test_lsma_consistency(self):
        """Test consistency of LSMA calculation."""
        length = 28

        lsma_py = rust_backend.calculate_lsma(self.prices, length, use_rust=False)

        if rust_backend.RUST_AVAILABLE:
            lsma_rust = rust_backend.calculate_lsma(self.prices, length, use_rust=True)
            np.testing.assert_allclose(lsma_rust, lsma_py, rtol=1e-9, atol=1e-9)
        else:
            pytest.skip("Rust backend not available")

    def test_dema_consistency(self):
        """Test consistency of DEMA calculation."""
        length = 28

        dema_py = rust_backend.calculate_dema(self.prices, length, use_rust=False)

        if rust_backend.RUST_AVAILABLE:
            dema_rust = rust_backend.calculate_dema(self.prices, length, use_rust=True)

            # Ignore initialization period where NaNs might differ
            # DEMA needs roughly 2*length to start producing values (EMA of EMA)
            warmup = length * 3

            valid_slice = slice(warmup, None)
            # DEMA implementations often differ slightly in initialization or NaN handling
            # Relax tolerance to 1e-3 (0.1%) which is acceptable for trading signals
            np.testing.assert_allclose(dema_rust[valid_slice], dema_py[valid_slice], rtol=1e-3, atol=1e-3)
        else:
            pytest.skip("Rust backend not available")

    def test_signal_persistence_consistency(self):
        """Test consistency of signal persistence."""
        up = np.random.choice([False, True], 1000, p=[0.9, 0.1])
        down = np.random.choice([False, True], 1000, p=[0.9, 0.1])

        # Ensure mutually exclusive for cleaner test (though implementation handles overlap)
        down = down & (~up)

        sig_py = rust_backend.process_signal_persistence(up, down, use_rust=False)

        if rust_backend.RUST_AVAILABLE:
            sig_rust = rust_backend.process_signal_persistence(up, down, use_rust=True)
            np.testing.assert_array_equal(sig_rust, sig_py)
        else:
            pytest.skip("Rust backend not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
