"""Tests for Rust incremental ATC backend.

This module verifies correctness of Rust incremental backend by:
1. Comparing Rust backend against Python backend
2. Testing edge cases
3. Testing state serialization/deserialization
"""

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_LTS.core.compute_atc_signals.incremental_backend import (
    check_rust_available,
    update_incremental_rust,
    update_incremental_python,
    update_incremental_auto,
)


@pytest.fixture
def sample_config():
    """Standard ATC configuration for testing."""
    return {
        "ema_len": 20,
        "hma_len": 20,
        "wma_len": 20,
        "dema_len": 20,
        "lsma_len": 20,
        "kama_len": 20,
        "La": 0.02,
        "De": 0.03,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
    }


@pytest.fixture
def sample_initial_state(sample_config):
    """Create initial state for testing."""
    return {
        "ema_value": 100.0,
        "ema2_value": 100.0,
        "wma_length": 20,
        "wma_denominator": 210.0,
        "wma_window": [100.0] * 20,
        "wma_weighted_sum": 100.0,
        "wma_value": 100.0,
        "wma_initialized": True,
        "hma_half_length": 10,
        "hma_half_denominator": 55.0,
        "hma_half_window": [100.0] * 10,
        "hma_half_weighted_sum": 100.0,
        "hma_half_value": 100.0,
        "hma_half_initialized": True,
        "hma_full_length": 20,
        "hma_full_denominator": 210.0,
        "hma_full_window": [100.0] * 20,
        "hma_full_weighted_sum": 100.0,
        "hma_full_value": 100.0,
        "hma_full_initialized": True,
        "hma_final_length": 4,
        "hma_final_denominator": 10.0,
        "hma_final_window": [100.0] * 4,
        "hma_final_weighted_sum": 100.0,
        "hma_final_value": 100.0,
        "hma_final_initialized": True,
        "hma_intermediate_series": [100.0] * 4,
        "hma_value": 100.0,
        "hma_initialized": True,
        "lsma_length": 20,
        "lsma_x_values": [float(i) for i in range(20)],
        "lsma_sum_x": 190.0,
        "lsma_sum_x2": 2470.0,
        "lsma_denom": 2470.0 * 20.0 - 190.0 * 190.0,
        "lsma_window": [100.0] * 20,
        "lsma_sum_y": 2000.0,
        "lsma_sum_xy": 100.0,
        "lsma_value": 100.0,
        "lsma_initialized": True,
        "kama_length": 20,
        "kama_fast_sc": 2.0 / 3.0,
        "kama_slow_sc": 2.0 / 31.0,
        "kama_window": [100.0] * 21,
        "kama_volatility_sum": 0.0,
        "kama_value": 100.0,
        "kama_initialized": True,
        "equity_ema": 1.0,
        "equity_hma": 1.0,
        "equity_wma": 1.0,
        "equity_dema": 1.0,
        "equity_lsma": 1.0,
        "equity_kama": 1.0,
        "decay": 0.0003,
        "la": 0.00002,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
        "ema_length": 20,
        "price_window": [100.0] * 21,
        "initialized": True,
    }


class TestRustBackendAvailability:
    """Test Rust backend availability checks."""

    def test_check_rust_available(self):
        """Test that we can check Rust backend availability."""
        is_available = check_rust_available()
        assert isinstance(is_available, bool)
        # Note: We don't assert True or False as it depends on environment

    def test_import_error_when_rust_not_available(self):
        """Test that appropriate error is raised when trying to use unavailable Rust."""
        if check_rust_available():
            pytest.skip("Rust backend is available")
        
        state = {"initialized": True}
        with pytest.raises(ImportError, match="Rust incremental backend is not available"):
            update_incremental_rust(state, 100.0, {})


class TestRustVsPythonConsistency:
    """Test consistency between Rust and Python backends."""

    @pytest.mark.skipif(not check_rust_available(), reason="Rust backend not available")
    def test_rust_vs_python_single_update(self, sample_initial_state, sample_config):
        """Test that Rust and Python produce same results for single update."""
        new_price = 101.0
        
        # Update with Rust
        state_rust = sample_initial_state.copy()
        signal_rust, state_rust = update_incremental_rust(state_rust, new_price, sample_config)
        
        # Update with Python (fallback)
        state_python = sample_initial_state.copy()
        signal_python, state_python = update_incremental_python(state_python, new_price, sample_config)
        
        # Signals should be the same (or very close)
        assert abs(signal_rust - signal_python) < 1e-6, \
            f"Rust signal ({signal_rust}) != Python signal ({signal_python})"
    
    @pytest.mark.skipif(not check_rust_available(), reason="Rust backend not available")
    def test_rust_vs_python_sequence(self, sample_initial_state, sample_config):
        """Test that Rust and Python produce same results for a sequence of updates."""
        np.random.seed(42)
        prices = np.random.randn(100) * 10 + 100
        
        state_rust = sample_initial_state.copy()
        state_python = sample_initial_state.copy()
        
        signals_rust = []
        signals_python = []
        
        for price in prices:
            signal_rust, state_rust = update_incremental_rust(state_rust, price, sample_config)
            signal_python, state_python = update_incremental_python(state_python, price, sample_config)
            
            signals_rust.append(signal_rust)
            signals_python.append(signal_python)
        
        # All signals should be very close
        for i, (sr, sp) in enumerate(zip(signals_rust, signals_python)):
            assert abs(sr - sp) < 1e-5, \
                f"Mismatch at index {i}: Rust={sr}, Python={sp}"


class TestRustBackendEdgeCases:
    """Test edge cases for Rust backend."""

    @pytest.mark.skipif(not check_rust_available(), reason="Rust backend not available")
    def test_constant_price_series(self, sample_initial_state, sample_config):
        """Test Rust backend with constant price series."""
        state = sample_initial_state.copy()
        
        signals = []
        for _ in range(50):
            signal, state = update_incremental_rust(state, 100.0, sample_config)
            signals.append(signal)
            assert not np.isnan(signal), "Signal should not be NaN"
            assert not np.isinf(signal), "Signal should not be inf"
        
        # All signals should be valid
        assert all(np.isfinite(s) for s in signals)

    @pytest.mark.skipif(not check_rust_available(), reason="Rust backend not available")
    def test_extreme_price_jumps(self, sample_initial_state, sample_config):
        """Test Rust backend with extreme price movements."""
        state = sample_initial_state.copy()
        
        # Simulate extreme jumps
        prices = [100.0, 200.0, 50.0, 150.0, 300.0, 20.0]
        
        signals = []
        for price in prices:
            signal, state = update_incremental_rust(state, price, sample_config)
            signals.append(signal)
            assert not np.isnan(signal), "Signal should not be NaN"
            assert not np.isinf(signal), "Signal should not be inf"
            assert -10 < signal < 10, f"Signal {signal} should be in reasonable range"
        
        assert len(signals) == len(prices)

    @pytest.mark.skipif(not check_rust_available(), reason="Rust backend not available")
    def test_zero_price(self, sample_initial_state, sample_config):
        """Test Rust backend with zero price."""
        state = sample_initial_state.copy()
        
        signal, state = update_incremental_rust(state, 0.0, sample_config)
        
        assert not np.isnan(signal), "Signal should not be NaN"
        assert not np.isinf(signal), "Signal should not be inf"


class TestRustStateSerialization:
    """Test state serialization and deserialization."""

    @pytest.mark.skipif(not check_rust_available(), reason="Rust backend not available")
    def test_state_structure(self, sample_initial_state):
        """Test that state has expected structure."""
        required_keys = [
            "ema_value", "ema2_value", "wma_value", "hma_value", "lsma_value", "kama_value",
            "equity_ema", "equity_hma", "equity_wma", "equity_dema", "equity_lsma", "equity_kama",
            "initialized"
        ]
        
        for key in required_keys:
            assert key in sample_initial_state, f"State missing key: {key}"

    @pytest.mark.skipif(not check_rust_available(), reason="Rust backend not available")
    def test_state_persistence_across_updates(self, sample_initial_state, sample_config):
        """Test that state changes correctly across updates."""
        state = sample_initial_state.copy()
        initial_ema = state["ema_value"]
        
        # Update with new price
        new_price = 101.0
        signal, state = update_incremental_rust(state, new_price, sample_config)
        
        # EMA should have changed
        assert state["ema_value"] != initial_ema, "EMA should change after update"
        
        # State should still be valid
        assert state["initialized"] == True


class TestAutoBackendSelection:
    """Test automatic backend selection."""

    def test_auto_uses_rust_when_available(self, sample_initial_state, sample_config):
        """Test that auto uses Rust when available."""
        if not check_rust_available():
            pytest.skip("Rust backend not available")
        
        config = sample_config.copy()
        config["use_rust_incremental"] = True
        
        # This should use Rust backend without errors
        signal, state = update_incremental_auto(sample_initial_state.copy(), 100.0, config)
        
        assert not np.isnan(signal)

    def test_auto_falls_back_to_python(self, sample_initial_state, sample_config):
        """Test that auto falls back to Python when Rust disabled."""
        config = sample_config.copy()
        config["use_rust_incremental"] = False
        
        # This should use Python backend without errors
        signal, state = update_incremental_auto(sample_initial_state.copy(), 100.0, config)
        
        assert not np.isnan(signal)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
