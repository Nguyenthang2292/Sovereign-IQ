"""
Validation tests for Incremental ATC (Phase 6).

This module verifies correctness of incremental ATC computation by:
1. Comparing incremental updates against full recalculation
2. Testing state persistence and recovery
3. Validating memory efficiency
4. Ensuring signal consistency
"""

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_LTS.core.compute_atc_signals import IncrementalATC, compute_atc_signals
from modules.adaptive_trend_LTS.utils.config import ATCConfig


@pytest.fixture
def sample_config():
    """Standard ATC configuration for testing."""
    return {
        "ema_len": 28,
        "hma_len": 28,
        "wma_len": 28,
        "dema_len": 28,
        "lsma_len": 28,
        "kama_len": 28,
        "ema_w": 1.0,
        "hma_w": 1.0,
        "wma_w": 1.0,
        "dema_w": 1.0,
        "lsma_w": 1.0,
        "kama_w": 1.0,
        "La": 0.02,
        "De": 0.03,
        "long_threshold": 0.1,
        "short_threshold": -0.1,
    }


@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    np.random.seed(42)
    base_price = 100.0
    n = 200
    returns = np.random.normal(0.001, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)
    return pd.Series(prices, index=range(n))


class TestIncrementalATCCorrectness:
    """Test correctness of incremental ATC vs full recalculation."""

    def test_initialization(self, sample_config, sample_prices):
        """Test that initialization produces valid results."""
        atc = IncrementalATC(sample_config)
        results = atc.initialize(sample_prices)

        # Verify results structure
        assert isinstance(results, dict)
        assert "Average_Signal" in results
        assert len(results["Average_Signal"]) == len(sample_prices)

        # Verify state was initialized
        assert atc.state["initialized"] is True
        assert atc.state["ma_values"] is not None
        assert atc.state["equity"] is not None

    def test_incremental_vs_full_calculation(self, sample_config, sample_prices):
        """Test that incremental updates match full recalculation."""
        # Initialize with first N-10 prices
        init_prices = sample_prices[:-10]
        atc = IncrementalATC(sample_config)
        atc.initialize(init_prices)

        # Update incrementally with remaining prices
        incremental_signals = []
        for price in sample_prices[-10:]:
            signal = atc.update(price)
            incremental_signals.append(signal)

        # Full recalculation on all prices
        full_results = compute_atc_signals(sample_prices, **sample_config)
        full_signals = full_results["Average_Signal"].iloc[-10:].values

        # Compare (allow small numerical differences due to floating point)
        np.testing.assert_allclose(
            incremental_signals,
            full_signals,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Incremental signals don't match full recalculation",
        )

    def test_single_update_accuracy(self, sample_config, sample_prices):
        """Test accuracy of a single incremental update."""
        # Initialize with all but last price
        init_prices = sample_prices[:-1]
        atc = IncrementalATC(sample_config)
        atc.initialize(init_prices)

        # Update with last price
        incremental_signal = atc.update(sample_prices.iloc[-1])

        # Full calculation with all prices
        full_results = compute_atc_signals(sample_prices, **sample_config)
        full_signal = full_results["Average_Signal"].iloc[-1]

        # Should match very closely
        assert abs(incremental_signal - full_signal) < 1e-3, f"Incremental: {incremental_signal}, Full: {full_signal}"

    def test_state_persistence(self, sample_config, sample_prices):
        """Test that state is correctly maintained between updates."""
        atc = IncrementalATC(sample_config)
        atc.initialize(sample_prices[:-5])

        # Store initial state
        initial_ma_values = atc.state["ma_values"].copy()
        initial_equity = atc.state["equity"].copy() if atc.state["equity"] else None

        # Update once
        atc.update(sample_prices.iloc[-5])

        # State should have changed
        assert atc.state["ma_values"] != initial_ma_values
        if initial_equity:
            assert atc.state["equity"] != initial_equity

    def test_reset_functionality(self, sample_config, sample_prices):
        """Test that reset properly clears state."""
        atc = IncrementalATC(sample_config)
        atc.initialize(sample_prices)

        # Reset
        atc.reset()

        # State should be cleared
        assert atc.state["initialized"] is False
        assert atc.state["ma_values"] == {}
        assert atc.state["equity"] is None
        assert len(atc.state["price_history"]) == 0


class TestIncrementalATCEdgeCases:
    """Test edge cases and error handling."""

    def test_update_before_initialize(self, sample_config):
        """Test that updating before initialization raises error."""
        atc = IncrementalATC(sample_config)

        with pytest.raises(RuntimeError, match="Must call initialize"):
            atc.update(100.0)

    def test_minimum_data_length(self, sample_config):
        """Test behavior with minimum required data."""
        # Need at least max(ma_lengths) + 1 for initialization
        min_length = 30  # Slightly above ma_len=28
        prices = pd.Series(np.random.randn(min_length) + 100)

        atc = IncrementalATC(sample_config)
        results = atc.initialize(prices)

        # Should not raise errors
        assert isinstance(results, dict)
        assert "Average_Signal" in results

    def test_extreme_price_movements(self, sample_config):
        """Test stability with extreme price movements."""
        # Create prices with large jumps
        prices = pd.Series([100.0] * 50 + [200.0] * 50 + [50.0] * 50)

        atc = IncrementalATC(sample_config)
        atc.initialize(prices[:-10])

        # Update with extreme values
        for price in prices[-10:]:
            signal = atc.update(price)
            # Signal should still be finite
            assert np.isfinite(signal)
            # Signal should be in reasonable range
            assert -10 < signal < 10


class TestIncrementalATCPerformance:
    """Test performance characteristics of incremental ATC."""

    def test_memory_efficiency(self, sample_config, sample_prices):
        """Test that incremental ATC uses limited memory."""
        import sys

        atc = IncrementalATC(sample_config)
        atc.initialize(sample_prices[:-50])

        # Get size of state
        state_size = sys.getsizeof(atc.state)

        # Update 50 times
        for price in sample_prices[-50:]:
            atc.update(price)

        # State size should not grow significantly
        new_state_size = sys.getsizeof(atc.state)
        growth_ratio = new_state_size / state_size

        assert growth_ratio < 1.5, f"State grew by {growth_ratio:.1f}x, should be nearly constant"

    def test_update_is_fast(self, sample_config, sample_prices):
        """Test that incremental update is significantly faster than full calc."""
        import time

        atc = IncrementalATC(sample_config)
        atc.initialize(sample_prices[:-1])

        # Time incremental update
        start = time.perf_counter()
        for _ in range(100):
            atc.update(sample_prices.iloc[-1])
        incremental_time = time.perf_counter() - start

        # Time full calculation
        start = time.perf_counter()
        for _ in range(100):
            compute_atc_signals(sample_prices, **sample_config)
        full_time = time.perf_counter() - start

        # Incremental should be at least 5x faster
        speedup = full_time / incremental_time
        assert speedup > 5, f"Incremental only {speedup:.1f}x faster, expected >5x"


class TestIncrementalATCIntegration:
    """Integration tests for real-world usage scenarios."""

    def test_streaming_simulation(self, sample_config, sample_prices):
        """Simulate streaming price updates (live trading scenario)."""
        # Initialize with historical data
        historical = sample_prices[:150]
        atc = IncrementalATC(sample_config)
        atc.initialize(historical)

        # Simulate streaming updates
        streaming_signals = []
        for price in sample_prices[150:]:
            signal = atc.update(price)
            streaming_signals.append(signal)

        # Verify all signals are valid
        assert len(streaming_signals) == len(sample_prices) - 150
        assert all(np.isfinite(s) for s in streaming_signals)

    def test_multiple_resets(self, sample_config, sample_prices):
        """Test behavior with multiple reset/initialize cycles."""
        atc = IncrementalATC(sample_config)

        for _ in range(3):
            # Initialize
            atc.initialize(sample_prices[:-10])

            # Update a few times
            for price in sample_prices[-10:-5]:
                signal = atc.update(price)
                assert np.isfinite(signal)

            # Reset
            atc.reset()
            assert not atc.state["initialized"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
