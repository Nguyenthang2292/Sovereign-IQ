"""
Batch update tests for Incremental ATC (Phase 9 Task 4).

This module verifies batch_update functionality by:
1. Comparing batch_update vs repeated update() calls
2. Testing state consistency between batch and sequential updates
3. Validating edge cases (empty, single bar, large batches)
"""

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import IncrementalATC, compute_atc_signals


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


class TestBatchUpdateCorrectness:
    """Test correctness of batch_update vs sequential updates."""

    def test_batch_vs_sequential_small_batch(self, sample_config, sample_prices):
        """Test batch_update matches repeated update() for small batches."""
        init_prices = sample_prices[:-10]
        test_prices = sample_prices[-10:]

        # Sequential updates
        atc_sequential = IncrementalATC(sample_config)
        atc_sequential.initialize(init_prices)
        sequential_signals = []
        for price in test_prices:
            sequential_signals.append(atc_sequential.update(price))

        # Batch update
        atc_batch = IncrementalATC(sample_config)
        atc_batch.initialize(init_prices)
        batch_signals = atc_batch.batch_update(test_prices.tolist())

        # Compare signals
        assert len(batch_signals) == len(sequential_signals)
        np.testing.assert_allclose(
            batch_signals,
            sequential_signals,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Batch signals don't match sequential signals",
        )

        # Compare final state
        for key in ["ma_values", "equity", "signal"]:
            assert atc_sequential.state[key] == atc_batch.state[key], f"State {key} mismatch"

    def test_batch_vs_sequential_large_batch(self, sample_config, sample_prices):
        """Test batch_update matches repeated update() for larger batches."""
        init_prices = sample_prices[:-100]
        test_prices = sample_prices[-100:]

        # Sequential updates
        atc_sequential = IncrementalATC(sample_config)
        atc_sequential.initialize(init_prices)
        sequential_signals = [atc_sequential.update(price) for price in test_prices]

        # Batch update
        atc_batch = IncrementalATC(sample_config)
        atc_batch.initialize(init_prices)
        batch_signals = atc_batch.batch_update(test_prices.tolist())

        # Compare signals
        assert len(batch_signals) == len(sequential_signals)
        np.testing.assert_allclose(
            batch_signals,
            sequential_signals,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Large batch signals don't match sequential signals",
        )

        # Compare final state
        for key in ["ma_values", "equity", "signal"]:
            assert atc_sequential.state[key] == atc_batch.state[key], f"State {key} mismatch"

    def test_batch_vs_full_calculation_consistency(self, sample_config, sample_prices):
        """Test that batch updates match full ATC calculation on the same data."""
        init_prices = sample_prices[:-20]
        test_prices = sample_prices[-20:]

        # Initialize
        atc_batch = IncrementalATC(sample_config)
        atc_batch.initialize(init_prices)

        # Batch update
        batch_signals = atc_batch.batch_update(test_prices.tolist())

        # Full calculation on test prices
        full_results = compute_atc_signals(test_prices, **sample_config)
        full_signals = full_results["Average_Signal"].values

        # Compare
        assert len(batch_signals) == len(full_signals)
        np.testing.assert_allclose(
            batch_signals,
            full_signals,
            rtol=1e-3,
            atol=1e-4,
            err_msg="Batch signals don't match full ATC calculation",
        )


class TestBatchUpdateEdgeCases:
    """Test edge cases and error handling for batch_update."""

    def test_empty_batch(self, sample_config, sample_prices):
        """Test batch_update with empty list."""
        atc = IncrementalATC(sample_config)
        atc.initialize(sample_prices)

        signals = atc.batch_update([])
        assert len(signals) == 0

    def test_single_bar_batch(self, sample_config, sample_prices):
        """Test batch_update with single price."""
        atc = IncrementalATC(sample_config)
        atc.initialize(sample_prices)

        signals = atc.batch_update([sample_prices.iloc[-1]])
        assert len(signals) == 1
        assert np.isfinite(signals[0])

    def test_batch_with_numpy_array(self, sample_config, sample_prices):
        """Test batch_update with numpy array input."""
        atc = IncrementalATC(sample_config)
        atc.initialize(sample_prices)

        batch_signals = atc.batch_update(sample_prices[-5:].values)
        assert len(batch_signals) == 5
        assert all(np.isfinite(s) for s in batch_signals)

    def test_batch_before_initialize(self, sample_config):
        """Test that batch_update before initialization raises error."""
        atc = IncrementalATC(sample_config)

        with pytest.raises(RuntimeError, match="Must call initialize"):
            atc.batch_update([100.0, 101.0, 102.0])

    def test_batch_state_consistency(self, sample_config, sample_prices):
        """Test that batch update maintains state correctly across calls."""
        atc = IncrementalATC(sample_config)
        atc.initialize(sample_prices[:-30])

        # First batch
        batch1_prices = sample_prices[-30:-15]
        signals1 = atc.batch_update(batch1_prices.tolist())

        # Verify first batch size
        assert len(signals1) == 15, f"Expected 15 signals, got {len(signals1)}"

        # Verify state
        initial_ma_values = atc.state["ma_values"].copy()

        # Second batch (continued from where we left off)
        batch2_prices = sample_prices[-15:]
        signals2 = atc.batch_update(batch2_prices.tolist())

        # Verify second batch size
        assert len(signals2) == 15, f"Expected 15 signals, got {len(signals2)}"

        # Total should match all
        total_signals = signals1 + signals2
        assert len(total_signals) == 30

        # Compare against full calculation
        full_results = compute_atc_signals(sample_prices[-30:], **sample_config)
        full_signals = full_results["Average_Signal"].values

        np.testing.assert_allclose(
            total_signals,
            full_signals,
            rtol=1e-3,
            atol=1e-4,
        )


class TestBatchUpdateIntegration:
    """Integration tests for real-world usage scenarios."""

    def test_batch_streaming_simulation(self, sample_config, sample_prices):
        """Simulate batch streaming updates (live trading scenario)."""
        atc = IncrementalATC(sample_config)
        atc.initialize(sample_prices[:150])

        batch_size = 10
        batch_signals = []
        for i in range(150, 200, batch_size):
            batch = sample_prices[i : i + batch_size]
            signals = atc.batch_update(batch.tolist())
            batch_signals.extend(signals)

        # Verify all signals are valid
        assert len(batch_signals) == 50
        assert all(np.isfinite(s) for s in batch_signals)

    def test_batch_reset_workflow(self, sample_config, sample_prices):
        """Test batch updates with reset workflow."""
        atc = IncrementalATC(sample_config)

        # First cycle
        atc.initialize(sample_prices[:-100])
        atc.batch_update(sample_prices[-100:-50].tolist())

        # Reset
        atc.reset()
        assert not atc.state["initialized"]

        # Second cycle
        atc.initialize(sample_prices[:-50])
        atc.batch_update(sample_prices[-50:].tolist())

        # Verify final state
        assert atc.state["initialized"]
        assert atc.state["signal"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
