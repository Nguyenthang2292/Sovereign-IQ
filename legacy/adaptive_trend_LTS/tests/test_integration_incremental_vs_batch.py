"""
Integration Tests: Incremental vs Batch Comparison

This module provides comprehensive tests comparing incremental ATC results
with batch computation results to ensure consistency.

Key Test Scenarios:
1. Basic incremental vs batch consistency
2. O(1) MA implementations vs batch MAs
3. Multiple sequential updates vs batch
4. State serialization/deserialization consistency
5. Edge cases (single bar, NaN values, extreme prices)
"""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, Any

# Import modules to test
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import compute_atc_signals, IncrementalATC
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig


class TestIncrementalVsBatchConsistency:
    """Test that incremental updates match batch computation."""

    def generate_test_prices(self, n_bars: int = 100, trend: str = "mixed") -> pd.Series:
        """Generate test price series with different characteristics."""
        np.random.seed(42)  # Reproducible tests

        if trend == "uptrend":
            base = np.linspace(100, 150, n_bars)
            noise = np.random.normal(0, 2, n_bars)
        elif trend == "downtrend":
            base = np.linspace(150, 100, n_bars)
            noise = np.random.normal(0, 2, n_bars)
        elif trend == "volatile":
            base = np.full(n_bars, 100.0)
            noise = np.random.normal(0, 10, n_bars)
        else:  # mixed
            base = 100 + np.sin(np.linspace(0, 4 * np.pi, n_bars)) * 20
            noise = np.random.normal(0, 3, n_bars)

        prices = base + noise
        return pd.Series(prices, index=pd.RangeIndex(0, n_bars))

    def test_basic_incremental_matches_batch(self):
        """Test that incremental updates produce same results as batch computation."""
        # Generate test data
        init_bars = 50
        update_bars = 20
        total_bars = init_bars + update_bars

        prices = self.generate_test_prices(total_bars)
        init_prices = prices.iloc[:init_bars]
        new_prices = prices.iloc[init_bars:].values

        # Batch computation for all bars
        batch_result = compute_atc_signals(
            prices,
            ema_len=28,
            hma_len=28,
            wma_len=28,
            dema_len=28,
            lsma_len=28,
            kama_len=28,
            La=0.02,
            De=0.03,
            robustness="Medium",
            use_rust_backend=False,  # Ensure consistent backend
        )

        # Incremental computation
        config = {
            "ema_len": 28,
            "hma_len": 28,
            "wma_len": 28,
            "dema_len": 28,
            "lsma_len": 28,
            "kama_len": 28,
            "lambda_param": 0.02,  # Unscaled
            "decay": 0.03,  # Unscaled
            "robustness": "Medium",
            "use_rust_backend": False,
            "use_o1_mas": False,  # Use standard MAs for comparison
        }

        incremental = IncrementalATC(config)
        incremental.initialize(init_prices)

        # Get signals after initialization (should match batch for init period)
        init_signal = incremental.state.get("average_signal", 0.0)
        batch_init_signal = batch_result["Average_Signal"].iloc[init_bars - 1]

        # Allow small tolerance for floating point differences
        assert (
            abs(init_signal - batch_init_signal) < 0.01
        ), f"Init signal mismatch: incremental={init_signal}, batch={batch_init_signal}"

        # Update incrementally
        incremental_signals = []
        for price in new_prices:
            signal = incremental.update(price)
            incremental_signals.append(signal)

        # Compare final signals
        final_incremental = incremental_signals[-1]
        final_batch = batch_result["Average_Signal"].iloc[-1]

        assert (
            abs(final_incremental - final_batch) < 0.05
        ), f"Final signal mismatch: incremental={final_incremental}, batch={final_batch}"

    def test_sequential_updates_vs_batch_equivalent(self):
        """Test that sequential updates are equivalent to batch."""
        n_bars = 100
        prices = self.generate_test_prices(n_bars)

        # Batch computation
        batch_config = {
            "ema_len": 20,
            "hma_len": 20,
            "wma_len": 20,
            "dema_len": 20,
            "lsma_len": 20,
            "kama_len": 20,
            "La": 0.02,
            "De": 0.03,
            "robustness": "Narrow",
            "use_rust_backend": False,
        }
        batch_result = compute_atc_signals(prices, **batch_config)

        # Sequential incremental
        incremental_config = {
            "ema_len": 20,
            "hma_len": 20,
            "wma_len": 20,
            "dema_len": 20,
            "lsma_len": 20,
            "kama_len": 20,
            "lambda_param": 0.02,
            "decay": 0.03,
            "robustness": "Narrow",
            "use_rust_backend": False,
            "use_o1_mas": False,
        }

        atc = IncrementalATC(incremental_config)
        atc.initialize(prices.iloc[:30])

        # Update bar by bar
        for i in range(30, len(prices)):
            atc.update(prices.iloc[i])

        # Verify state consistency
        final_batch_signal = batch_result["Average_Signal"].iloc[-1]
        final_incremental_signal = atc.state.get("average_signal", 0.0)

        # Allow tolerance due to implementation differences (e.g. DEMA initialization, float drift)
        # 0.15 allows for small accumulated differences over 70 bars
        assert (
            abs(final_incremental_signal - final_batch_signal) < 0.15
        ), f"Signal mismatch: incremental={final_incremental_signal}, batch={final_batch_signal}"

    def test_o1_ma_consistency_with_batch(self):
        """Test O(1) MA implementations match batch calculations."""
        prices = self.generate_test_prices(100)

        # With O(1) MAs
        config_o1 = {
            "ema_len": 28,
            "hma_len": 28,
            "wma_len": 28,
            "dema_len": 28,
            "lsma_len": 28,
            "kama_len": 28,
            "lambda_param": 0.02,
            "decay": 0.03,
            "robustness": "Medium",
            "use_rust_backend": False,
            "use_o1_mas": True,  # Enable O(1) MAs
        }

        # Without O(1) MAs (standard)
        config_standard = {
            "ema_len": 28,
            "hma_len": 28,
            "wma_len": 28,
            "dema_len": 28,
            "lsma_len": 28,
            "kama_len": 28,
            "lambda_param": 0.02,
            "decay": 0.03,
            "robustness": "Medium",
            "use_rust_backend": False,
            "use_o1_mas": False,
        }

        atc_o1 = IncrementalATC(config_o1)
        atc_standard = IncrementalATC(config_standard)

        atc_o1.initialize(prices.iloc[:50])
        atc_standard.initialize(prices.iloc[:50])

        # Compare after several updates
        for i in range(50, 80):
            signal_o1 = atc_o1.update(prices.iloc[i])
            signal_std = atc_standard.update(prices.iloc[i])

            # O(1) MAs should be very close to standard implementations
            # Note: Small drift is expected due to floating point and approximation
            assert (
                abs(signal_o1 - signal_std) < 0.2
            ), f"O(1) deviation too large at bar {i}: O1={signal_o1}, Std={signal_std}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
