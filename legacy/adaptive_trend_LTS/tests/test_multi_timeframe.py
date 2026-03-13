"""
Tests for Multi-Timeframe Incremental ATC.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental.multi_timeframe import MultiTimeframeIncrementalATC
from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental.constants import TF_RESOLUTION_MAP


class TestMultiTimeframeATC:
    def setup_method(self):
        self.config = {
            "lambda_param": 0.02,
            "decay": 0.03,
            "ema_len": 28,
            "hma_len": 28,
            "wma_len": 28,
            "dema_len": 28,
            "lsma_len": 28,
            "kama_len": 28,
            "robustness": "Medium",
            "use_rust_backend": False,
            "use_o1_mas": False,
        }

    def generate_data(self, n=100):
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0, 1, n))
        return pd.Series(prices)

    def test_initialization(self):
        """Test initialization with single and multiple datasets."""
        mtf = MultiTimeframeIncrementalATC(self.config, timeframes=["1m", "5m"])

        # Init with single dataset
        data_1m = self.generate_data(100)
        mtf.initialize(data_1m)

        assert mtf.atcs["1m"].state["initialized"]
        assert mtf.atcs["5m"].state["initialized"]

        # Price history is capped by maxlen in state manager
        # For default config (len=28, medium), max_history is 44
        # We assert it's filled up to capacity or data length
        max_hist_1m = mtf.atcs["1m"].state["price_history"].maxlen
        assert len(mtf.atcs["1m"].state["price_history"]) == min(100, max_hist_1m)

        max_hist_5m = mtf.atcs["5m"].state["price_history"].maxlen
        assert len(mtf.atcs["5m"].state["price_history"]) == min(100, max_hist_5m)

        # Init with dict
        data_5m = self.generate_data(20)  # 5m bars
        mtf.initialize({"1m": data_1m, "5m": data_5m})
        assert len(mtf.atcs["5m"].state["price_history"]) == min(20, max_hist_5m)

    def test_bar_completion_logic(self):
        """Test detection of higher timeframe bar completion."""
        mtf = MultiTimeframeIncrementalATC(self.config, timeframes=["1m", "5m"])

        # 5m bar completes every 5th 1m bar (index 4, 9, 14...)
        # Indices are 0-based

        assert not mtf._is_bar_completed(0, "5m")  # 1m bar 0 done -> 1/5 of 5m
        assert not mtf._is_bar_completed(3, "5m")  # 1m bar 3 done -> 4/5 of 5m
        assert mtf._is_bar_completed(4, "5m")  # 1m bar 4 done -> 5/5 of 5m (Completed!)

        assert not mtf._is_bar_completed(5, "5m")
        assert mtf._is_bar_completed(9, "5m")  # 1m bar 9 done -> 10/5 of 5m (Completed!)

    def test_update_propagation(self):
        """Test that updates propagate to higher timeframes correctly."""
        mtf = MultiTimeframeIncrementalATC(self.config, timeframes=["1m", "5m"])
        data_1m = self.generate_data(50)
        mtf.initialize(data_1m)

        # Reset bar counters for deterministic test
        mtf.bar_counters["1m"] = 0
        mtf.bar_counters["5m"] = 0

        # Mock the higher timeframe update to verify it's called
        mtf.atcs["5m"].update = Mock(return_value=0.5)

        # Update 1, 2, 3, 4 (Indices 0, 1, 2, 3) - No 5m update expected
        for i in range(4):
            mtf.update(101.0 + i)
            mtf.atcs["5m"].update.assert_not_called()

        # Update 5 (Index 4) - 5m update EXPECTED
        mtf.update(105.0)
        mtf.atcs["5m"].update.assert_called_once()

        # Verify the price passed to 5m update (should be the closing price of the 5m bar)
        # In this implementation, it uses the last price seen (105.0)
        mtf.atcs["5m"].update.assert_called_with(105.0)

    def test_multiple_timeframes(self):
        """Test with 3 timeframes: 1m, 5m, 15m."""
        mtf = MultiTimeframeIncrementalATC(self.config, timeframes=["1m", "5m", "15m"])
        data_1m = self.generate_data(60)
        mtf.initialize(data_1m)

        mtf.bar_counters["1m"] = 0
        mtf.bar_counters["5m"] = 0
        mtf.bar_counters["15m"] = 0

        mtf.atcs["5m"].update = Mock(return_value=0.5)
        mtf.atcs["15m"].update = Mock(return_value=0.8)

        # Run 15 minutes of 1m bars
        for i in range(15):
            mtf.update(100.0 + i)

        # 5m should be updated at i=4, 9, 14 (3 times)
        assert mtf.atcs["5m"].update.call_count == 3

        # 15m should be updated at i=14 (1 time)
        assert mtf.atcs["15m"].update.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
