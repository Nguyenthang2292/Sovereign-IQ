import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from modules.adaptive_trend_LTS.core.compute_atc_signals.incremental_atc import IncrementalATC


class TestIncrementalSerialization:
    @pytest.fixture
    def config(self):
        return {
            "ema_len": 10,
            "hull_len": 10,
            "wma_len": 10,
            "dema_len": 10,
            "lsma_len": 10,
            "kama_len": 10,
            "De": 0.03,
            "La": 0.02,
            "long_threshold": 0.1,
            "short_threshold": -0.1,
            "use_o1_mas": True,
            "use_rust_incremental": False,  # Disable Rust backend for pure Python testing
        }

    @pytest.fixture
    def prices(self):
        # Generate synthetic random walk prices
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 100)
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices)

    def test_roundtrip_serialization(self, config, prices):
        """Test that saving and loading state preserves behavior."""

        # 1. Initialize and run first instance
        atc1 = IncrementalATC(config)

        # Split data: initialization, pre-save updates, post-save updates
        init_len = 30
        pre_save_len = 20
        post_save_len = 20

        init_data = prices.iloc[:init_len]
        pre_save_data = prices.iloc[init_len : init_len + pre_save_len]
        post_save_data = prices.iloc[init_len + pre_save_len : init_len + pre_save_len + post_save_len]

        atc1.initialize(init_data)

        # Run updates
        for price in pre_save_data:
            atc1.update(price)

        # 2. Save state
        with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            atc1.save_state(tmp_path)

            # 3. Load state into new instance
            atc2 = IncrementalATC.load_state(tmp_path)

            # Verify loaded config matches
            assert atc2.config == atc1.config

            # Verify state matches immediately after load
            assert atc2.state["signal"] == atc1.state["signal"]
            assert atc2.state["initialized"] == True

            # Compare deque contents
            assert list(atc2.state["price_history"]) == list(atc1.state["price_history"])

            # Compare MA values
            for k, v in atc1.state["ma_values"].items():
                assert np.isclose(atc2.state["ma_values"][k], v, rtol=1e-10), f"MA {k} mismatch"

            # 4. Run more updates on both
            signals1 = []
            signals2 = []

            for price in post_save_data:
                s1 = atc1.update(price)
                s2 = atc2.update(price)
                signals1.append(s1)
                signals2.append(s2)

            # 5. Verify trajectories match
            np.testing.assert_allclose(signals1, signals2, rtol=1e-10, err_msg="Signals diverged after load")

            # Verify final states
            for k, v in atc1.state["ma_values"].items():
                assert np.isclose(atc2.state["ma_values"][k], v, rtol=1e-10), f"Final MA {k} mismatch"

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_legacy_fallback(self, config, prices):
        """Test that fallback to legacy (non-O(1)) works with serialization."""
        config["use_o1_mas"] = False
        atc1 = IncrementalATC(config)

        init_data = prices.iloc[:30]
        update_data = prices.iloc[30:40]

        atc1.initialize(init_data)
        for p in update_data:
            atc1.update(p)

        with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            atc1.save_state(tmp_path)
            atc2 = IncrementalATC.load_state(tmp_path)

            # Update both
            next_price = 105.0
            s1 = atc1.update(next_price)
            s2 = atc2.update(next_price)

            assert np.isclose(s1, s2, rtol=1e-10)

        finally:
            if tmp_path.exists():
                tmp_path.unlink()
