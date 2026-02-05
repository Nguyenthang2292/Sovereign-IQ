"""
Validation tests for Multi-Timeframe Incremental ATC (Phase 9 Task 3).

This module verifies correctness of MTF incremental ATC computation by:
1. Testing TF resolution mapping and bar completion logic
2. Validating initialization with multiple timeframes
3. Verifying proper TF synchronization during updates
4. Ensuring higher TFs only advance when their bars complete
5. Comparing MTF signals against separate IncrementalATC instances
"""

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend_LTS_mini.core.compute_atc_signals import (
    IncrementalATC,
    MultiTimeframeIncrementalATC,
)


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
        "use_o1_mas": False,
        "use_rust_incremental": False,
    }


@pytest.fixture
def sample_prices_1m():
    """Generate sample 1-minute price data for testing."""
    np.random.seed(42)
    base_price = 100.0
    n = 200
    returns = np.random.normal(0.001, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)
    return pd.Series(prices, index=range(n))


@pytest.fixture
def sample_prices_multi(sample_prices_1m):
    """Generate sample price data for multiple timeframes."""
    np.random.seed(42)

    prices_5m = sample_prices_1m.iloc[::5].reset_index(drop=True)
    prices_15m = sample_prices_1m.iloc[::15].reset_index(drop=True)

    return {
        "1m": sample_prices_1m,
        "5m": prices_5m,
        "15m": prices_15m,
    }


class TestTFResolutionLogic:
    """Test TF resolution mapping and bar completion logic."""

    def test_tf_resolution_map(self):
        """Test that TF resolution mapping is correct."""
        from modules.adaptive_trend_LTS_mini.core.compute_atc_signals.incremental_atc import (
            TF_RESOLUTION_MAP,
        )

        assert TF_RESOLUTION_MAP["1m"] == 1
        assert TF_RESOLUTION_MAP["5m"] == 5
        assert TF_RESOLUTION_MAP["15m"] == 15
        assert TF_RESOLUTION_MAP["30m"] == 30
        assert TF_RESOLUTION_MAP["1h"] == 60
        assert TF_RESOLUTION_MAP["4h"] == 240
        assert TF_RESOLUTION_MAP["1d"] == 1440

    def test_1m_to_5m_bar_completion(self, sample_config):
        """Test that 5m bar completes after 5x 1m bars."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m"])

        for i in range(10):
            completed = mtf._is_bar_completed(i, "5m")
            expected = (i + 1) % 5 == 0
            assert completed == expected, f"Bar {i}: expected {expected}, got {completed}"

    def test_1m_to_15m_bar_completion(self, sample_config):
        """Test that 15m bar completes after 15x 1m bars."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "15m"])

        for i in range(30):
            completed = mtf._is_bar_completed(i, "15m")
            expected = (i + 1) % 15 == 0
            assert completed == expected, f"Bar {i}: expected {expected}, got {completed}"

    def test_5m_to_15m_bar_completion(self, sample_config):
        """Test that 15m bar completes after 3x 5m bars."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["5m", "15m"])

        for i in range(10):
            completed = mtf._is_bar_completed(i, "15m")
            expected = (i + 1) % 3 == 0
            assert completed == expected, f"Bar {i}: expected {expected}, got {completed}"

    def test_bars_per_tf(self, sample_config):
        """Test _bars_per_tf helper method."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m", "15m", "1h"])

        assert mtf._bars_per_tf("1m") == 1
        assert mtf._bars_per_tf("5m") == 5
        assert mtf._bars_per_tf("15m") == 15
        assert mtf._bars_per_tf("1h") == 60


class TestMTFInitialization:
    """Test initialization of MultiTimeframeIncrementalATC."""

    def test_init_creates_atc_per_tf(self, sample_config):
        """Test that initialization creates one IncrementalATC per TF."""
        timeframes = ["1m", "5m", "15m"]
        mtf = MultiTimeframeIncrementalATC(sample_config, timeframes)

        assert "atcs" in mtf.__dict__
        assert len(mtf.atcs) == 3
        assert all(tf in mtf.atcs for tf in timeframes)
        assert all(isinstance(mtf.atcs[tf], IncrementalATC) for tf in timeframes)

    def test_init_sets_base_tf(self, sample_config):
        """Test that base timeframe is correctly identified."""
        timeframes = ["1m", "5m", "15m"]
        mtf = MultiTimeframeIncrementalATC(sample_config, timeframes)

        assert mtf.base_tf == "1m"
        assert mtf.higher_tfs == ["5m", "15m"]

    def test_init_creates_bar_counters(self, sample_config):
        """Test that bar counters are initialized for each TF."""
        timeframes = ["1m", "5m", "15m"]
        mtf = MultiTimeframeIncrementalATC(sample_config, timeframes)

        assert "bar_counters" in mtf.__dict__
        assert len(mtf.bar_counters) == 3
        assert all(tf in mtf.bar_counters for tf in timeframes)
        assert all(mtf.bar_counters[tf] == 0 for tf in timeframes)

    def test_initialize_with_dict(self, sample_config, sample_prices_multi):
        """Test initialization with dict of price series."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m", "15m"])

        results = mtf.initialize(sample_prices_multi)

        assert isinstance(results, dict)
        assert all(tf in results for tf in ["1m", "5m", "15m"])

        for tf in ["1m", "5m", "15m"]:
            assert mtf.atcs[tf].state["initialized"] is True

    def test_initialize_with_single_series(self, sample_config, sample_prices_1m):
        """Test initialization with single price series."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m"])

        results = mtf.initialize(sample_prices_1m)

        assert isinstance(results, dict)
        assert "1m" in results
        assert "5m" in results

        for tf in ["1m", "5m"]:
            assert mtf.atcs[tf].state["initialized"] is True

    def test_default_timeframes(self, sample_config):
        """Test default timeframes parameter."""
        mtf = MultiTimeframeIncrementalATC(sample_config)

        assert mtf.timeframes == ["1m", "5m", "15m"]
        assert mtf.base_tf == "1m"
        assert len(mtf.atcs) == 3


class TestMTFUpdates:
    """Test update method and TF synchronization."""

    def test_single_update(self, sample_config, sample_prices_1m):
        """Test a single update."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m"])
        mtf.initialize(sample_prices_1m[:-10])

        signals = mtf.update(sample_prices_1m.iloc[-10])

        assert isinstance(signals, dict)
        assert "1m" in signals
        assert "5m" in signals
        assert isinstance(signals["1m"], float)
        # 5m signal might be None if bar hasn't completed
        assert signals["5m"] is None or isinstance(signals["5m"], float)

    def test_base_bar_counter_increments(self, sample_config, sample_prices_1m):
        """Test that base TF bar counter increments each update."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m"])
        mtf.initialize(sample_prices_1m[:-15])

        for i in range(1, 6):
            mtf.update(sample_prices_1m.iloc[-15 + i])
            # After i-th update (starting from 1), counter should be at i
            assert mtf.bar_counters["1m"] == i, f"Expected {i}, got {mtf.bar_counters['1m']}"

    def test_5m_updates_every_5_1m_bars(self, sample_config, sample_prices_1m):
        """Test that 5m ATC updates every 5x 1m updates."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m"])
        mtf.initialize(sample_prices_1m[:-20])

        initial_5m_state = mtf.get_state("5m").copy()

        updates_5m = 0
        for i in range(15):
            price = sample_prices_1m.iloc[-20 + i]
            mtf.update(price)

            if (i + 1) % 5 == 0:
                updates_5m += 1
                current_5m_state = mtf.get_state("5m")
                assert mtf.bar_counters["5m"] == updates_5m

        assert updates_5m == 3

    def test_15m_updates_every_15_1m_bars(self, sample_config, sample_prices_1m):
        """Test that 15m ATC updates every 15x 1m updates."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m", "15m"])
        mtf.initialize(sample_prices_1m[:-30])

        updates_15m = 0
        for i in range(30):
            price = sample_prices_1m.iloc[-30 + i]
            signals = mtf.update(price)

            if (i + 1) % 15 == 0:
                updates_15m += 1
                assert mtf.bar_counters["15m"] == updates_15m

        assert updates_15m == 2

    def test_5m_to_15m_updates_every_3_bars(self, sample_config, sample_prices_1m):
        """Test that when base is 5m, 15m updates every 3 bars."""
        prices_5m = sample_prices_1m.iloc[::5].reset_index(drop=True)
        mtf = MultiTimeframeIncrementalATC(sample_config, ["5m", "15m"])
        mtf.initialize(prices_5m[:-15])

        updates_15m = 0
        for i in range(15):
            price = prices_5m.iloc[-15 + i]
            mtf.update(price)

            if (i + 1) % 3 == 0:
                updates_15m += 1
                assert mtf.bar_counters["15m"] == updates_15m

        assert updates_15m == 5

    def test_signals_dict_structure(self, sample_config, sample_prices_1m):
        """Test that signals dict contains all timeframes."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m", "15m"])
        mtf.initialize(sample_prices_1m[:-10])

        signals = mtf.update(sample_prices_1m.iloc[-10])

        assert isinstance(signals, dict)
        assert len(signals) == 3
        assert "1m" in signals
        assert "5m" in signals
        assert "15m" in signals

    def test_non_base_tf_update_warning(self, sample_config, sample_prices_1m):
        """Test that direct updates to non-base TF are handled."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["5m", "15m"])
        mtf.initialize(sample_prices_1m.iloc[::5].reset_index(drop=True)[:-5])

        signals = mtf.update(sample_prices_1m.iloc[-1], "15m")

        assert isinstance(signals, dict)
        assert "5m" in signals
        assert "15m" in signals


class TestMTFReset:
    """Test reset functionality."""

    def test_reset_clears_all_states(self, sample_config, sample_prices_1m):
        """Test that reset clears all timeframe states."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m", "15m"])
        mtf.initialize(sample_prices_1m[:-10])

        for i in range(5):
            mtf.update(sample_prices_1m.iloc[-10 + i])

        mtf.reset()

        for tf in ["1m", "5m", "15m"]:
            assert mtf.atcs[tf].state["initialized"] is False
            assert mtf.bar_counters[tf] == 0
            assert mtf.last_bar_prices.get(tf) is None

    def test_reset_allows_reinit(self, sample_config, sample_prices_1m):
        """Test that reset allows re-initialization."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m"])
        mtf.initialize(sample_prices_1m[:-10])
        mtf.update(sample_prices_1m.iloc[-10])
        mtf.reset()

        results = mtf.initialize(sample_prices_1m[:-5])

        assert all(tf in results for tf in ["1m", "5m"])
        assert all(mtf.atcs[tf].state["initialized"] for tf in ["1m", "5m"])


class TestMTFHelpers:
    """Test helper methods."""

    def test_get_state_all(self, sample_config, sample_prices_1m):
        """Test get_state returns all TFs when no TF specified."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m"])
        mtf.initialize(sample_prices_1m[:-10])

        states = mtf.get_state()

        assert isinstance(states, dict)
        assert len(states) == 2
        assert "1m" in states
        assert "5m" in states
        assert all("initialized" in states[tf] for tf in states)

    def test_get_state_single_tf(self, sample_config, sample_prices_1m):
        """Test get_state returns specific TF when specified."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m"])
        mtf.initialize(sample_prices_1m[:-10])

        state_1m = mtf.get_state("1m")

        assert isinstance(state_1m, dict)
        assert state_1m["initialized"] is True

    def test_get_signal_all(self, sample_config, sample_prices_1m):
        """Test get_signal returns all TF signals when no TF specified."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m", "15m"])
        mtf.initialize(sample_prices_1m[:-10])
        mtf.update(sample_prices_1m.iloc[-10])

        signals = mtf.get_signal()

        assert isinstance(signals, dict)
        assert len(signals) == 3
        # 5m and 15m might be None if bars haven't completed
        assert isinstance(signals["1m"], float)
        assert signals["5m"] is None or isinstance(signals["5m"], float)
        assert signals["15m"] is None or isinstance(signals["15m"], float)

    def test_get_signal_single_tf(self, sample_config, sample_prices_1m):
        """Test get_signal returns specific TF signal when specified."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m"])
        mtf.initialize(sample_prices_1m[:-10])
        mtf.update(sample_prices_1m.iloc[-10])

        signal_1m = mtf.get_signal("1m")

        assert isinstance(signal_1m, float)
        assert np.isfinite(signal_1m)


class TestMTFComparisonWithSeparate:
    """Compare MTF signals against separate IncrementalATC instances."""

    def test_mtf_vs_separate_single_update(self, sample_config, sample_prices_1m):
        """Test MTF signals match separate IncrementalATC for single update."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m"])
        mtf.initialize(sample_prices_1m[:-20])

        separate_1m = IncrementalATC(sample_config)
        separate_5m = IncrementalATC(sample_config)

        separate_1m.initialize(sample_prices_1m[:-20])
        separate_5m.initialize(sample_prices_1m[::5].reset_index(drop=True)[:-4])

        mtf_signals = mtf.update(sample_prices_1m.iloc[-20])
        separate_1m_signal = separate_1m.update(sample_prices_1m.iloc[-20])

        np.testing.assert_allclose(mtf_signals["1m"], separate_1m_signal, rtol=1e-3, atol=1e-4)

    def test_mtf_vs_separate_15_updates(self, sample_config, sample_prices_1m):
        """Test MTF signals match separate IncrementalATC over 15 1m updates."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m", "15m"])
        mtf.initialize(sample_prices_1m[:-30])

        separate_1m = IncrementalATC(sample_config)
        separate_5m = IncrementalATC(sample_config)
        separate_15m = IncrementalATC(sample_config)

        separate_1m.initialize(sample_prices_1m[:-30])
        separate_5m.initialize(sample_prices_1m[::5].reset_index(drop=True)[:-6])
        separate_15m.initialize(sample_prices_1m[::15].reset_index(drop=True)[:-2])

        for i in range(15):
            price = sample_prices_1m.iloc[-30 + i]
            mtf_signal = mtf.update(price)
            separate_1m_signal = separate_1m.update(price)

            np.testing.assert_allclose(mtf_signal["1m"], separate_1m_signal, rtol=1e-3, atol=1e-4)

            if (i + 1) % 5 == 0:
                price_5m = sample_prices_1m[::5].reset_index(drop=True).iloc[-6 + (i + 1) // 5]
                separate_5m_signal = separate_5m.update(price_5m)
                np.testing.assert_allclose(mtf_signal["5m"], separate_5m_signal, rtol=1e-3, atol=1e-4)

            if (i + 1) % 15 == 0:
                price_15m = sample_prices_1m[::15].reset_index(drop=True).iloc[-2 + (i + 1) // 15]
                separate_15m_signal = separate_15m.update(price_15m)
                np.testing.assert_allclose(mtf_signal["15m"], separate_15m_signal, rtol=1e-3, atol=1e-4)


class TestMTFEdgeCases:
    """Test edge cases."""

    def test_update_before_initialize(self, sample_config):
        """Test that update before initialization fails gracefully."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m", "5m"])

        with pytest.raises(RuntimeError):
            mtf.atcs["1m"].update(100.0)

    def test_custom_timeframes(self, sample_config):
        """Test initialization with custom timeframes."""
        custom_tfs = ["15m", "30m", "1h"]
        mtf = MultiTimeframeIncrementalATC(sample_config, custom_tfs)

        assert mtf.timeframes == custom_tfs
        assert mtf.base_tf == "15m"
        assert mtf.higher_tfs == ["30m", "1h"]
        assert all(tf in mtf.atcs for tf in custom_tfs)

    def test_single_tf(self, sample_config, sample_prices_1m):
        """Test MTF with single timeframe (should still work)."""
        mtf = MultiTimeframeIncrementalATC(sample_config, ["1m"])
        mtf.initialize(sample_prices_1m[:-10])

        signal = mtf.update(sample_prices_1m.iloc[-10])

        assert isinstance(signal, dict)
        assert "1m" in signal
        assert np.isfinite(signal["1m"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
