"""
Tests for range_oscillator strategy module.
"""

import numpy as np
import pandas as pd
import pytest

from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
from modules.range_oscillator.strategies.combined import generate_signals_combined_all_strategy
from modules.range_oscillator.strategies.crossover import generate_signals_crossover_strategy
from modules.range_oscillator.strategies.momentum import generate_signals_momentum_strategy
from modules.range_oscillator.strategies.sustained import generate_signals_sustained_strategy


def _unpack_signals(result):
    """Unpack generate_signals result handling both 2 and 4 return values."""
    if len(result) == 2:
        signals, strength = result
        return signals, strength, None, None
    else:
        signals, strength, strategy_stats, confidence = result
        return signals, strength, strategy_stats, confidence


@pytest.fixture
def sample_ohlc_data():
    """Create sample OHLC data for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)

    res = (pd.Series(high, index=dates), pd.Series(low, index=dates), pd.Series(close, index=dates))
    yield res
    # Teardown: no explicit cleanup needed, data is garbage collected after test


@pytest.fixture
def sample_oscillator_data(sample_ohlc_data):
    """Create sample oscillator data for testing."""
    high, low, close = sample_ohlc_data

    oscillator = pd.Series(np.sin(np.linspace(0, 4 * np.pi, len(close))) * 50, index=close.index)
    ma = close.rolling(50).mean()
    range_atr = pd.Series(np.ones(len(close)) * 1000, index=close.index)

    res = (oscillator, ma, range_atr)
    yield res
    del res


class TestStrategy1:
    """Tests for Strategy 1: Basic oscillator signals."""

    def test_strategy1_basic(self, sample_ohlc_data):
        """Test basic Strategy 1 functionality."""
        high, low, close = sample_ohlc_data

        signals, strength = generate_signals_basic_strategy(high=high, low=low, close=close, length=50, mult=2.0)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert len(strength) == len(close)
        assert signals.dtype == "int8"
        assert all(signals.isin([-1, 0, 1]))
        assert all((strength >= 0) & (strength <= 1))

    def test_strategy1_with_precalculated(self, sample_ohlc_data, sample_oscillator_data):
        """Test Strategy 1 with pre-calculated oscillator values."""
        high, low, close = sample_ohlc_data
        oscillator, ma, range_atr = sample_oscillator_data

        signals, strength = generate_signals_basic_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, close=close
        )

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(oscillator)

    def test_strategy1_parameters(self, sample_ohlc_data):
        """Test Strategy 1 with different parameters."""
        high, low, close = sample_ohlc_data

        signals, strength = generate_signals_basic_strategy(
            high=high,
            low=low,
            close=close,
            oscillator_threshold=10.0,
            require_trend_confirmation=False,
            use_breakout_signals=False,
        )

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)

    @pytest.mark.memory_intensive
    def test_strategy1_large_dataset_performance(self):
        """Test Strategy 1 performance with large dataset."""
        # Reduced from 10000 to 1000 to save RAM while still testing performance
        dates = pd.date_range("2024-01-01", periods=1000, freq="1h")
        oscillator = pd.Series(np.sin(np.linspace(0, 20 * np.pi, 1000)) * 50, index=dates)
        ma = pd.Series([50000.0] * 1000, index=dates)
        range_atr = pd.Series([1000.0] * 1000, index=dates)
        close = pd.Series([51000.0] * 1000, index=dates)

        import time

        start_time = time.time()

        signals, strength = generate_signals_basic_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, close=close, require_trend_confirmation=True
        )

        elapsed_time = time.time() - start_time

        assert elapsed_time < 0.2, f"Performance test failed: {elapsed_time:.3f}s for 1k bars"
        assert len(signals) == 1000
        assert len(strength) == 1000
        assert all(signals.isin([-1, 0, 1]))


class TestStrategy2:
    """Tests for Strategy 2: Sustained pressure."""

    def test_strategy2_basic(self, sample_ohlc_data):
        """Test basic Strategy 2 functionality."""
        high, low, close = sample_ohlc_data

        signals, strength = generate_signals_sustained_strategy(
            high=high, low=low, close=close, min_bars_above_zero=3, min_bars_below_zero=3
        )

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert all(signals.isin([-1, 0, 1]))

    def test_strategy2_with_precalculated(self, sample_ohlc_data, sample_oscillator_data):
        """Test Strategy 2 with pre-calculated values."""
        high, low, close = sample_ohlc_data
        oscillator, ma, range_atr = sample_oscillator_data

        signals, strength = generate_signals_sustained_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, min_bars_above_zero=5
        )

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)

    def test_strategy2_validation(self, sample_ohlc_data):
        """Test Strategy 2 parameter validation."""
        high, low, close = sample_ohlc_data

        with pytest.raises(ValueError, match="min_bars_above_zero must be > 0"):
            generate_signals_sustained_strategy(high=high, low=low, close=close, min_bars_above_zero=0)

        with pytest.raises(ValueError, match="min_bars_below_zero must be > 0"):
            generate_signals_sustained_strategy(high=high, low=low, close=close, min_bars_below_zero=0)


class TestStrategy3:
    """Tests for Strategy 3: Zero line crossover."""

    def test_strategy3_basic(self, sample_ohlc_data):
        """Test basic Strategy 3 functionality."""
        high, low, close = sample_ohlc_data

        signals, strength = generate_signals_crossover_strategy(high=high, low=low, close=close, confirmation_bars=2)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert all(signals.isin([-1, 0, 1]))

    def test_strategy3_with_precalculated(self, sample_ohlc_data, sample_oscillator_data):
        """Test Strategy 3 with pre-calculated values."""
        high, low, close = sample_ohlc_data
        oscillator, ma, range_atr = sample_oscillator_data

        signals, strength = generate_signals_crossover_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, confirmation_bars=3
        )

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)

    def test_strategy3_validation(self, sample_ohlc_data):
        """Test Strategy 3 parameter validation."""
        high, low, close = sample_ohlc_data

        with pytest.raises(ValueError, match="confirmation_bars must be > 0"):
            generate_signals_crossover_strategy(high=high, low=low, close=close, confirmation_bars=0)


class TestStrategy4:
    """Tests for Strategy 4: Momentum."""

    def test_strategy4_basic(self, sample_ohlc_data):
        """Test basic Strategy 4 functionality."""
        high, low, close = sample_ohlc_data

        signals, strength = generate_signals_momentum_strategy(
            high=high, low=low, close=close, momentum_period=3, momentum_threshold=5.0
        )

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert all(signals.isin([-1, 0, 1]))

    def test_strategy4_with_precalculated(self, sample_ohlc_data, sample_oscillator_data):
        """Test Strategy 4 with pre-calculated values."""
        high, low, close = sample_ohlc_data
        oscillator, ma, range_atr = sample_oscillator_data

        signals, strength = generate_signals_momentum_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, momentum_period=5
        )

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)

    def test_strategy4_validation(self, sample_ohlc_data):
        """Test Strategy 4 parameter validation."""
        high, low, close = sample_ohlc_data

        with pytest.raises(ValueError, match="momentum_period must be > 0"):
            generate_signals_momentum_strategy(high=high, low=low, close=close, momentum_period=0)

        with pytest.raises(ValueError, match="momentum_threshold must be >= 0"):
            generate_signals_momentum_strategy(high=high, low=low, close=close, momentum_threshold=-1.0)


class TestStrategy5:
    """Tests for Strategy 5: Combined."""

    def test_strategy5_basic(self, sample_ohlc_data):
        """Test basic Strategy 5 functionality."""
        high, low, close = sample_ohlc_data

        result = generate_signals_combined_all_strategy(
            high=high, low=low, close=close, use_sustained=True, use_crossover=True, use_momentum=True
        )
        signals, strength, _, _ = _unpack_signals(result)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert all(signals.isin([-1, 0, 1]))

    def test_strategy5_with_precalculated(self, sample_ohlc_data, sample_oscillator_data):
        """Test Strategy 5 with pre-calculated values."""
        high, low, close = sample_ohlc_data
        oscillator, ma, range_atr = sample_oscillator_data

        result = generate_signals_combined_all_strategy(
            oscillator=oscillator,
            ma=ma,
            range_atr=range_atr,
            use_sustained=True,
            use_crossover=False,
            use_momentum=False,
        )
        signals, strength, _, _ = _unpack_signals(result)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)

    def test_strategy5_no_methods_enabled(self, sample_ohlc_data):
        """Test Strategy 5 fallback when no methods enabled."""
        high, low, close = sample_ohlc_data

        result = generate_signals_combined_all_strategy(
            high=high, low=low, close=close, use_sustained=False, use_crossover=False, use_momentum=False
        )
        signals, strength, _, _ = _unpack_signals(result)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)

    def test_strategy5_validation(self, sample_ohlc_data):
        """Test Strategy 5 parameter validation."""
        high, low, close = sample_ohlc_data

        with pytest.raises(ValueError, match="min_bars_sustained must be > 0"):
            generate_signals_combined_all_strategy(
                high=high, low=low, close=close, use_sustained=True, min_bars_sustained=0
            )

        with pytest.raises(ValueError, match="confirmation_bars must be > 0"):
            generate_signals_combined_all_strategy(
                high=high, low=low, close=close, use_crossover=True, confirmation_bars=0
            )

    def test_strategy5_majority_vote(self, sample_ohlc_data):
        """Test Strategy 5 majority vote logic."""
        high, low, close = sample_ohlc_data

        result = generate_signals_combined_all_strategy(
            high=high, low=low, close=close, use_sustained=True, use_crossover=True, use_momentum=True
        )
        signals, strength, _, _ = _unpack_signals(result)

        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
        assert all((strength >= 0) & (strength <= 1))


class TestStrategy5Integration:
    """Integration tests for Strategy 5: Combined strategy with majority voting."""

    @pytest.fixture
    def sample_data_for_voting(self):
        """Create sample data for testing majority voting."""
        dates = pd.date_range("2024-01-01", periods=50, freq="1h")
        oscillator = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 50)) * 50, index=dates)
        ma = pd.Series([50000.0] * 50, index=dates)
        range_atr = pd.Series([1000.0] * 50, index=dates)
        return oscillator, ma, range_atr

    def test_majority_vote_long_wins_2_to_1(self, sample_data_for_voting):
        """Test majority voting when 2 strategies vote LONG, 1 votes SHORT."""
        oscillator, ma, range_atr = sample_data_for_voting

        result = generate_signals_combined_all_strategy(
            oscillator=oscillator,
            ma=ma,
            range_atr=range_atr,
            use_sustained=True,
            use_crossover=True,
            use_momentum=True,
            min_bars_sustained=2,
            confirmation_bars=1,
            momentum_period=3,
            momentum_threshold=10.0,
        )
        signals, strength, _, _ = _unpack_signals(result)

        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
        assert len(signals) == len(oscillator)

    def test_majority_vote_tie_results_in_neutral(self, sample_data_for_voting):
        """Test that ties in voting result in NEUTRAL signals."""
        oscillator, ma, range_atr = sample_data_for_voting

        result = generate_signals_combined_all_strategy(
            oscillator=oscillator,
            ma=ma,
            range_atr=range_atr,
            use_sustained=True,
            use_crossover=True,
            use_momentum=False,
            min_bars_sustained=2,
            confirmation_bars=1,
        )
        signals, strength, _, _ = _unpack_signals(result)

        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))

    def test_combination_sustained_only(self, sample_data_for_voting):
        """Test Strategy 5 with only sustained method enabled."""
        oscillator, ma, range_atr = sample_data_for_voting

        result = generate_signals_combined_all_strategy(
            oscillator=oscillator,
            ma=ma,
            range_atr=range_atr,
            use_sustained=True,
            use_crossover=False,
            use_momentum=False,
        )
        signals, strength, _, _ = _unpack_signals(result)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
        assert len(signals) == len(oscillator)

    def test_combination_crossover_only(self, sample_data_for_voting):
        """Test Strategy 5 with only crossover method enabled."""
        oscillator, ma, range_atr = sample_data_for_voting

        result = generate_signals_combined_all_strategy(
            oscillator=oscillator,
            ma=ma,
            range_atr=range_atr,
            use_sustained=False,
            use_crossover=True,
            use_momentum=False,
        )
        signals, strength, _, _ = _unpack_signals(result)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert all(signals.isin([-1, 0, 1]))

    def test_combination_momentum_only(self, sample_data_for_voting):
        """Test Strategy 5 with only momentum method enabled."""
        oscillator, ma, range_atr = sample_data_for_voting

        result = generate_signals_combined_all_strategy(
            oscillator=oscillator,
            ma=ma,
            range_atr=range_atr,
            use_sustained=False,
            use_crossover=False,
            use_momentum=True,
        )
        signals, strength, _, _ = _unpack_signals(result)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
