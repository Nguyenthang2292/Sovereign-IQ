
import numpy as np
import pandas as pd
import pytest

from modules.range_oscillator.analysis.summary import get_signal_summary
from modules.range_oscillator.config import CombinedStrategyConfig, StrategySpecificConfig
from modules.range_oscillator.strategies.combined import CombinedStrategy, generate_signals_combined_all_strategy
from modules.range_oscillator.strategies.combined import CombinedStrategy, generate_signals_combined_all_strategy

"""
Tests for range_oscillator analysis module.

Tests analysis functions and classes:
- CombinedStrategy class
- get_signal_summary (additional edge cases)
"""




@pytest.fixture
def sample_ohlc_data():
    """Create sample OHLC data for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)

    return (
        pd.Series(high, index=dates, name="high"),
        pd.Series(low, index=dates, name="low"),
        pd.Series(close, index=dates, name="close"),
    )


class TestCombinedStrategy:
    """Tests for CombinedStrategy class."""

    def test_combined_strategy_default_config(self, sample_ohlc_data):
        """Test CombinedStrategy with default config."""
        high, low, close = sample_ohlc_data

        strategy = CombinedStrategy()

        signals, strength = strategy.generate_signals(high=high, low=low, close=close, length=50, mult=2.0)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert len(strength) == len(close)
        assert all(signals.isin([-1, 0, 1]))

    def test_combined_strategy_custom_config(self, sample_ohlc_data):
        """Test CombinedStrategy with custom config."""
        high, low, close = sample_ohlc_data

        config = CombinedStrategyConfig(
            enabled_strategies=[2, 3, 4],
            min_signal_strength=0.1,
            return_confidence_score=True,
            return_strategy_stats=True,
        )

        strategy = CombinedStrategy(config=config)

        signals, strength, stats, confidence = strategy.generate_signals(
            high=high, low=low, close=close, length=50, mult=2.0
        )

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert isinstance(stats, dict)
        assert isinstance(confidence, pd.Series)

    def test_combined_strategy_legacy_kwargs(self, sample_ohlc_data):
        """Test CombinedStrategy with legacy kwargs."""
        high, low, close = sample_ohlc_data

        strategy = CombinedStrategy(
            enabled_strategies=[2, 3], min_signal_strength=0.2, consensus_mode="weighted", consensus_threshold=0.6
        )

        signals, strength = strategy.generate_signals(high=high, low=low, close=close, length=50, mult=2.0)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert strategy.config.consensus.mode == "weighted"
        assert strategy.config.consensus.threshold == 0.6

    def test_combined_strategy_dynamic_selection(self, sample_ohlc_data):
        """Test CombinedStrategy with dynamic selection enabled."""
        high, low, close = sample_ohlc_data

        from modules.range_oscillator.config import DynamicSelectionConfig

        config = CombinedStrategyConfig(
            enabled_strategies=[2, 3, 4, 6, 7, 8, 9], dynamic=DynamicSelectionConfig(enabled=True, lookback=20)
        )

        strategy = CombinedStrategy(config=config)

        signals, strength = strategy.generate_signals(high=high, low=low, close=close, length=50, mult=2.0)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert config.dynamic.enabled is True

    def test_combined_strategy_adaptive_weights(self, sample_ohlc_data):
        """Test CombinedStrategy with adaptive weights."""
        high, low, close = sample_ohlc_data

        from modules.range_oscillator.config import ConsensusConfig

        config = CombinedStrategyConfig(
            enabled_strategies=[2, 3, 4], consensus=ConsensusConfig(mode="weighted", adaptive_weights=True)
        )

        strategy = CombinedStrategy(config=config)

        signals, strength = strategy.generate_signals(high=high, low=low, close=close, length=50, mult=2.0)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert config.consensus.mode == "weighted"
        assert config.consensus.adaptive_weights is True

    def test_combined_strategy_no_enabled_strategies(self, sample_ohlc_data):
        """Test CombinedStrategy with no enabled strategies (fallback)."""
        high, low, close = sample_ohlc_data

        from modules.range_oscillator.config import StrategySpecificConfig

        config = CombinedStrategyConfig(
            enabled_strategies=[],
            params=StrategySpecificConfig(use_sustained=False, use_crossover=False, use_momentum=False),
        )

        strategy = CombinedStrategy(config=config)

        signals, strength = strategy.generate_signals(high=high, low=low, close=close, length=50, mult=2.0)

        # Should fallback to basic strategy
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)

    def test_combined_strategy_input_validation(self):
        """Test CombinedStrategy input validation."""
        # Test with missing required inputs
        strategy = CombinedStrategy()

        with pytest.raises(ValueError, match="Either provide"):
            strategy.generate_signals(length=50, mult=2.0)

    def test_combined_strategy_with_precalculated_oscillator(self, sample_ohlc_data):
        """Test CombinedStrategy with pre-calculated oscillator."""
        high, low, close = sample_ohlc_data

        # Create pre-calculated oscillator data
        oscillator = pd.Series(np.sin(np.linspace(0, 4 * np.pi, len(close))) * 50, index=close.index)
        ma = close.rolling(50).mean()
        range_atr = pd.Series(np.ones(len(close)) * 1000, index=close.index)

        strategy = CombinedStrategy()

        signals, strength = strategy.generate_signals(
            oscillator=oscillator, ma=ma, range_atr=range_atr, length=50, mult=2.0
        )

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)

    def test_combined_strategy_with_dynamic_exhaustion(self, sample_ohlc_data):
        """Test CombinedStrategy with dynamic exhaustion threshold for breakout strategy."""
        high, low, close = sample_ohlc_data

        # Create config with breakout strategy enabled and dynamic exhaustion
        config = CombinedStrategyConfig(
            enabled_strategies=[6],  # Only breakout strategy
            params=StrategySpecificConfig(
                use_breakout=True,
                breakout_use_dynamic_exhaustion=True,
                breakout_exhaustion_atr_multiplier=1.0,
                breakout_base_exhaustion_threshold=150.0,
                breakout_exhaustion_atr_period=50,
            ),
        )

        strategy = CombinedStrategy(config=config)

        signals, strength = strategy.generate_signals(high=high, low=low, close=close, length=50, mult=2.0)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert all(signals.isin([-1, 0, 1]))

    def test_combined_strategy_breakout_dynamic_exhaustion_via_kwargs(self, sample_ohlc_data):
        """Test CombinedStrategy breakout dynamic exhaustion via legacy kwargs."""
        high, low, close = sample_ohlc_data

        strategy = CombinedStrategy(
            enabled_strategies=[6],  # Only breakout strategy
            use_breakout=True,
            breakout_use_dynamic_exhaustion=True,
            breakout_exhaustion_atr_multiplier=1.5,
            breakout_base_exhaustion_threshold=150.0,
            breakout_exhaustion_atr_period=50,
        )

        signals, strength = strategy.generate_signals(high=high, low=low, close=close, length=50, mult=2.0)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert all(signals.isin([-1, 0, 1]))


class TestGenerateSignalsCombinedAllStrategy:
    """Tests for generate_signals_combined_all_strategy function."""

    def test_generate_signals_combined_all_strategy_basic(self, sample_ohlc_data):
        """Test generate_signals_combined_all_strategy basic functionality."""
        high, low, close = sample_ohlc_data

        signals, strength = generate_signals_combined_all_strategy(high=high, low=low, close=close, length=50, mult=2.0)

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)

    def test_generate_signals_combined_all_strategy_with_config(self, sample_ohlc_data):
        """Test generate_signals_combined_all_strategy with config."""
        high, low, close = sample_ohlc_data

        config = CombinedStrategyConfig(enabled_strategies=[2, 3, 4], min_signal_strength=0.1)

        signals, strength = generate_signals_combined_all_strategy(
            high=high, low=low, close=close, config=config, length=50, mult=2.0
        )

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)

    def test_generate_signals_combined_all_strategy_with_kwargs(self, sample_ohlc_data):
        """Test generate_signals_combined_all_strategy with kwargs."""
        high, low, close = sample_ohlc_data

        signals, strength = generate_signals_combined_all_strategy(
            high=high, low=low, close=close, enabled_strategies=[2, 3], min_signal_strength=0.2, length=50, mult=2.0
        )

        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)


class TestGetSignalSummary:
    """Additional tests for get_signal_summary function (edge cases)."""

    def test_get_signal_summary_empty_signals(self):
        """Test get_signal_summary with empty signals."""
        dates = pd.date_range("2024-01-01", periods=0, freq="1h")
        signals = pd.Series([], dtype="int8", index=dates)
        signal_strength = pd.Series([], dtype="float64", index=dates)
        close = pd.Series([], dtype="float64", index=dates)

        summary = get_signal_summary(signals, signal_strength, close)

        assert summary["total_signals"] == 0
        assert summary["long_signals"] == 0
        assert summary["short_signals"] == 0
        assert summary["current_signal"] == 0

    def test_get_signal_summary_all_neutral(self):
        """Test get_signal_summary with all neutral signals."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        signals = pd.Series([0] * 10, index=dates, dtype="int8")
        signal_strength = pd.Series([0.0] * 10, index=dates, dtype="float64")
        close = pd.Series([100.0] * 10, index=dates)

        summary = get_signal_summary(signals, signal_strength, close)

        assert summary["total_signals"] == 10
        assert summary["long_signals"] == 0
        assert summary["short_signals"] == 0
        assert summary["neutral_signals"] == 10
        assert summary["current_signal"] == 0

    def test_get_signal_summary_index_mismatch(self):
        """Test get_signal_summary with index mismatch (should handle gracefully)."""
        dates1 = pd.date_range("2024-01-01", periods=10, freq="1h")
        dates2 = pd.date_range("2024-01-02", periods=10, freq="1h")

        signals = pd.Series([1, -1, 0, 1, -1, 0, 1, -1, 0, 1], index=dates1, dtype="int8")
        signal_strength = pd.Series([0.5] * 10, index=dates2, dtype="float64")  # Different index
        close = pd.Series([100.0] * 10, index=dates1)

        # Should handle index mismatch with reindex
        summary = get_signal_summary(signals, signal_strength, close)

        assert summary["total_signals"] == 10
        assert "long_signals" in summary
        assert "short_signals" in summary

    def test_get_signal_summary_with_nan(self):
        """Test get_signal_summary with NaN values."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        signals = pd.Series([1, -1, np.nan, 1, -1, 0, 1, -1, 0, 1], index=dates, dtype="float64")
        signals = signals.astype("Int8")  # Nullable integer type
        signal_strength = pd.Series([0.5, 0.6, np.nan, 0.7, 0.8, 0.0, 0.9, 0.4, 0.0, 0.5], index=dates)
        close = pd.Series([100.0] * 10, index=dates)

        summary = get_signal_summary(signals, signal_strength, close)

        assert summary["total_signals"] == 10
        # Should handle NaN values gracefully
        assert "current_signal" in summary

    def test_get_signal_summary_input_validation(self):
        """Test get_signal_summary input validation."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        signals = pd.Series([1, -1, 0] * 3 + [1], index=dates, dtype="int8")
        signal_strength = pd.Series([0.5] * 10, index=dates)
        close = pd.Series([100.0] * 10, index=dates)

        # Test with None inputs
        with pytest.raises(ValueError):
            get_signal_summary(None, signal_strength, close)

        with pytest.raises(ValueError):
            get_signal_summary(signals, None, close)

        with pytest.raises(ValueError):
            get_signal_summary(signals, signal_strength, None)

        # Test with non-Series inputs
        with pytest.raises(TypeError):
            get_signal_summary([1, -1, 0], signal_strength, close)

        with pytest.raises(TypeError):
            get_signal_summary(signals, [0.5, 0.6, 0.7], close)

        with pytest.raises(TypeError):
            get_signal_summary(signals, signal_strength, [100, 101, 102])

    def test_get_signal_summary_percentages(self):
        """Test get_signal_summary percentage calculations."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        signals = pd.Series([1] * 6 + [-1] * 3 + [0], index=dates, dtype="int8")
        signal_strength = pd.Series([0.5] * 10, index=dates)
        close = pd.Series([100.0] * 10, index=dates)

        summary = get_signal_summary(signals, signal_strength, close)

        assert summary["long_signals"] == 6
        assert summary["short_signals"] == 3
        assert summary["neutral_signals"] == 1
        assert summary["long_percentage"] == 60.0
        assert summary["short_percentage"] == 30.0
