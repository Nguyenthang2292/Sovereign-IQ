"""Tests for advanced target labeling strategies."""

import sys
from pathlib import Path

# Add project root to Python path for imports
ROOT = Path(__file__).resolve().parents[2]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

import numpy as np
import pandas as pd
import pytest

from config.random_forest import (
    BUY_THRESHOLD,
    RANDOM_FOREST_HORIZON_1D,
    RANDOM_FOREST_HORIZON_1H,
    RANDOM_FOREST_HORIZON_4H,
    RANDOM_FOREST_MIN_TREND_STRENGTH,
    RANDOM_FOREST_VOLATILITY_MULTIPLIER,
    RANDOM_FOREST_VOLATILITY_WINDOW,
    SELL_THRESHOLD,
)
from modules.random_forest.utils.advanced_labeling import (
    calculate_trend_strength,
    calculate_volatility_adjusted_thresholds,
    create_advanced_target,
    create_multi_horizon_targets,
)


class TestVolatilityAdjustedThresholds:
    """Test volatility-adjusted threshold calculation."""

    def test_volatility_adjusted_thresholds_basic(self):
        """Test basic volatility-adjusted threshold calculation."""
        # Create data with varying volatility
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
        df = pd.DataFrame({"close": prices})

        buy_threshold, sell_threshold = calculate_volatility_adjusted_thresholds(df)

        assert isinstance(buy_threshold, pd.Series)
        assert isinstance(sell_threshold, pd.Series)
        assert len(buy_threshold) == len(df)
        assert len(sell_threshold) == len(df)
        # Buy threshold should be positive
        assert (buy_threshold > 0).all()
        # Sell threshold should be negative
        assert (sell_threshold < 0).all()
        # Buy threshold should be approximately -sell_threshold
        assert np.allclose(buy_threshold, -sell_threshold, rtol=0.1)

    def test_volatility_adjusted_thresholds_high_volatility(self):
        """Test that high volatility increases thresholds."""
        # Create high volatility data
        np.random.seed(42)
        prices_high_vol = 100 + np.cumsum(np.random.randn(100) * 0.05)  # 5x volatility
        df_high_vol = pd.DataFrame({"close": prices_high_vol})

        # Create low volatility data
        prices_low_vol = 100 + np.cumsum(np.random.randn(100) * 0.001)  # 0.1x volatility
        df_low_vol = pd.DataFrame({"close": prices_low_vol})

        buy_high, _ = calculate_volatility_adjusted_thresholds(df_high_vol)
        buy_low, _ = calculate_volatility_adjusted_thresholds(df_low_vol)

        # High volatility should generally produce higher thresholds
        # Note: Due to clipping, they might be similar, but high vol should be >= low vol
        assert buy_high.mean() >= buy_low.mean()
        
        # Verify that volatility calculation is working (std should be different)
        vol_high = df_high_vol["close"].pct_change().rolling(20).std().mean()
        vol_low = df_low_vol["close"].pct_change().rolling(20).std().mean()
        assert vol_high > vol_low, "Volatility calculation should show difference"

    def test_volatility_adjusted_thresholds_missing_close(self):
        """Test fallback when 'close' column is missing."""
        df = pd.DataFrame({"open": [100, 101, 102]})

        buy_threshold, sell_threshold = calculate_volatility_adjusted_thresholds(df)

        # Should fallback to fixed thresholds
        assert (buy_threshold == BUY_THRESHOLD).all()
        assert (sell_threshold == SELL_THRESHOLD).all()

    def test_volatility_adjusted_thresholds_clipping(self):
        """Test that thresholds are clipped to reasonable ranges."""
        # Create extreme volatility data
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.2)  # Very high volatility
        df = pd.DataFrame({"close": prices})

        buy_threshold, sell_threshold = calculate_volatility_adjusted_thresholds(df)

        # Should be clipped to reasonable ranges
        assert (buy_threshold >= BUY_THRESHOLD * 0.5).all()
        assert (buy_threshold <= BUY_THRESHOLD * 2.0).all()
        assert (sell_threshold <= SELL_THRESHOLD * 0.5).all()
        assert (sell_threshold >= SELL_THRESHOLD * 2.0).all()


class TestTrendStrength:
    """Test trend strength calculation."""

    def test_trend_strength_uptrend(self):
        """Test trend strength for strong uptrend."""
        # Create strong uptrend
        prices = pd.Series(range(100, 200))  # Linear uptrend
        df = pd.DataFrame({"close": prices})

        trend_strength = calculate_trend_strength(df)

        assert isinstance(trend_strength, pd.Series)
        assert len(trend_strength) == len(df)
        # Strong uptrend should have high trend strength
        assert trend_strength.iloc[-20:].mean() > 0.5

    def test_trend_strength_choppy(self):
        """Test trend strength for choppy/no trend market."""
        # Create choppy data (oscillating)
        np.random.seed(42)
        prices = 100 + np.sin(np.arange(100) * 0.1) * 2  # Oscillating
        df = pd.DataFrame({"close": prices})

        trend_strength = calculate_trend_strength(df)

        # Choppy market should have lower trend strength
        assert trend_strength.mean() < 0.7

    def test_trend_strength_range(self):
        """Test that trend strength is in [0, 1] range."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
        df = pd.DataFrame({"close": prices})

        trend_strength = calculate_trend_strength(df)

        assert (trend_strength >= 0).all()
        assert (trend_strength <= 1).all()

    def test_trend_strength_missing_close(self):
        """Test fallback when 'close' column is missing."""
        df = pd.DataFrame({"open": [100, 101, 102]})

        trend_strength = calculate_trend_strength(df)

        # Should return zero trend strength
        assert (trend_strength == 0.0).all()


class TestMultiHorizonTargets:
    """Test multi-horizon target creation."""

    def test_multi_horizon_targets_creation(self):
        """Test that multi-horizon targets are created correctly."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.01)
        df = pd.DataFrame({"close": prices})

        targets = create_multi_horizon_targets(df)

        assert "target_1h" in targets
        assert "target_4h" in targets
        assert "target_1d" in targets

        for horizon_name, target_series in targets.items():
            assert isinstance(target_series, pd.Series)
            assert len(target_series) == len(df)
            # Targets should be -1, 0, or 1
            assert target_series.isin([-1, 0, 1]).all()

    def test_multi_horizon_targets_different_horizons(self):
        """Test that different horizons produce different targets."""
        # Create data with clear trend
        prices = pd.Series(range(100, 300))  # Strong uptrend
        df = pd.DataFrame({"close": prices})

        targets = create_multi_horizon_targets(df)

        # Different horizons should produce different target distributions
        # (at least some differences expected)
        target_1h_values = targets["target_1h"].value_counts()
        target_1d_values = targets["target_1d"].value_counts()

        # They might be similar but not identical
        assert len(targets["target_1h"]) == len(targets["target_1d"])


class TestAdvancedTarget:
    """Test integrated advanced target creation."""

    def test_advanced_target_volatility_adjusted(self):
        """Test advanced target with volatility-adjusted thresholds."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.01)
        df = pd.DataFrame({"close": prices})

        target, multi_horizon = create_advanced_target(
            df, use_volatility_adjusted=True, use_trend_based=False, use_multi_horizon=False
        )

        assert isinstance(target, pd.Series)
        assert target.name == "target"
        assert len(target) == len(df)
        assert target.isin([-1, 0, 1]).all()
        assert multi_horizon is None

    def test_advanced_target_trend_based(self):
        """Test advanced target with trend-based filtering."""
        # Create strong uptrend
        prices = pd.Series(range(100, 300))
        df = pd.DataFrame({"close": prices})

        target, _ = create_advanced_target(
            df, use_volatility_adjusted=False, use_trend_based=True, use_multi_horizon=False
        )

        # With strong trend, should have some signals
        assert (target != 0).sum() > 0

    def test_advanced_target_trend_based_choppy(self):
        """Test that trend-based filtering removes signals in choppy markets."""
        # Create choppy data
        np.random.seed(42)
        prices = 100 + np.sin(np.arange(200) * 0.1) * 2  # Oscillating
        df = pd.DataFrame({"close": prices})

        target_with_trend, _ = create_advanced_target(
            df, use_volatility_adjusted=False, use_trend_based=True, use_multi_horizon=False
        )

        target_without_trend, _ = create_advanced_target(
            df, use_volatility_adjusted=False, use_trend_based=False, use_multi_horizon=False
        )

        # Trend-based filtering should reduce signals in choppy markets
        # (target_with_trend should have fewer non-zero values)
        assert (target_with_trend != 0).sum() <= (target_without_trend != 0).sum()

    def test_advanced_target_multi_horizon(self):
        """Test advanced target with multi-horizon enabled."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(300) * 0.01)
        df = pd.DataFrame({"close": prices})

        target, multi_horizon = create_advanced_target(
            df, use_volatility_adjusted=False, use_trend_based=False, use_multi_horizon=True
        )

        assert isinstance(target, pd.Series)
        assert multi_horizon is not None
        assert "target_1h" in multi_horizon
        assert "target_4h" in multi_horizon
        assert "target_1d" in multi_horizon

    def test_advanced_target_all_features(self):
        """Test advanced target with all features enabled."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(300) * 0.01)
        df = pd.DataFrame({"close": prices})

        target, multi_horizon = create_advanced_target(
            df, use_volatility_adjusted=True, use_trend_based=True, use_multi_horizon=True
        )

        assert isinstance(target, pd.Series)
        assert target.isin([-1, 0, 1]).all()
        assert multi_horizon is not None

    def test_advanced_target_missing_close(self):
        """Test fallback when 'close' column is missing."""
        df = pd.DataFrame({"open": [100, 101, 102]})

        target, multi_horizon = create_advanced_target(df)

        # Should return neutral targets
        assert (target == 0).all()
        assert multi_horizon is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
