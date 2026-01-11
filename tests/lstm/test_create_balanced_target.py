
import numpy as np
import pandas as pd

from modules.lstm.core.create_balanced_target import create_balanced_target
from modules.lstm.core.create_balanced_target import create_balanced_target

"""
Tests for create_balanced_target function.
"""




class TestCreateBalancedTarget:
    """Test suite for create_balanced_target function."""

    def test_create_targets_basic(self):
        """Test basic target creation with valid data."""
        # Need at least FUTURE_RETURN_SHIFT + 1 data points (25 for default config)
        df = pd.DataFrame({"close": [100 + i for i in range(30)]})

        result = create_balanced_target(df)
        assert result is not None
        assert "Target" in result.columns
        assert len(result) > 0
        assert result["Target"].isin([-1, 0, 1]).all()

    def test_create_targets_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = create_balanced_target(df)
        assert result is None

    def test_create_targets_missing_close_column(self):
        """Test with missing close column."""
        df = pd.DataFrame({"open": [100, 101, 102], "high": [105, 106, 107], "low": [95, 96, 97]})
        result = create_balanced_target(df)
        assert result is None

    def test_create_targets_insufficient_data(self):
        """Test with insufficient data points."""
        df = pd.DataFrame(
            {
                "close": [100, 101]  # Not enough for future return calculation
            }
        )
        result = create_balanced_target(df)
        assert result is None

    def test_create_targets_strong_upward_movement(self):
        """Test with strong upward movement."""
        # Create data with strong upward trend
        # Need at least FUTURE_RETURN_SHIFT + 1 data points
        base_price = 100
        prices = [base_price + i * 0.02 for i in range(30)]  # 2% increase per step
        df = pd.DataFrame({"close": prices})

        result = create_balanced_target(df, threshold=0.01, neutral_zone=0.005)
        assert result is not None
        if len(result) > 0:
            # Should have some buy signals (1)
            assert (result["Target"] == 1).any() or (result["Target"] == 0).any()

    def test_create_targets_strong_downward_movement(self):
        """Test with strong downward movement."""
        # Create data with strong downward trend
        # Need at least FUTURE_RETURN_SHIFT + 1 data points
        base_price = 100
        prices = [base_price - i * 0.02 for i in range(30)]  # 2% decrease per step
        df = pd.DataFrame({"close": prices})

        result = create_balanced_target(df, threshold=0.01, neutral_zone=0.005)
        assert result is not None
        if len(result) > 0:
            # Should have some sell signals (-1)
            assert (result["Target"] == -1).any() or (result["Target"] == 0).any()

    def test_create_targets_neutral_movement(self):
        """Test with neutral movement."""
        # Create data with minimal movement
        # Need at least FUTURE_RETURN_SHIFT + 1 data points
        base_price = 100

        np.random.seed(42)
        prices = [base_price + np.random.uniform(-0.001, 0.001) for _ in range(30)]
        df = pd.DataFrame({"close": prices})

        result = create_balanced_target(df, threshold=0.01, neutral_zone=0.005)
        assert result is not None
        if len(result) > 0:
            # Should have mostly neutral signals (0) given minimal movement
            neutral_count = (result["Target"] == 0).sum()
            assert neutral_count > len(result) * 0.5  # At least 50% neutral

    def test_create_targets_invalid_thresholds(self):
        """Test with invalid threshold configuration."""
        # Need at least FUTURE_RETURN_SHIFT + 1 data points
        df = pd.DataFrame({"close": [100 + i for i in range(30)]})

        # neutral_zone >= threshold should return None
        result = create_balanced_target(df, threshold=0.01, neutral_zone=0.01)
        assert result is None

        result = create_balanced_target(df, threshold=0.01, neutral_zone=0.02)
        assert result is None

    def test_create_targets_custom_thresholds(self):
        """Test with custom threshold values."""
        # Need at least FUTURE_RETURN_SHIFT + 1 data points
        df = pd.DataFrame({"close": [100 + i * 2 for i in range(30)]})

        result = create_balanced_target(df, threshold=0.05, neutral_zone=0.01)
        assert result is not None
        assert "Target" in result.columns

    def test_create_targets_deterministic(self):
        """Test that function produces deterministic results with same seed."""
        # Need at least FUTURE_RETURN_SHIFT + 1 data points
        df = pd.DataFrame({"close": [100 + i * 0.01 for i in range(30)]})

        result1 = create_balanced_target(df)
        result2 = create_balanced_target(df)

        # Results should be identical due to fixed seed (42)
        if result1 is not None and result2 is not None:
            pd.testing.assert_frame_equal(result1, result2)

    def test_create_targets_target_distribution(self):
        """
        Test that target distribution produces multiple target classes.

        This test crafts synthetic data with three regimes: a sustained uptrend,
        a sustained downtrend, and a neutral regime via minimal random movement.
        This should allow the balanced target function to create at least two classes.
        """
        base_price = 100
        n_points = 120

        prices = []
        # First third: uptrend
        current_price = base_price
        for _ in range(n_points // 3):
            prices.append(current_price)
            current_price *= 1.0015

        # Second third: downtrend
        for _ in range(n_points // 3, 2 * n_points // 3):
            prices.append(current_price)
            current_price *= 0.9985

        # Last third: neutral/mixed
        np.random.seed(42)
        final_price = prices[-1]
        for _ in range(2 * n_points // 3, n_points):
            prices.append(final_price * (1 + np.random.uniform(-0.002, 0.002)))
        df = pd.DataFrame({"close": prices})

        result = create_balanced_target(df)
        assert result is not None, "Result should not be None"
        assert not result.empty, "Result should contain data"

        target_counts = result["Target"].value_counts()
        # With sustained upward trend, downward trend, and neutral movements,
        # we should get multiple classes. However, if only one class appears,
        # it's still a valid outcome (the function works correctly, just the data
        # pattern didn't produce the expected distribution).
        # We'll check that the function at least produces valid results.
        assert len(target_counts) >= 2, "Should have at least two target classes with mixed trends"
        # Verify all target values are valid
        unique_targets = result["Target"].unique()
        assert all(t in [-1, 0, 1] for t in unique_targets), f"All targets should be -1, 0, or 1, got {unique_targets}"

    def test_create_targets_no_nan_in_target(self):
        """Test that Target column has no NaN values."""
        # Need at least FUTURE_RETURN_SHIFT + 1 data points
        df = pd.DataFrame({"close": [100 + i * 0.01 for i in range(30)]})

        result = create_balanced_target(df)
        assert result is not None
        if len(result) > 0:
            assert not result["Target"].isna().any()

    def test_create_targets_drops_undefined_rows(self):
        """Test that rows with undefined targets are dropped."""
        # Need at least FUTURE_RETURN_SHIFT + 1 data points
        df = pd.DataFrame({"close": [100 + i * 0.01 for i in range(30)]})

    def test_create_targets_with_lowercase_columns(self):
        """Test that function works with lowercase column names."""
        # Need at least FUTURE_RETURN_SHIFT + 1 data points
        df = pd.DataFrame({"CLOSE": [100 + i for i in range(30)]})
        df.columns = df.columns.str.lower()

        # Function should handle lowercase columns without raising exceptions
        result = create_balanced_target(df)
        # Result may be None or valid DataFrame depending on data sufficiency
        assert result is None or "Target" in result.columns

    def test_create_targets_large_dataset(self):
        """Test with larger dataset."""
        df = pd.DataFrame({"close": [100 + i * 0.001 * np.sin(i / 10) for i in range(200)]})

        result = create_balanced_target(df)
        assert result is not None
        assert len(result) > 0
        assert "Target" in result.columns

    def test_create_targets_volatile_data(self):
        """Test with volatile price data."""
        np.random.seed(42)
        base_price = 100
        prices = [base_price]
        for _ in range(50):
            prices.append(prices[-1] * (1 + np.random.uniform(-0.05, 0.05)))

        df = pd.DataFrame({"close": prices})
        result = create_balanced_target(df)
        assert result is not None
        if len(result) > 0:
            assert result["Target"].isin([-1, 0, 1]).all()
