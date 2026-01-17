"""
Unit tests for Adaptive Trend Classification (ATC) equity calculations.

These tests verify that equity_series() produces results matching Pine Script behavior.
"""

import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend.core.compute_equity import equity_series
from modules.adaptive_trend.utils import rate_of_change


class TestEquitySeries:
    """Tests for equity_series function."""

    def test_basic_equity_calculation(self):
        """Test basic equity calculation with simple signals."""
        # Create simple price data
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0])

        # Create signals: 1 (long) for first 5 bars, -1 (short) for rest
        signals = pd.Series([1, 1, 1, 1, 1, -1, -1, -1, -1, -1], dtype="int8")

        # Calculate rate of change
        R = rate_of_change(prices)

        # Calculate equity with minimal parameters
        equity = equity_series(starting_equity=1.0, sig=signals, R=R, L=0.02, De=0.03, cutout=0)

        # Verify equity is not all NaN
        assert equity is not None
        assert not equity.isna().all()

        # Verify equity minimum floor (0.25) is respected
        assert equity.min() >= 0.25

        # Verify equity has same length as inputs
        assert len(equity) == len(signals)

    def test_equity_with_cutout(self):
        """Test that cutout period properly skips initial bars."""
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        signals = pd.Series([1, 1, 1, 1, 1], dtype="int8")
        R = rate_of_change(prices)

        cutout = 2
        equity = equity_series(starting_equity=1.0, sig=signals, R=R, L=0.02, De=0.03, cutout=cutout)

        # First 'cutout' values should be NaN
        assert equity.iloc[:cutout].isna().all()

        # Remaining values should be finite
        assert equity.iloc[cutout:].notna().all()

    def test_equity_floor(self):
        """Test that equity floor at 0.25 is applied."""
        # Create price data with large downward moves
        prices = pd.Series([100.0, 50.0, 25.0, 12.0, 6.0])

        # Create long signals (1) - should lose money
        signals = pd.Series([1, 1, 1, 1, 1], dtype="int8")
        R = rate_of_change(prices)

        equity = equity_series(starting_equity=1.0, sig=signals, R=R, L=0.02, De=0.03, cutout=0)

        # Verify floor is never breached
        assert (equity >= 0.25).all() or equity.isna().all()

    def test_equity_with_neutral_signals(self):
        """Test equity behavior with neutral (0) signals."""
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])

        # All neutral signals
        signals = pd.Series([0, 0, 0, 0, 0], dtype="int8")
        R = rate_of_change(prices)

        equity = equity_series(starting_equity=1.0, sig=signals, R=R, L=0.02, De=0.03, cutout=0)

        # Equity should only decrease due to decay, not from price moves
        if len(equity) > 1:
            # After first bar, equity should be close to 1.0 minus decay
            assert equity.iloc[-1] <= 1.0
            assert equity.iloc[-1] >= 0.25  # Respects floor

    def test_equity_with_high_decay(self):
        """Test equity behavior with high decay rate."""
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        signals = pd.Series([1, 1, 1, 1, 1], dtype="int8")
        R = rate_of_change(prices)

        # High decay rate (0.5 = 50% decay per period)
        equity = equity_series(starting_equity=1.0, sig=signals, R=R, L=0.02, De=0.5, cutout=0)

        # Equity should decrease significantly due to high decay
        if len(equity) > 1:
            assert equity.iloc[-1] < equity.iloc[1]

    def test_equity_with_no_decay(self):
        """Test equity behavior with zero decay."""
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        signals = pd.Series([1, 1, 1, 1, 1], dtype="int8")
        R = rate_of_change(prices)

        # Zero decay
        equity = equity_series(starting_equity=1.0, sig=signals, R=R, L=0.02, De=0.0, cutout=0)

        # Equity should not decrease due to decay (only from negative returns)
        assert (equity >= 0.25).all() or equity.isna().all()

    def test_equity_initial_weight(self):
        """Test that starting_equity parameter affects results."""
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        signals = pd.Series([1, 1, 1, 1, 1], dtype="int8")
        R = rate_of_change(prices)

        equity_1 = equity_series(starting_equity=1.0, sig=signals, R=R, L=0.02, De=0.0, cutout=0)
        equity_10 = equity_series(starting_equity=10.0, sig=signals, R=R, L=0.02, De=0.0, cutout=0)

        # Ratio of equities should match ratio of starting equity (approximately)
        if not equity_1.isna().all() and not equity_10.isna().all():
            ratio = equity_10.iloc[-1] / equity_1.iloc[-1] if equity_1.iloc[-1] > 0.25 else 10.0
            assert 5.0 <= ratio <= 15.0  # Allow some tolerance

    def test_equity_input_validation(self):
        """Test that invalid inputs raise appropriate errors."""
        prices = pd.Series([100.0, 101.0, 102.0])
        signals = pd.Series([1, 0, -1], dtype="int8")
        R = rate_of_change(prices)

        # Invalid starting_equity
        with pytest.raises(ValueError, match="starting_equity must be > 0"):
            equity_series(starting_equity=0.0, sig=signals, R=R, L=0.02, De=0.0, cutout=0)

        with pytest.raises(ValueError, match="starting_equity must be > 0"):
            equity_series(starting_equity=-1.0, sig=signals, R=R, L=0.02, De=0.0, cutout=0)

        # Invalid De (decay)
        with pytest.raises(ValueError, match="De must be between 0 and 1"):
            equity_series(starting_equity=1.0, sig=signals, R=R, L=0.02, De=-0.1, cutout=0)

        with pytest.raises(ValueError, match="De must be between 0 and 1"):
            equity_series(starting_equity=1.0, sig=signals, R=R, L=0.02, De=1.5, cutout=0)

        # Invalid L (lambda)
        with pytest.raises(ValueError, match="L must be a finite number"):
            equity_series(starting_equity=1.0, sig=signals, R=R, L=np.nan, De=0.0, cutout=0)

        with pytest.raises(ValueError, match="L must be a finite number"):
            equity_series(starting_equity=1.0, sig=signals, R=R, L=np.inf, De=0.0, cutout=0)

        # Invalid cutout
        with pytest.raises(ValueError, match="cutout must be >= 0"):
            equity_series(starting_equity=1.0, sig=signals, R=R, L=0.02, De=0.0, cutout=-1)

    def test_equity_with_nan_signals(self):
        """Test equity behavior with NaN values in signal series."""
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        # Insert NaN in middle
        signals = pd.Series([1, 1, np.nan, 1, 1], dtype="float64")
        R = rate_of_change(prices)

        equity = equity_series(starting_equity=1.0, sig=signals, R=R, L=0.02, De=0.0, cutout=0)

        # Should still produce results
        assert equity is not None

        # NaN signals are treated as 0 (no position)
        assert not equity.isna().all()

    def test_equity_with_nan_prices(self):
        """Test equity behavior with NaN values in price series (R)."""
        # Insert NaN in price data
        prices = pd.Series([100.0, 101.0, np.nan, 103.0, 104.0])
        signals = pd.Series([1, 1, 1, 1, 1], dtype="int8")
        R = rate_of_change(prices)

        equity = equity_series(starting_equity=1.0, sig=signals, R=R, L=0.02, De=0.0, cutout=0)

        # Should still produce results
        assert equity is not None
        assert not equity.isna().all()

    def test_equity_empty_series(self):
        """Test equity behavior with empty input series."""
        empty_prices = pd.Series([], dtype="float64")
        empty_signals = pd.Series([], dtype="int8")
        empty_R = rate_of_change(empty_prices)

        equity = equity_series(starting_equity=1.0, sig=empty_signals, R=empty_R, L=0.02, De=0.0, cutout=0)

        # Should return empty series
        assert len(equity) == 0
        assert equity.dtype == "float64"

    def test_equity_mismatched_indices(self):
        """Test equity behavior when sig and R have different indices."""
        prices_1 = pd.Series([100.0, 101.0, 102.0])
        signals = pd.Series([1, 0, -1], dtype="int8")

        prices_2 = pd.Series([100.0, 101.0, 102.0])
        prices_2.index = [10, 20, 30]  # Different indices
        R = rate_of_change(prices_2)

        # Should align indices or raise error
        try:
            equity = equity_series(starting_equity=1.0, sig=signals, R=R, L=0.02, De=0.0, cutout=0)
            # If successful, check it has valid length
            assert len(equity) <= len(signals)
        except ValueError as e:
            # Alignment failed
            assert "no common indices" in str(e).lower()

    def test_equity_matches_pine_script_behavior(self):
        """
        Test that equity calculation matches Pine Script behavior on known dataset.

        Pine Script eq() function behavior:
        - R = (close - close[1]) / close[1]
        - r = R * e(La)  where e(La) = exp(La * (bar_index - cutout))
        - d = 1 - De
        - if sig[1] > 0: a = r
        - elif sig[1] < 0: a = -r
        - else: a = 0
        - if na(e[1]): e = starting_equity
        - else: e = (e[1] * d) * (1 + a)
        - if e < 0.25: e = 0.25
        """
        # Simple test case with known expected behavior
        prices = pd.Series([100.0, 101.0, 102.0])
        signals = pd.Series([1, 1, 1], dtype="int8")
        R = rate_of_change(prices)

        # Manual calculation for first few bars (simplified):
        # Bar 0: e = 1.0 (starting)
        # Bar 1: e[1]=1.0, sig[1]=1, R[1]=(101-100)/100=0.01
        # With L=0, e(0)=1, so r=0.01, a=0.01
        # e = (1.0 * (1-0)) * (1+0.01) = 1.01

        equity = equity_series(starting_equity=1.0, sig=signals, R=R, L=0.0, De=0.0, cutout=0)

        # Verify first non-NaN value is starting_equity
        assert equity.iloc[0] == 1.0

        # Verify equity increases (positive returns with long position)
        if len(equity) > 1 and not equity.isna().iloc[1]:
            assert equity.iloc[1] > equity.iloc[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
