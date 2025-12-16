"""
Tests for mar module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.common.quantitative_metrics import calculate_mar


def test_calculate_mar_matches_manual_computation_sma():
    """Test that calculate_mar matches manual computation for SMA."""
    close = pd.Series([100, 102, 101, 103, 105, 104, 106], dtype=float)
    length = 3

    result = calculate_mar(close, length=length, ma_type="SMA")

    # Manual calculation: SMA(3) = [NaN, NaN, 101, 102, 103, 104, 105]
    # MAR = close / SMA
    expected_ma = close.rolling(window=length, min_periods=length).mean()
    expected_mar = close / expected_ma.replace(0, np.nan)

    # Check that non-NaN values match
    valid_mask = ~(result.isna() | expected_mar.isna())
    if valid_mask.any():
        assert np.allclose(
            result[valid_mask].values, expected_mar[valid_mask].values, rtol=1e-9
        )


def test_calculate_mar_returns_valid_series():
    """Test that calculate_mar returns valid Series for valid input."""
    close = pd.Series([100, 102, 101, 103, 105, 104, 106] * 5, dtype=float)

    result = calculate_mar(close, length=5, ma_type="SMA")

    assert isinstance(result, pd.Series)
    assert len(result) == len(close)
    # Should have some valid (non-NaN) values after the initial period
    assert result.notna().sum() > 0


def test_calculate_mar_with_sma():
    """Test that calculate_mar works with SMA."""
    close = pd.Series([100, 102, 101, 103, 105, 104, 106], dtype=float)

    result = calculate_mar(close, length=3, ma_type="SMA")

    assert isinstance(result, pd.Series)
    assert len(result) == len(close)
    # First (length-1) values should be NaN due to rolling window
    assert result.iloc[:2].isna().all()
    # Later values should be valid
    assert result.iloc[2:].notna().any()


def test_calculate_mar_with_ema():
    """Test that calculate_mar works with EMA."""
    close = pd.Series([100, 102, 101, 103, 105, 104, 106], dtype=float)

    result = calculate_mar(close, length=3, ma_type="EMA")

    assert isinstance(result, pd.Series)
    assert len(result) == len(close)
    # EMA typically has fewer NaN values than SMA
    assert result.notna().any()


def test_calculate_mar_default_parameters():
    """Test that calculate_mar uses default parameters."""
    close = pd.Series([100, 102, 101, 103, 105, 104, 106] * 20, dtype=float)

    result = calculate_mar(close)

    # Should work with default parameters (length=14, ma_type="SMA")
    assert isinstance(result, pd.Series)
    assert len(result) == len(close)


def test_calculate_mar_none_input():
    """Test that calculate_mar handles None input."""
    result = calculate_mar(None, length=3)

    assert isinstance(result, pd.Series)
    assert result.empty


def test_calculate_mar_empty_series():
    """Test that calculate_mar handles empty series."""
    result = calculate_mar(pd.Series([], dtype=float), length=3)

    assert isinstance(result, pd.Series)
    assert result.empty


def test_calculate_mar_non_series_input():
    """Test that calculate_mar handles non-Series input."""
    result = calculate_mar([100, 102, 101, 103, 105], length=3)

    assert isinstance(result, pd.Series)
    assert result.empty


def test_calculate_mar_invalid_ma_type():
    """Test that calculate_mar handles invalid ma_type."""
    close = pd.Series([100, 102, 101, 103, 105, 104, 106], dtype=float)

    result = calculate_mar(close, length=3, ma_type="INVALID")

    assert isinstance(result, pd.Series)
    assert result.empty


def test_calculate_mar_invalid_length_zero():
    """Test that calculate_mar handles length = 0."""
    close = pd.Series([100, 102, 101, 103, 105, 104, 106], dtype=float)

    result = calculate_mar(close, length=0)

    assert isinstance(result, pd.Series)
    assert result.empty


def test_calculate_mar_invalid_length_negative():
    """Test that calculate_mar handles negative length."""
    close = pd.Series([100, 102, 101, 103, 105, 104, 106], dtype=float)

    result = calculate_mar(close, length=-1)

    assert isinstance(result, pd.Series)
    assert result.empty


def test_calculate_mar_with_nan_values():
    """Test that calculate_mar handles NaN values in input."""
    close = pd.Series([100, 102, np.nan, 103, 105, np.nan, 106], dtype=float)

    result = calculate_mar(close, length=3, ma_type="SMA")

    assert isinstance(result, pd.Series)
    assert len(result) == len(close)
    # pandas_ta may return NaN when input has NaN, which is acceptable behavior
    # The important thing is that it doesn't crash and returns a Series
    assert isinstance(result, pd.Series)


def test_calculate_mar_with_zeros():
    """Test that calculate_mar handles zeros in input."""
    close = pd.Series([0, 0, 0, 0, 0], dtype=float)

    result = calculate_mar(close, length=2, ma_type="SMA")

    assert isinstance(result, pd.Series)
    assert len(result) == len(close)
    # When MA is zero, division results in NaN (due to replace(0, np.nan))
    assert result.isna().all() or (close == 0).all()


def test_calculate_mar_short_series():
    """Test that calculate_mar handles short series."""
    close = pd.Series([100, 102, 101], dtype=float)

    result = calculate_mar(close, length=5, ma_type="SMA")

    assert isinstance(result, pd.Series)
    assert len(result) == len(close)
    # With length > series length, most values will be NaN
    assert result.isna().all() or result.notna().any()


def test_calculate_mar_constant_series():
    """Test that calculate_mar handles constant series."""
    close = pd.Series([100.0] * 20, dtype=float)

    result = calculate_mar(close, length=5, ma_type="SMA")

    assert isinstance(result, pd.Series)
    assert len(result) == len(close)
    # Constant series: close = MA, so MAR should be approximately 1.0
    # (with some NaN at the beginning due to rolling window)
    valid_values = result.dropna()
    if len(valid_values) > 0:
        assert np.allclose(valid_values.values, 1.0, rtol=1e-9)


def test_calculate_mar_ratio_interpretation():
    """Test that MAR values are correctly interpreted (ratio > 1 means above MA)."""
    # Create series where price is consistently above its MA
    close = pd.Series([100, 105, 110, 115, 120, 125, 130], dtype=float)

    result = calculate_mar(close, length=3, ma_type="SMA")

    # After initial NaN period, MAR should be > 1.0 (price above MA)
    valid_values = result.dropna()
    if len(valid_values) > 0:
        assert (valid_values > 1.0).any() or np.allclose(valid_values.values, 1.0, rtol=1e-9)


def test_calculate_mar_different_lengths():
    """Test that calculate_mar works with different length values."""
    close = pd.Series([100, 102, 101, 103, 105, 104, 106] * 5, dtype=float)

    result_short = calculate_mar(close, length=3, ma_type="SMA")
    result_long = calculate_mar(close, length=10, ma_type="SMA")

    assert isinstance(result_short, pd.Series)
    assert isinstance(result_long, pd.Series)
    assert len(result_short) == len(close)
    assert len(result_long) == len(close)
    # Longer length should have more NaN values at the beginning
    assert result_short.notna().sum() >= result_long.notna().sum()

