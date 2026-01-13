import numpy as np
import pandas as pd

from modules.common.quantitative_metrics import calculate_fisher_transform

"""
Tests for fisher_transform module.
"""


def test_calculate_fisher_transform_returns_valid_series():
    """Test that calculate_fisher_transform returns valid Series for valid input."""
    high = pd.Series([100, 102, 101, 105, 106, 104, 108], dtype=float)
    low = pd.Series([98, 99, 100, 101, 102, 103, 105], dtype=float)
    close = pd.Series([99, 101, 100.5, 103, 104, 103.5, 106.5], dtype=float)

    result = calculate_fisher_transform(high, low, close, length=3)

    assert isinstance(result, pd.Series)
    assert len(result) == len(close)
    # Should have valid (finite) values
    assert np.isfinite(result).any()


def test_calculate_fisher_transform_default_length():
    """Test that calculate_fisher_transform uses default length when not specified."""
    high = pd.Series([100, 102, 101, 105, 106, 104, 108] * 5, dtype=float)
    low = pd.Series([98, 99, 100, 101, 102, 103, 105] * 5, dtype=float)
    close = pd.Series([99, 101, 100.5, 103, 104, 103.5, 106.5] * 5, dtype=float)

    result = calculate_fisher_transform(high, low, close)

    # Should work with default length (9)
    assert isinstance(result, pd.Series)
    assert len(result) == len(close)


def test_calculate_fisher_transform_none_input():
    """Test that calculate_fisher_transform handles None input."""
    high = pd.Series([100, 102, 101], dtype=float)
    low = pd.Series([98, 99, 100], dtype=float)
    close = pd.Series([99, 101, 100.5], dtype=float)

    result = calculate_fisher_transform(None, low, close, length=3)
    assert isinstance(result, pd.Series)
    assert result.empty

    result = calculate_fisher_transform(high, None, close, length=3)
    assert isinstance(result, pd.Series)
    assert result.empty

    result = calculate_fisher_transform(high, low, None, length=3)
    assert isinstance(result, pd.Series)
    assert result.empty


def test_calculate_fisher_transform_empty_series():
    """Test that calculate_fisher_transform handles empty series."""
    high = pd.Series([], dtype=float)
    low = pd.Series([98, 99, 100], dtype=float)
    close = pd.Series([99, 101, 100.5], dtype=float)

    result = calculate_fisher_transform(high, low, close, length=3)
    assert isinstance(result, pd.Series)
    assert result.empty

    result = calculate_fisher_transform(low, high, close, length=3)
    assert isinstance(result, pd.Series)
    assert result.empty

    result = calculate_fisher_transform(low, close, high, length=3)
    assert isinstance(result, pd.Series)
    assert result.empty


def test_calculate_fisher_transform_non_series_input():
    """Test that calculate_fisher_transform handles non-Series input."""
    high = pd.Series([100, 102, 101], dtype=float)
    low = pd.Series([98, 99, 100], dtype=float)
    close = pd.Series([99, 101, 100.5], dtype=float)

    result = calculate_fisher_transform([100, 102, 101], low, close, length=3)
    assert isinstance(result, pd.Series)
    assert result.empty

    result = calculate_fisher_transform(high, [98, 99, 100], close, length=3)
    assert isinstance(result, pd.Series)
    assert result.empty

    result = calculate_fisher_transform(high, low, [99, 101, 100.5], length=3)
    assert isinstance(result, pd.Series)
    assert result.empty


def test_calculate_fisher_transform_invalid_length_zero():
    """Test that calculate_fisher_transform handles length = 0."""
    high = pd.Series([100, 102, 101, 105, 106], dtype=float)
    low = pd.Series([98, 99, 100, 101, 102], dtype=float)
    close = pd.Series([99, 101, 100.5, 103, 104], dtype=float)

    result = calculate_fisher_transform(high, low, close, length=0)
    assert isinstance(result, pd.Series)
    assert result.empty


def test_calculate_fisher_transform_invalid_length_negative():
    """Test that calculate_fisher_transform handles negative length."""
    high = pd.Series([100, 102, 101, 105, 106], dtype=float)
    low = pd.Series([98, 99, 100, 101, 102], dtype=float)
    close = pd.Series([99, 101, 100.5, 103, 104], dtype=float)

    result = calculate_fisher_transform(high, low, close, length=-1)
    assert isinstance(result, pd.Series)
    assert result.empty


def test_calculate_fisher_transform_with_nan_values():
    """Test that calculate_fisher_transform handles NaN values correctly."""
    high = pd.Series([100, 102, np.nan, 105, 106, 104, 108], dtype=float)
    low = pd.Series([98, 99, 100, 101, np.nan, 103, 105], dtype=float)
    close = pd.Series([99, 101, 100.5, 103, 104, np.nan, 106.5], dtype=float)

    result = calculate_fisher_transform(high, low, close, length=3)

    assert isinstance(result, pd.Series)
    assert len(result) == len(close)
    # Should still compute where possible (NaN handling in core function)


def test_calculate_fisher_transform_with_inf_values():
    """Test that calculate_fisher_transform handles Inf values."""
    high = pd.Series([100, 102, np.inf, 105, 106], dtype=float)
    low = pd.Series([98, 99, 100, 101, 102], dtype=float)
    close = pd.Series([99, 101, 100.5, 103, 104], dtype=float)

    result = calculate_fisher_transform(high, low, close, length=3)

    # Should still compute (Inf values are handled in core function)
    assert isinstance(result, pd.Series)
    assert len(result) == len(close)


def test_calculate_fisher_transform_constant_prices():
    """Test that calculate_fisher_transform handles constant prices (high == low)."""
    high = pd.Series([100.0] * 10, dtype=float)
    low = pd.Series([100.0] * 10, dtype=float)
    close = pd.Series([100.0] * 10, dtype=float)

    result = calculate_fisher_transform(high, low, close, length=3)

    assert isinstance(result, pd.Series)
    assert len(result) == len(close)
    # Constant prices should result in normalized = 0.0, so Fisher Transform should be stable
    assert np.isfinite(result).all()


def test_calculate_fisher_transform_different_lengths():
    """Test that calculate_fisher_transform works with different length values."""
    high = pd.Series([100, 102, 101, 105, 106, 104, 108] * 5, dtype=float)
    low = pd.Series([98, 99, 100, 101, 102, 103, 105] * 5, dtype=float)
    close = pd.Series([99, 101, 100.5, 103, 104, 103.5, 106.5] * 5, dtype=float)

    result_short = calculate_fisher_transform(high, low, close, length=3)
    result_long = calculate_fisher_transform(high, low, close, length=10)

    assert isinstance(result_short, pd.Series)
    assert isinstance(result_long, pd.Series)
    assert len(result_short) == len(close)
    assert len(result_long) == len(close)


def test_calculate_fisher_transform_different_series_lengths():
    """Test that calculate_fisher_transform handles different series lengths."""
    # Different lengths should work (pandas will align)
    high = pd.Series([100, 102, 101, 105, 106], dtype=float)
    low = pd.Series([98, 99, 100, 101, 102, 103], dtype=float)
    close = pd.Series([99, 101, 100.5, 103, 104, 103.5, 106.5], dtype=float)

    result = calculate_fisher_transform(high, low, close, length=3)

    # Should use length of close series
    assert isinstance(result, pd.Series)
    assert len(result) == len(close)


def test_calculate_fisher_transform_fisher_values_range():
    """Test that Fisher Transform values are in reasonable range."""
    high = pd.Series([100, 102, 101, 105, 106, 104, 108] * 10, dtype=float)
    low = pd.Series([98, 99, 100, 101, 102, 103, 105] * 10, dtype=float)
    close = pd.Series([99, 101, 100.5, 103, 104, 103.5, 106.5] * 10, dtype=float)

    result = calculate_fisher_transform(high, low, close, length=5)

    assert isinstance(result, pd.Series)
    # Fisher Transform values should be finite
    assert np.isfinite(result).any()
    # Fisher Transform typically ranges from -2 to 2, but can be larger
    # We just check that values are finite and reasonable


def test_calculate_fisher_transform_returns_same_length_as_close():
    """Test that result has same length as close series."""
    high = pd.Series([100, 102, 101, 105, 106], dtype=float)
    low = pd.Series([98, 99, 100, 101, 102], dtype=float)
    close = pd.Series([99, 101, 100.5, 103, 104, 103.5, 106.5], dtype=float)

    result = calculate_fisher_transform(high, low, close, length=3)

    # Result should have same length as close (index is from close)
    assert len(result) == len(close)
    assert result.index.equals(close.index)


def test_calculate_fisher_transform_with_short_series():
    """Test that calculate_fisher_transform works with short series."""
    high = pd.Series([100, 102, 101], dtype=float)
    low = pd.Series([98, 99, 100], dtype=float)
    close = pd.Series([99, 101, 100.5], dtype=float)

    result = calculate_fisher_transform(high, low, close, length=2)

    assert isinstance(result, pd.Series)
    assert len(result) == len(close)
    # Should still compute with short series
    assert np.isfinite(result).any()
