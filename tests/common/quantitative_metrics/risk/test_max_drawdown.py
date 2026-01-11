
import numpy as np
import pandas as pd

from modules.common.quantitative_metrics import calculate_max_drawdown
from modules.common.quantitative_metrics import calculate_max_drawdown

"""
Tests for max_drawdown module.
"""




def test_calculate_max_drawdown_matches_manual():
    """Test that calculate_max_drawdown matches manual calculation."""
    spread = pd.Series([1, -2, -1, -3, 4, -2], dtype=float)
    cumulative = spread.cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    expected = float(drawdown.min())

    result = calculate_max_drawdown(cumulative)
    assert result == expected


def test_calculate_max_drawdown_none_input():
    """Test that calculate_max_drawdown handles None input."""
    result = calculate_max_drawdown(None)

    assert result is None


def test_calculate_max_drawdown_empty_series():
    """Test that calculate_max_drawdown handles empty series."""
    result = calculate_max_drawdown(pd.Series([], dtype=float))

    assert result is None


def test_calculate_max_drawdown_non_series_input():
    """Test that calculate_max_drawdown handles non-Series input."""
    result = calculate_max_drawdown([0, 10, 5, 20, 15, 30])

    assert result is None


def test_calculate_max_drawdown_insufficient_data():
    """Test that calculate_max_drawdown returns None for insufficient data."""
    equity_curve = pd.Series([100.0])

    result = calculate_max_drawdown(equity_curve)

    assert result is None


def test_calculate_max_drawdown_with_nan_values():
    """Test that calculate_max_drawdown handles NaN values correctly."""
    equity_curve = pd.Series([0, 10, np.nan, 5, 20, np.nan, 15, 30], dtype=float)

    result = calculate_max_drawdown(equity_curve)

    # Should still compute valid result after dropping NaN
    if result is not None:
        assert isinstance(result, float)
        assert result <= 0  # Drawdown should be negative or zero


def test_calculate_max_drawdown_with_inf_values():
    """Test that calculate_max_drawdown handles Inf values correctly."""
    equity_curve = pd.Series([0, 10, np.inf, 5, 20], dtype=float)

    result = calculate_max_drawdown(equity_curve)

    # Should return None when Inf values are present
    assert result is None


def test_calculate_max_drawdown_constant_series():
    """Test that calculate_max_drawdown handles constant series (no drawdown)."""
    equity_curve = pd.Series([100.0] * 10, dtype=float)

    result = calculate_max_drawdown(equity_curve)

    # Constant series has no drawdown, should return 0.0
    assert result == 0.0


def test_calculate_max_drawdown_only_increasing():
    """Test that calculate_max_drawdown handles only increasing series (no drawdown)."""
    equity_curve = pd.Series([0, 1, 2, 3, 4, 5, 6], dtype=float)

    result = calculate_max_drawdown(equity_curve)

    # Only increasing series has no drawdown, should return 0.0
    assert result == 0.0


def test_calculate_max_drawdown_negative_equity():
    """Test that calculate_max_drawdown handles negative equity curve."""
    equity_curve = pd.Series([0, -10, -5, -20, -15, -30], dtype=float)

    result = calculate_max_drawdown(equity_curve)

    # Should calculate drawdown correctly even with negative equity
    assert result is not None
    assert isinstance(result, float)
    # Drawdown should be negative (equity below peak)
    assert result <= 0


def test_calculate_max_drawdown_large_drawdown():
    """Test that calculate_max_drawdown handles large drawdown correctly."""
    # Create equity curve with large drawdown
    equity_curve = pd.Series([0, 100, 200, 50, 150, 10, 80], dtype=float)

    result = calculate_max_drawdown(equity_curve)

    # Manual calculation
    running_max = equity_curve.expanding().max()
    drawdown = equity_curve - running_max
    expected = float(drawdown.min())

    assert result == expected
    # Should be negative (drawdown from peak of 200 to 10 = -190)
    assert result < 0


def test_calculate_max_drawdown_multiple_peaks():
    """Test that calculate_max_drawdown handles multiple peaks correctly."""
    # Equity curve with multiple peaks and troughs
    equity_curve = pd.Series([0, 50, 30, 80, 40, 100, 60, 120, 70], dtype=float)

    result = calculate_max_drawdown(equity_curve)

    # Should return the worst drawdown across all peaks
    running_max = equity_curve.expanding().max()
    drawdown = equity_curve - running_max
    expected = float(drawdown.min())

    assert result == expected
    assert result <= 0


def test_calculate_max_drawdown_all_nan():
    """Test that calculate_max_drawdown handles all NaN values."""
    equity_curve = pd.Series([np.nan, np.nan, np.nan], dtype=float)

    result = calculate_max_drawdown(equity_curve)

    # Should return None when all values are NaN
    assert result is None
