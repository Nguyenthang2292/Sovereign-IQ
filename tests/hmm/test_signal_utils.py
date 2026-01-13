import sys
from pathlib import Path

"""
Test script for modules.hmm.signal_utils - Signal utilities.
"""


# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from modules.common.indicators import calculate_returns_volatility
from modules.hmm.signals.utils import validate_dataframe


def _sample_ohlcv_dataframe(length: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV DataFrame for testing."""
    idx = pd.date_range("2024-01-01", periods=length, freq="h")
    np.random.seed(42)
    base_price = 100.0
    prices = []
    for i in range(length):
        change = np.random.normal(0, 0.5)
        base_price += change
        prices.append(base_price)

    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
        },
        index=idx,
    )
    return df


def test_validate_dataframe_valid():
    """Test validate_dataframe with valid DataFrame."""
    df = _sample_ohlcv_dataframe(50)
    assert validate_dataframe(df) is True


def test_validate_dataframe_empty():
    """Test validate_dataframe with empty DataFrame."""
    df = pd.DataFrame()
    assert validate_dataframe(df) is False


def test_validate_dataframe_none():
    """Test validate_dataframe with None."""
    assert validate_dataframe(None) is False


def test_validate_dataframe_missing_columns():
    """Test validate_dataframe with missing required columns."""
    df = pd.DataFrame({"close": [100, 101, 102]})
    assert validate_dataframe(df) is False


def test_validate_dataframe_insufficient_rows():
    """Test validate_dataframe with insufficient rows."""
    df = _sample_ohlcv_dataframe(10)
    assert validate_dataframe(df) is False


def test_validate_dataframe_minimum_rows():
    """Test validate_dataframe with exactly minimum required rows."""
    df = _sample_ohlcv_dataframe(20)
    assert validate_dataframe(df) is True


def test_calculate_returns_volatility_valid():
    """Test calculate_returns_volatility with valid data."""
    df = _sample_ohlcv_dataframe(100)
    volatility = calculate_returns_volatility(df)
    assert isinstance(volatility, float)
    assert volatility >= 0.0


def test_calculate_returns_volatility_no_close():
    """Test calculate_returns_volatility without close column."""
    df = pd.DataFrame({"open": [100, 101, 102]})
    volatility = calculate_returns_volatility(df)
    assert volatility == 0.0


def test_calculate_returns_volatility_insufficient_data():
    """Test calculate_returns_volatility with insufficient data."""
    df = pd.DataFrame({"close": [100]})
    volatility = calculate_returns_volatility(df)
    assert volatility == 0.0


def test_calculate_returns_volatility_empty_dataframe():
    """Test calculate_returns_volatility with empty DataFrame."""
    df = pd.DataFrame({"close": []})
    volatility = calculate_returns_volatility(df)
    assert volatility == 0.0


def test_calculate_returns_volatility_constant_price():
    """Test calculate_returns_volatility with constant price (zero volatility)."""
    df = pd.DataFrame({"close": [100.0] * 50})
    volatility = calculate_returns_volatility(df)
    assert volatility == 0.0


def test_calculate_returns_volatility_high_volatility():
    """Test calculate_returns_volatility with high volatility data."""
    np.random.seed(42)
    prices = 100.0 + np.random.normal(0, 5.0, 100)  # High volatility
    df = pd.DataFrame({"close": prices})
    volatility = calculate_returns_volatility(df)
    assert volatility > 0.0
    assert volatility < 10.0  # Reasonable upper bound
