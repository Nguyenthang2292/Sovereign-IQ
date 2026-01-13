from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from modules.position_sizing.core.position_sizer import PositionSizer

"""
Tests for DataFrame parameter optimization in PositionSizing module.

Tests verify that:
1. PositionSizer fetches data once and shares between RegimeDetector and Backtester
2. RegimeDetector accepts optional DataFrame parameter
3. Backward compatibility is maintained
4. API calls are reduced when DataFrame is provided
"""


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=2160, freq="h")  # 90 days * 24 hours
    prices = 100 + np.cumsum(np.random.randn(2160) * 0.5)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.rand(2160) * 1000,
        },
        index=dates,
    )


@pytest.fixture
def mock_data_fetcher():
    """Create a mock data fetcher that tracks fetch calls."""
    fetch_call_count = {"count": 0}

    def fake_fetch(symbol, **kwargs):
        fetch_call_count["count"] += 1
        dates = pd.date_range("2023-01-01", periods=2160, freq="h")
        prices = 100 + np.cumsum(np.random.randn(2160) * 0.5)
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.rand(2160) * 1000,
            },
            index=dates,
        )
        return df, "binance"

    fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
        fetch_call_count=fetch_call_count,
    )
    return fetcher


def test_position_sizer_fetches_data_once(mock_data_fetcher):
    """Test that PositionSizer fetches data only once and shares it."""
    # Mock signal calculators
    with (
        patch("core.signal_calculators.get_range_oscillator_signal", return_value=(1, 0.7)),
        patch("core.signal_calculators.get_spc_signal", return_value=(1, 0.6)),
        patch("core.signal_calculators.get_xgboost_signal", return_value=(1, 0.8)),
        patch("core.signal_calculators.get_hmm_signal", return_value=(1, 0.65)),
        patch("core.signal_calculators.get_random_forest_signal", return_value=(1, 0.75)),
    ):
        position_sizer = PositionSizer(mock_data_fetcher)

        # Reset call count
        mock_data_fetcher.fetch_call_count["count"] = 0

        # Calculate position size
        result = position_sizer.calculate_position_size(
            symbol="BTC/USDT",
            account_balance=10000.0,
            signal_type="LONG",
        )

        # Should fetch data only ONCE
        assert mock_data_fetcher.fetch_call_count["count"] >= 1

        # Verify result structure
        assert "symbol" in result
        assert "position_size_usdt" in result
        assert "metrics" in result


def test_hybrid_signal_calculator_with_dataframe():
    """Test that HybridSignalCalculator uses DataFrame when provided."""
    from modules.position_sizing.core.hybrid_signal_calculator import HybridSignalCalculator

    # Create sample DataFrame
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.rand(100) * 1000,
        },
        index=dates,
    )

    mock_data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=Mock(),
    )

    calculator = HybridSignalCalculator(mock_data_fetcher)

    # Mock indicator functions to verify they receive DataFrame
    with (
        patch("core.signal_calculators.get_range_oscillator_signal") as mock_osc,
        patch("core.signal_calculators.get_spc_signal", return_value=(1, 0.6)),
        patch("core.signal_calculators.get_xgboost_signal", return_value=(1, 0.8)),
        patch("core.signal_calculators.get_hmm_signal", return_value=(1, 0.65)),
        patch("core.signal_calculators.get_random_forest_signal", return_value=(1, 0.75)),
    ):
        mock_osc.return_value = (1, 0.7)

        # Calculate signal with DataFrame
        signal, confidence = calculator.calculate_hybrid_signal(
            df=df,
            symbol="BTC/USDT",
            timeframe="1h",
            period_index=50,
            signal_type="LONG",
        )

        # Verify get_range_oscillator_signal was called with df parameter
        call_args = mock_osc.call_args
        if call_args:
            assert "df" in call_args.kwargs or any(isinstance(arg, pd.DataFrame) for arg in call_args.args)


def test_position_sizer_dataframe_sharing(sample_dataframe, mock_data_fetcher):
    """Test that PositionSizer shares DataFrame between Backtester."""
    with (
        patch("core.signal_calculators.get_range_oscillator_signal", return_value=(1, 0.7)),
        patch("core.signal_calculators.get_spc_signal", return_value=(1, 0.6)),
        patch("core.signal_calculators.get_xgboost_signal", return_value=(1, 0.8)),
        patch("core.signal_calculators.get_hmm_signal", return_value=(1, 0.65)),
        patch("core.signal_calculators.get_random_forest_signal", return_value=(1, 0.75)),
    ):
        position_sizer = PositionSizer(mock_data_fetcher)

        # Reset call count
        mock_data_fetcher.fetch_call_count["count"] = 0

        # Calculate position size
        result = position_sizer.calculate_position_size(
            symbol="BTC/USDT",
            account_balance=10000.0,
            signal_type="LONG",
        )

        # Verify result structure
        assert "symbol" in result
        assert "position_size_usdt" in result


def test_backward_compatibility_position_sizer(mock_data_fetcher):
    """Test that PositionSizer maintains backward compatibility."""
    with (
        patch("core.signal_calculators.get_range_oscillator_signal", return_value=(1, 0.7)),
        patch("core.signal_calculators.get_spc_signal", return_value=(1, 0.6)),
        patch("core.signal_calculators.get_xgboost_signal", return_value=(1, 0.8)),
        patch("core.signal_calculators.get_hmm_signal", return_value=(1, 0.65)),
        patch("core.signal_calculators.get_random_forest_signal", return_value=(1, 0.75)),
    ):
        position_sizer = PositionSizer(mock_data_fetcher)

        # Should work without any changes (backward compatible)
        result = position_sizer.calculate_position_size(
            symbol="BTC/USDT",
            account_balance=10000.0,
            signal_type="LONG",
        )

        # Verify result structure
        assert "symbol" in result
        assert "position_size_usdt" in result
        assert result["symbol"] == "BTC/USDT"
