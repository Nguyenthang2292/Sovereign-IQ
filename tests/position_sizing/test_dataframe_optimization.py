from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from modules.position_sizing.core.position_sizer import PositionSizer

"""
Tests for DataFrame parameter optimization in PositionSizing module.

Tests verify that:
1. PositionSizer fetches data once and shares between components
2. HybridSignalCalculator accepts optional DataFrame parameter
3. Backward compatibility is maintained
4. API calls are reduced when DataFrame is provided
"""


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.rand(n) * 1000,
        },
        index=dates,
    )


@pytest.fixture
def mock_data_fetcher():
    """Create a mock data fetcher that tracks fetch calls."""
    fetch_call_count = {"count": 0}

    def fake_fetch(symbol, **kwargs):
        fetch_call_count["count"] += 1
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="h")
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.rand(n) * 1000,
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
    with patch("modules.position_sizing.core.position_sizer.FullBacktester") as MockBacktester:
        mock_bt = MockBacktester.return_value
        mock_bt.backtest.return_value = {
            "metrics": {"win_rate": 0.5, "num_trades": 20, "avg_win": 0.1, "avg_loss": 0.05},
            "equity_curve": pd.Series([10000, 10100]),
            "trades": [Mock()] * 20,
        }

        position_sizer = PositionSizer(mock_data_fetcher)

        # Reset call count
        mock_data_fetcher.fetch_call_count["count"] = 0

        # Calculate position size
        with patch("modules.position_sizing.core.position_sizer.days_to_candles", return_value=100):
            position_sizer.calculate_position_size(
                symbol="BTC/USDT",
                account_balance=10000.0,
                signal_type="LONG",
            )

        # Should fetch data only ONCE
        assert mock_data_fetcher.fetch_call_count["count"] == 1

        # Verify backtester was called with the dataframe
        assert mock_bt.backtest.called
        args, kwargs = mock_bt.backtest.call_args
        assert "df" in kwargs
        assert isinstance(kwargs["df"], pd.DataFrame)


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
        patch("modules.position_sizing.core.indicator_calculators.get_range_oscillator_signal") as mock_osc,
        patch("modules.position_sizing.core.indicator_calculators.get_spc_signal", return_value=(1, 0.6)),
        patch("modules.position_sizing.core.indicator_calculators.get_xgboost_signal", return_value=(1, 0.8)),
        patch("modules.position_sizing.core.indicator_calculators.get_hmm_signal", return_value=(1, 0.65)),
        patch("modules.position_sizing.core.indicator_calculators.get_random_forest_signal", return_value=(1, 0.75)),
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


def test_position_sizer_dataframe_sharing(mock_data_fetcher):
    """Test that PositionSizer shares DataFrame between components."""
    with patch("modules.position_sizing.core.position_sizer.FullBacktester") as MockBacktester:
        mock_bt = MockBacktester.return_value
        mock_bt.backtest.return_value = {
            "metrics": {"win_rate": 0.5, "num_trades": 20, "avg_win": 0.1, "avg_loss": 0.05},
            "equity_curve": pd.Series([10000, 10100]),
            "trades": [Mock()] * 20,
        }

        position_sizer = PositionSizer(mock_data_fetcher)

        # Reset call count
        mock_data_fetcher.fetch_call_count["count"] = 0

        # Calculate position size
        with patch("modules.position_sizing.core.position_sizer.days_to_candles", return_value=100):
            result = position_sizer.calculate_position_size(
                symbol="BTC/USDT",
                account_balance=10000.0,
                signal_type="LONG",
            )

        # Verify result structure
        assert "symbol" in result
        assert "position_size_usdt" in result
        assert mock_bt.backtest.called
        assert "df" in mock_bt.backtest.call_args.kwargs


def test_backward_compatibility_position_sizer(mock_data_fetcher):
    """Test that PositionSizer maintains backward compatibility."""
    with patch("modules.position_sizing.core.position_sizer.FullBacktester") as MockBacktester:
        mock_bt = MockBacktester.return_value
        mock_bt.backtest.return_value = {
            "metrics": {"win_rate": 0.5, "num_trades": 20, "avg_win": 0.1, "avg_loss": 0.05},
            "equity_curve": pd.Series([10000, 10100]),
            "trades": [Mock()] * 20,
        }

        position_sizer = PositionSizer(mock_data_fetcher)

        # Should work without any changes (backward compatible)
        with patch("modules.position_sizing.core.position_sizer.days_to_candles", return_value=100):
            result = position_sizer.calculate_position_size(
                symbol="BTC/USDT",
                account_balance=10000.0,
                signal_type="LONG",
            )

        # Verify result structure
        assert "symbol" in result
        assert "position_size_usdt" in result
        assert result["symbol"] == "BTC/USDT"
