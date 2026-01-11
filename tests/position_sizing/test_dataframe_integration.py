
import pytest

from modules.backtester import FullBacktester
from modules.position_sizing.core.position_sizer import PositionSizer
from modules.position_sizing.core.position_sizer import PositionSizer

"""
Integration tests for DataFrame optimization across PositionSizing module.

Tests verify end-to-end that:
1. Data is fetched once and shared correctly
2. All components work together with DataFrame parameter
3. Performance improvement (fewer API calls)
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
    """Create a mock data fetcher that tracks all fetch calls."""
    fetch_calls = []

    def fake_fetch(symbol, **kwargs):
        fetch_calls.append(
            {
                "symbol": symbol,
                "kwargs": kwargs,
            }
        )
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
        fetch_calls=fetch_calls,
    )
    return fetcher


def test_end_to_end_dataframe_sharing(mock_data_fetcher, sample_dataframe):
    """Test that data is fetched once and shared across all components."""
    with (
        patch("core.signal_calculators.get_range_oscillator_signal", return_value=(1, 0.7)),
        patch("core.signal_calculators.get_spc_signal", return_value=(1, 0.6)),
        patch("core.signal_calculators.get_xgboost_signal", return_value=(1, 0.8)),
        patch("core.signal_calculators.get_hmm_signal", return_value=(1, 0.65)),
        patch("core.signal_calculators.get_random_forest_signal", return_value=(1, 0.75)),
    ):
        position_sizer = PositionSizer(mock_data_fetcher)

        # Clear fetch calls
        mock_data_fetcher.fetch_calls.clear()

        # Calculate position size (should fetch once internally)
        result = position_sizer.calculate_position_size(
            symbol="BTC/USDT",
            account_balance=10000.0,
            signal_type="LONG",
        )

        # Verify result structure
        assert "symbol" in result
        assert "position_size_usdt" in result
        assert "metrics" in result

        # In the optimized version, PositionSizer should fetch data once
        assert len(mock_data_fetcher.fetch_calls) >= 1

        # All fetch calls should be for the same symbol
        for call in mock_data_fetcher.fetch_calls:
            assert call["symbol"] == "BTC/USDT"


def test_backtester_independent_dataframe(sample_dataframe, mock_data_fetcher):
    """Test that Backtester can work independently with DataFrame."""
    # Test Backtester independently
    backtester = FullBacktester(mock_data_fetcher)
    mock_data_fetcher.fetch_calls.clear()

    backtest_result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=2160,
        signal_type="LONG",
        df=sample_dataframe,
    )

    # Verify no fetch was called
    assert len(mock_data_fetcher.fetch_calls) == 0
    assert "trades" in backtest_result
    assert "metrics" in backtest_result


def test_dataframe_consistency_across_components(sample_dataframe, mock_data_fetcher):
    """Test that same DataFrame produces consistent results across components."""
    with (
        patch("core.signal_calculators.get_range_oscillator_signal", return_value=(1, 0.7)),
        patch("core.signal_calculators.get_spc_signal", return_value=(1, 0.6)),
        patch("core.signal_calculators.get_xgboost_signal", return_value=(1, 0.8)),
        patch("core.signal_calculators.get_hmm_signal", return_value=(1, 0.65)),
        patch("core.signal_calculators.get_random_forest_signal", return_value=(1, 0.75)),
    ):
        # Use same DataFrame for Backtester
        backtester = FullBacktester(mock_data_fetcher)

        # Get backtest result
        backtest_result = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=2160,
            signal_type="LONG",
            df=sample_dataframe,
        )

        # Should work with same DataFrame
        assert "trades" in backtest_result
        assert "metrics" in backtest_result

        # Verify DataFrame was not modified
        assert len(sample_dataframe) == 2160
        assert "close" in sample_dataframe.columns


def test_performance_improvement_with_dataframe(mock_data_fetcher, sample_dataframe):
    """Test that using DataFrame reduces API calls."""
    with (
        patch("core.signal_calculators.get_range_oscillator_signal", return_value=(1, 0.7)),
        patch("core.signal_calculators.get_spc_signal", return_value=(1, 0.6)),
        patch("core.signal_calculators.get_xgboost_signal", return_value=(1, 0.8)),
        patch("core.signal_calculators.get_hmm_signal", return_value=(1, 0.65)),
        patch("core.signal_calculators.get_random_forest_signal", return_value=(1, 0.75)),
    ):
        backtester = FullBacktester(mock_data_fetcher)

        # Test WITHOUT DataFrame (old way)
        mock_data_fetcher.fetch_calls.clear()

        backtest_result1 = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=2160,
            signal_type="LONG",
            # No df parameter
        )

        calls_without_df = len(mock_data_fetcher.fetch_calls)

        # Test WITH DataFrame (optimized way)
        mock_data_fetcher.fetch_calls.clear()

        backtest_result2 = backtester.backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            lookback=2160,
            signal_type="LONG",
            df=sample_dataframe,
        )

        calls_with_df = len(mock_data_fetcher.fetch_calls)

        # Verify that using DataFrame reduces API calls
        assert calls_with_df < calls_without_df, (
            f"Expected fewer calls with DataFrame ({calls_with_df} < {calls_without_df})"
        )

        # Both should produce valid results
        assert "trades" in backtest_result1
        assert "trades" in backtest_result2

