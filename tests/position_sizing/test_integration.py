
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from modules.position_sizing.core.kelly_calculator import BayesianKellyCalculator
from modules.position_sizing.core.position_sizer import PositionSizer
from modules.position_sizing.core.kelly_calculator import BayesianKellyCalculator
from modules.position_sizing.core.position_sizer import PositionSizer

"""
Integration tests for Position Sizing module.
"""





@pytest.fixture
def mock_data_fetcher():
    """Create a mock data fetcher for testing."""

    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=200, freq="h")
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
            },
            index=dates,
        )
        return df, "binance"

    return SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )


def test_full_position_sizing_workflow(mock_data_fetcher):
    """Test the complete position sizing workflow."""
    position_sizer = PositionSizer(mock_data_fetcher)

    # Test single symbol
    result = position_sizer.calculate_position_size(
        symbol="BTC/USDT",
        account_balance=10000.0,
        signal_type="LONG",
    )

    # Verify result structure
    assert "symbol" in result
    assert "position_size_usdt" in result
    assert "kelly_fraction" in result
    assert "regime" in result
    assert "metrics" in result

    # Verify values are reasonable
    assert result["position_size_usdt"] >= 0
    assert result["position_size_usdt"] <= 10000.0
    assert result["position_size_pct"] >= 0
    assert result["position_size_pct"] <= 100.0


def test_portfolio_allocation_workflow(mock_data_fetcher):
    """Test portfolio allocation workflow."""
    position_sizer = PositionSizer(mock_data_fetcher)

    symbols = [
        {"symbol": "BTC/USDT", "signal": 1},
        {"symbol": "ETH/USDT", "signal": 1},
        {"symbol": "BNB/USDT", "signal": -1},
    ]

    results_df = position_sizer.calculate_portfolio_allocation(
        symbols=symbols,
        account_balance=10000.0,
    )

    # Verify DataFrame structure
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 3

    # Verify columns
    required_cols = ["symbol", "position_size_usdt", "position_size_pct"]
    for col in required_cols:
        assert col in results_df.columns

    # Verify total exposure
    total_exposure = results_df["position_size_pct"].sum()
    assert total_exposure <= position_sizer.max_portfolio_exposure * 100


def test_regime_adjustment_affects_position_size(mock_data_fetcher):
    """Test that regime adjustment affects position size."""
    position_sizer = PositionSizer(mock_data_fetcher)

    # Calculate position size (regime will be detected)
    result = position_sizer.calculate_position_size(
        symbol="BTC/USDT",
        account_balance=10000.0,
        signal_type="LONG",
    )

    # Verify regime multiplier is applied
    assert "regime_multiplier" in result
    assert result["regime_multiplier"] in [0.8, 1.0, 1.2]  # BEARISH, NEUTRAL, BULLISH

    # Adjusted Kelly should be different from base Kelly if regime is not NEUTRAL
    if result["regime"] != "NEUTRAL":
        assert result["adjusted_kelly_fraction"] != result["kelly_fraction"]


def test_kelly_calculator_integration(mock_data_fetcher):
    """Test Kelly calculator integration with position sizer."""
    # Create metrics that would produce a valid Kelly fraction
    metrics = {
        "win_rate": 0.6,
        "avg_win": 0.05,
        "avg_loss": 0.02,
        "num_trades": 100,
    }

    calculator = BayesianKellyCalculator()
    kelly = calculator.calculate_kelly_from_metrics(metrics)

    # Kelly should be positive for good metrics
    assert kelly >= 0
    assert kelly <= 0.25  # Max Kelly fraction


def test_error_handling_in_position_sizing(mock_data_fetcher):
    """Test error handling in position sizing."""
    position_sizer = PositionSizer(mock_data_fetcher)

    # Test with invalid symbol (should return empty result)
    def failing_fetch(symbol, **kwargs):
        return None, None

    failing_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=failing_fetch,
    )

    failing_sizer = PositionSizer(failing_fetcher)

    result = failing_sizer.calculate_position_size(
        symbol="INVALID/USDT",
        account_balance=10000.0,
        signal_type="LONG",
    )

    # Should return empty result structure, not crash
    assert "symbol" in result
    assert result["position_size_usdt"] == 0.0
    assert result["position_size_pct"] == 0.0


def test_position_size_bounds_enforcement(mock_data_fetcher):
    """Test that position size bounds are enforced."""
    position_sizer = PositionSizer(
        mock_data_fetcher,
        max_position_size=0.1,  # 10% max
        min_position_size=0.01,  # 1% min
    )

    result = position_sizer.calculate_position_size(
        symbol="BTC/USDT",
        account_balance=10000.0,
        signal_type="LONG",
    )

    # Verify bounds
    position_pct = result["position_size_pct"]
    assert position_pct >= 1.0  # Min 1%
    assert position_pct <= 10.0  # Max 10%


def test_portfolio_normalization(mock_data_fetcher):
    """Test that portfolio allocation normalizes when exceeding max exposure."""
    position_sizer = PositionSizer(
        mock_data_fetcher,
        max_portfolio_exposure=0.2,  # 20% max exposure
    )

    # Create many symbols to force normalization
    symbols = [{"symbol": f"COIN{i}/USDT", "signal": 1} for i in range(20)]

    results_df = position_sizer.calculate_portfolio_allocation(
        symbols=symbols,
        account_balance=10000.0,
    )

    # Total exposure should be normalized
    total_exposure = results_df["position_size_pct"].sum()
    assert total_exposure <= 20.0 + 1.0  # Allow small rounding error


def test_different_signal_types(mock_data_fetcher):
    """Test position sizing with different signal types."""
    position_sizer = PositionSizer(mock_data_fetcher)

    # Test LONG
    result_long = position_sizer.calculate_position_size(
        symbol="BTC/USDT",
        account_balance=10000.0,
        signal_type="LONG",
    )

    # Test SHORT
    result_short = position_sizer.calculate_position_size(
        symbol="BTC/USDT",
        account_balance=10000.0,
        signal_type="SHORT",
    )

    # Both should return valid results
    assert result_long["signal_type"] == "LONG"
    assert result_short["signal_type"] == "SHORT"
    assert "position_size_usdt" in result_long
    assert "position_size_usdt" in result_short
