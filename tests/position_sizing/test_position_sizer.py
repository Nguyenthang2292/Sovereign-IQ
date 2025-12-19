"""
Tests for Position Sizer.
"""

import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace
from modules.position_sizing.core.position_sizer import PositionSizer


@pytest.fixture
def mock_data_fetcher():
    """Create a mock data fetcher for testing."""
    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=200, freq="h")
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
        }, index=dates)
        return df, "binance"
    
    return SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )


def test_calculate_position_size_structure(mock_data_fetcher):
    """Test that calculate_position_size returns valid structure."""
    position_sizer = PositionSizer(mock_data_fetcher)
    
    result = position_sizer.calculate_position_size(
        symbol="BTC/USDT",
        account_balance=10000.0,
        signal_type="LONG",
    )
    
    required_keys = [
        'symbol', 'signal_type', 'regime', 'position_size_usdt',
        'position_size_pct', 'kelly_fraction', 'adjusted_kelly_fraction',
        'regime_multiplier', 'metrics', 'backtest_result',
    ]
    
    for key in required_keys:
        assert key in result
    
    assert result['symbol'] == "BTC/USDT"
    assert result['signal_type'] == "LONG"
    assert result['regime'] in ["BULLISH", "NEUTRAL", "BEARISH"]
    assert result['position_size_usdt'] >= 0
    assert result['position_size_pct'] >= 0


def test_calculate_position_size_with_custom_params(mock_data_fetcher):
    """Test position size calculation with custom parameters."""
    position_sizer = PositionSizer(
        mock_data_fetcher,
        timeframe="4h",
        lookback_days=60,
        max_position_size=0.15,
    )
    
    result = position_sizer.calculate_position_size(
        symbol="ETH/USDT",
        account_balance=5000.0,
        signal_type="SHORT",
        timeframe="4h",
        lookback=60,
    )
    
    assert result['symbol'] == "ETH/USDT"
    assert result['signal_type'] == "SHORT"
    assert result['position_size_usdt'] <= 5000.0 * 0.15


def test_calculate_portfolio_allocation(mock_data_fetcher):
    """Test portfolio allocation calculation."""
    position_sizer = PositionSizer(mock_data_fetcher)
    
    symbols = [
        {'symbol': 'BTC/USDT', 'signal': 1},
        {'symbol': 'ETH/USDT', 'signal': 1},
    ]
    
    results_df = position_sizer.calculate_portfolio_allocation(
        symbols=symbols,
        account_balance=10000.0,
    )
    
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 2
    assert 'symbol' in results_df.columns
    assert 'position_size_usdt' in results_df.columns
    
    # Check that total exposure doesn't exceed max
    total_exposure = results_df['position_size_pct'].sum()
    assert total_exposure <= position_sizer.max_portfolio_exposure * 100


def test_calculate_portfolio_allocation_normalization(mock_data_fetcher):
    """Test that portfolio allocation normalizes when exceeding max exposure."""
    position_sizer = PositionSizer(
        mock_data_fetcher,
        max_portfolio_exposure=0.3,  # 30% max
    )
    
    # Create many symbols to force normalization
    symbols = [
        {'symbol': f'COIN{i}/USDT', 'signal': 1}
        for i in range(10)
    ]
    
    results_df = position_sizer.calculate_portfolio_allocation(
        symbols=symbols,
        account_balance=10000.0,
    )
    
    total_exposure = results_df['position_size_pct'].sum()
    # Should be normalized to max_portfolio_exposure
    assert total_exposure <= position_sizer.max_portfolio_exposure * 100 + 1.0  # Allow small rounding error


def test_empty_result_structure(mock_data_fetcher):
    """Test empty result structure."""
    position_sizer = PositionSizer(mock_data_fetcher)
    
    result = position_sizer._empty_result("TEST/USDT", "LONG")
    
    assert result['symbol'] == "TEST/USDT"
    assert result['signal_type'] == "LONG"
    assert result['position_size_usdt'] == 0.0
    assert result['position_size_pct'] == 0.0
    assert 'metrics' in result
    assert 'backtest_result' in result


def test_position_size_bounds(mock_data_fetcher):
    """Test that position size respects min/max bounds."""
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
    
    position_pct = result['position_size_pct']
    assert position_pct >= position_sizer.min_position_size * 100
    assert position_pct <= position_sizer.max_position_size * 100

