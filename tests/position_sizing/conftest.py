"""
Shared fixtures for position_sizing tests.
"""

import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_ohlcv_data():
    """Generate mock OHLCV data for testing."""
    def _generate_data(periods=200, base_price=100.0, volatility=0.5):
        dates = pd.date_range("2023-01-01", periods=periods, freq="h")
        # Create realistic price movement with random walk
        returns = np.random.randn(periods) * volatility
        prices = base_price + np.cumsum(returns)
        
        # Ensure prices are positive
        prices = np.maximum(prices, base_price * 0.1)
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(1000, 10000, periods),
        }, index=dates)
        
        return df
    
    return _generate_data


@pytest.fixture
def mock_data_fetcher(mock_ohlcv_data):
    """Create a fully mocked DataFetcher that doesn't call real APIs."""
    def fake_fetch(symbol, **kwargs):
        """Mock fetch function that returns generated data."""
        limit = kwargs.get('limit', 200)
        df = mock_ohlcv_data(periods=limit)
        return df, "binance"
    
    # Create a complete mock DataFetcher
    fetcher = SimpleNamespace()
    fetcher.fetch_ohlcv_with_fallback_exchange = fake_fetch
    fetcher.fetch_binance_account_balance = MagicMock(return_value=None)
    fetcher.market_prices = {}
    fetcher._ohlcv_dataframe_cache = {}
    
    return fetcher


@pytest.fixture(autouse=True)
def auto_mock_signal_calculators():
    """Automatically mock signal calculators for all tests to prevent API calls."""
    with patch('core.signal_calculators.get_range_oscillator_signal', return_value=(1, 0.7)), \
         patch('core.signal_calculators.get_spc_signal', return_value=(1, 0.6)), \
         patch('core.signal_calculators.get_xgboost_signal', return_value=(1, 0.8)), \
         patch('core.signal_calculators.get_hmm_signal', return_value=(1, 0.65)), \
         patch('core.signal_calculators.get_random_forest_signal', return_value=(1, 0.75)):
        yield


@pytest.fixture
def mock_regime_detector():
    """Mock RegimeDetector to return predictable regimes."""
    with patch('modules.position_sizing.core.regime_detector.RegimeDetector') as mock_class:
        mock_instance = MagicMock()
        mock_instance.detect_regime = MagicMock(return_value="NEUTRAL")
        mock_instance.get_regime_probabilities = MagicMock(return_value={
            "BULLISH": 0.33,
            "NEUTRAL": 0.34,
            "BEARISH": 0.33,
        })
        mock_class.return_value = mock_instance
        yield mock_instance

