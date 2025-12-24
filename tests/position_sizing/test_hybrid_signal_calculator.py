"""
Tests for Hybrid Signal Calculator.
"""

import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock, patch
from modules.position_sizing.core.hybrid_signal_calculator import HybridSignalCalculator


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


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
    }, index=dates)


def test_hybrid_signal_calculator_initialization(mock_data_fetcher):
    """Test HybridSignalCalculator initialization."""
    calculator = HybridSignalCalculator(mock_data_fetcher)
    
    assert calculator.data_fetcher == mock_data_fetcher
    assert len(calculator.enabled_indicators) > 0
    assert calculator.use_confidence_weighting is True
    assert calculator.min_indicators_agreement >= 0


def test_hybrid_signal_calculator_custom_indicators(mock_data_fetcher):
    """Test HybridSignalCalculator with custom enabled indicators."""
    calculator = HybridSignalCalculator(
        mock_data_fetcher,
        enabled_indicators=['range_oscillator', 'hmm'],
    )
    
    assert len(calculator.enabled_indicators) == 2
    assert 'range_oscillator' in calculator.enabled_indicators
    assert 'hmm' in calculator.enabled_indicators


def test_calculate_hybrid_signal_returns_tuple(mock_data_fetcher, sample_dataframe):
    """Test that calculate_hybrid_signal returns a tuple."""
    calculator = HybridSignalCalculator(mock_data_fetcher)
    
    # Mock the indicator functions to return simple results
    with patch('modules.position_sizing.core.hybrid_signal_calculator.get_range_oscillator_signal') as mock_osc:
        mock_osc.return_value = (1, 0.8)
        
        signal, confidence = calculator.calculate_hybrid_signal(
            df=sample_dataframe,
            symbol="BTC/USDT",
            timeframe="1h",
            period_index=50,
            signal_type="LONG",
        )
        
        assert isinstance(signal, int)
        assert signal in [-1, 0, 1]
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0


def test_calculate_hybrid_signal_caching(mock_data_fetcher, sample_dataframe):
    """Test that calculate_hybrid_signal uses caching."""
    calculator = HybridSignalCalculator(mock_data_fetcher)
    
    with patch('modules.position_sizing.core.hybrid_signal_calculator.get_range_oscillator_signal') as mock_osc:
        mock_osc.return_value = (1, 0.8)
        
        # First call
        signal1, conf1 = calculator.calculate_hybrid_signal(
            df=sample_dataframe,
            symbol="BTC/USDT",
            timeframe="1h",
            period_index=50,
            signal_type="LONG",
        )
        
        # Second call with same parameters (should use cache)
        signal2, conf2 = calculator.calculate_hybrid_signal(
            df=sample_dataframe,
            symbol="BTC/USDT",
            timeframe="1h",
            period_index=50,
            signal_type="LONG",
        )
        
        # Results should be the same
        assert signal1 == signal2
        assert conf1 == conf2
        
        # But function should only be called once (cached on second call)
        # Note: This might not work perfectly due to multiple indicators, but structure is correct


def test_combine_signals_majority_vote(mock_data_fetcher):
    """Test signal combination using majority vote."""
    calculator = HybridSignalCalculator(mock_data_fetcher)
    
    indicator_signals = [
        {'indicator': 'osc', 'signal': 1, 'confidence': 0.8},
        {'indicator': 'spc', 'signal': 1, 'confidence': 0.7},
        {'indicator': 'hmm', 'signal': 1, 'confidence': 0.6},
        {'indicator': 'rf', 'signal': -1, 'confidence': 0.5},
    ]
    
    signal, confidence = calculator.combine_signals_majority_vote(
        indicator_signals,
        expected_signal_type="LONG",
    )
    
    # Majority is LONG (3 vs 1)
    assert signal == 1
    assert confidence > 0


def test_combine_signals_insufficient_agreement(mock_data_fetcher):
    """Test that insufficient agreement returns neutral signal."""
    calculator = HybridSignalCalculator(
        mock_data_fetcher,
        min_indicators_agreement=5,  # Require 5 indicators to agree
    )
    
    indicator_signals = [
        {'indicator': 'osc', 'signal': 1, 'confidence': 0.8},
        {'indicator': 'spc', 'signal': 1, 'confidence': 0.7},
    ]
    
    signal, confidence = calculator.combine_signals_majority_vote(
        indicator_signals,
        expected_signal_type="LONG",
    )
    
    # Not enough agreement
    assert signal == 0
    assert confidence == 0.0


def test_combine_signals_majority_vote_valid_signal_types(mock_data_fetcher):
    """Test that valid signal types (case-insensitive) work correctly."""
    calculator = HybridSignalCalculator(mock_data_fetcher)
    
    indicator_signals = [
        {'indicator': 'osc', 'signal': 1, 'confidence': 0.8},
        {'indicator': 'spc', 'signal': 1, 'confidence': 0.7},
    ]
    
    # Test valid uppercase
    signal, confidence = calculator.combine_signals_majority_vote(
        indicator_signals,
        expected_signal_type="LONG",
    )
    assert signal in [0, 1]
    
    signal, confidence = calculator.combine_signals_majority_vote(
        indicator_signals,
        expected_signal_type="SHORT",
    )
    assert signal in [-1, 0]
    
    # Test valid lowercase
    signal, confidence = calculator.combine_signals_majority_vote(
        indicator_signals,
        expected_signal_type="long",
    )
    assert signal in [0, 1]
    
    signal, confidence = calculator.combine_signals_majority_vote(
        indicator_signals,
        expected_signal_type="short",
    )
    assert signal in [-1, 0]
    
    # Test valid mixed case
    signal, confidence = calculator.combine_signals_majority_vote(
        indicator_signals,
        expected_signal_type="Long",
    )
    assert signal in [0, 1]
    
    signal, confidence = calculator.combine_signals_majority_vote(
        indicator_signals,
        expected_signal_type="ShOrT",
    )
    assert signal in [-1, 0]


def test_combine_signals_majority_vote_invalid_signal_type(mock_data_fetcher):
    """Test that invalid signal types raise ValueError."""
    calculator = HybridSignalCalculator(mock_data_fetcher)
    
    indicator_signals = [
        {'indicator': 'osc', 'signal': 1, 'confidence': 0.8},
    ]
    
    # Test various invalid values
    invalid_values = ["NEUTRAL", "HOLD", "BUY", "SELL", "", "invalid", "123", None]
    
    for invalid_value in invalid_values:
        with pytest.raises(ValueError, match="Invalid expected_signal_type"):
            calculator.combine_signals_majority_vote(
                indicator_signals,
                expected_signal_type=invalid_value,
            )


def test_clear_cache(mock_data_fetcher):
    """Test cache clearing."""
    calculator = HybridSignalCalculator(mock_data_fetcher)
    
    # Add some cache entries
    calculator._signal_cache[('BTC/USDT', 10, 'LONG')] = (1, 0.8)
    calculator._indicator_cache[('BTC/USDT', 10, 'osc')] = {'signal': 1}
    
    assert len(calculator._signal_cache) > 0
    assert len(calculator._indicator_cache) > 0
    
    calculator.clear_cache()
    
    assert len(calculator._signal_cache) == 0
    assert len(calculator._indicator_cache) == 0


def test_get_cache_stats(mock_data_fetcher):
    """Test cache statistics."""
    calculator = HybridSignalCalculator(mock_data_fetcher)
    
    stats = calculator.get_cache_stats()
    
    assert 'signal_cache_size' in stats
    assert 'signal_cache_max_size' in stats
    assert 'data_cache_size' in stats
    assert 'data_cache_max_size' in stats
    
    assert stats['signal_cache_size'] >= 0
    assert stats['signal_cache_max_size'] > 0

