"""
Tests for Regime Detector.
"""

import pytest
import pandas as pd
from types import SimpleNamespace
from modules.position_sizing.core.regime_detector import RegimeDetector


def test_detect_regime_returns_valid_regime():
    """Test that regime detection returns valid regime string."""
    # Mock data fetcher
    def fake_fetch(symbol, **kwargs):
        # Create sample OHLCV data
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        df = pd.DataFrame({
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [101 + i * 0.1 for i in range(100)],
            'low': [99 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
        }, index=dates)
        return df, "binance"
    
    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )
    
    detector = RegimeDetector(data_fetcher)
    
    # This will fail if HMM module is not properly set up, but structure is correct
    # In real tests, you'd mock the HMM module
    regime = detector.detect_regime("BTC/USDT", "1h", 100)
    
    assert regime in ["BULLISH", "NEUTRAL", "BEARISH"]


def test_get_regime_probabilities():
    """Test regime probability calculation."""
    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        df = pd.DataFrame({
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [101 + i * 0.1 for i in range(100)],
            'low': [99 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
        }, index=dates)
        return df, "binance"
    
    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )
    
    detector = RegimeDetector(data_fetcher)
    
    probs = detector.get_regime_probabilities("BTC/USDT", "1h", 100)
    
    assert 'BULLISH' in probs
    assert 'NEUTRAL' in probs
    assert 'BEARISH' in probs
    
    # Probabilities should sum to approximately 1.0
    total = sum(probs.values())
    assert abs(total - 1.0) < 0.1


def test_detect_regime_batch():
    """Test batch regime detection."""
    def fake_fetch(symbol, **kwargs):
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        df = pd.DataFrame({
            'open': [100 + i * 0.1 for i in range(100)],
            'high': [101 + i * 0.1 for i in range(100)],
            'low': [99 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
        }, index=dates)
        return df, "binance"
    
    data_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )
    
    detector = RegimeDetector(data_fetcher)
    
    symbols = ["BTC/USDT", "ETH/USDT"]
    regimes = detector.detect_regime_batch(symbols, "1h", 100)
    
    assert len(regimes) == len(symbols)
    for symbol in symbols:
        assert symbol in regimes
        assert regimes[symbol] in ["BULLISH", "NEUTRAL", "BEARISH"]

