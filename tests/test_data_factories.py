"""Test Data Factories for Fast Test Data Generation

This module provides optimized factories for creating test data quickly.
Use these factories to avoid slow data generation in tests.

Usage:

    factory = TestDataFactory()
    ohlc_data = factory.create_ohlc_data(size=50)
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class TestDataFactory:
    """
    Factory for creating test data efficiently.

    All data is cached after first creation for reuse.
    """

    def __init__(self, seed: int = 42):
        """Initialize factory with random seed."""
        self._seed = seed
        self._cache = {}

    def create_ohlc_data(
        self,
        size: int = 100,
        base_price: float = 50000.0,
        volatility: float = 0.02,
        trend: float = 0.0,
        cache_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create OHLCV data with caching for speed.

        Args:
            size: Number of data points
            base_price: Starting price
            volatility: Price volatility (0.02 = 2%)
            trend: Trend factor per period
            cache_key: Optional cache key for reuse

        Returns:
            DataFrame with open, high, low, close, volume columns
        """
        # Check cache
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        np.random.seed(self._seed)
        dates = pd.date_range("2024-01-01", periods=size, freq="1h")

        # Generate price series
        returns = np.random.randn(size) * volatility + trend
        prices = base_price * (1 + np.cumsum(returns))

        # Create OHLCV
        noise = np.random.randn(size) * base_price * 0.01
        close = prices + noise
        high = close + np.abs(np.random.randn(size) * base_price * 0.005)
        low = close - np.abs(np.random.randn(size) * base_price * 0.005)
        open_price = close + np.random.randn(size) * base_price * 0.002
        volume = np.random.uniform(1000, 5000, size)

        df = pd.DataFrame({"open": open_price, "high": high, "low": low, "close": close, "volume": volume}, index=dates)

        # Cache if key provided
        if cache_key:
            self._cache[cache_key] = df

        return df

    def create_oscillator_data(
        self, size: int = 100, amplitude: float = 50.0, frequency: float = 1.0, cache_key: Optional[str] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Create oscillator data (oscillator, ma, range_atr).

        Args:
            size: Number of data points
            amplitude: Oscillator amplitude
            frequency: Sine wave frequency
            cache_key: Optional cache key

        Returns:
            Tuple of (oscillator, ma, range_atr) Series
        """
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        np.random.seed(self._seed)
        dates = pd.date_range("2024-01-01", periods=size, freq="1h")

        # Create oscillator with sine wave
        oscillator = pd.Series(np.sin(np.linspace(0, frequency * 2 * np.pi, size)) * amplitude, index=dates)

        # Create moving average
        ma = (
            pd.Series(np.cumsum(np.ones(size)) / np.arange(1, size + 1), index=dates) * 50000
        )  # Scale to realistic price

        # Create ATR-like range
        range_atr = pd.Series(np.ones(size) * amplitude * 2, index=dates)

        result = (oscillator, ma, range_atr)

        if cache_key:
            self._cache[cache_key] = result

        return result

    def create_series_data(
        self, size: int = 100, values: Optional[np.ndarray] = None, cache_key: Optional[str] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Create high, low, close series.

        Args:
            size: Number of data points
            values: Optional pre-generated values
            cache_key: Optional cache key

        Returns:
            Tuple of (high, low, close) Series
        """
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        np.random.seed(self._seed)
        dates = pd.date_range("2024-01-01", periods=size, freq="1h")

        if values is None:
            values = np.random.randn(size) * 100 + 50000

        high = pd.Series(values + np.abs(np.random.randn(size) * 50), index=dates)
        low = pd.Series(values - np.abs(np.random.randn(size) * 50), index=dates)
        close = pd.Series(values, index=dates)

        result = (high, low, close)

        if cache_key:
            self._cache[cache_key] = result

        return result

    def create_test_signals(
        self, size: int = 100, signal_types: list = [-1, 0, 1], cache_key: Optional[str] = None
    ) -> pd.Series:
        """
        Create test signal series with mixed signal types.

        Args:
            size: Number of data points
            signal_types: List of possible signals
            cache_key: Optional cache key

        Returns:
            Series of signals
        """
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        np.random.seed(self._seed)
        dates = pd.date_range("2024-01-01", periods=size, freq="1h")

        signals = pd.Series(np.random.choice(signal_types, size=size), index=dates, dtype="int8")

        if cache_key:
            self._cache[cache_key] = signals

        return signals

    def create_indicators(
        self, size: int = 100, rsi_period: int = 14, sma_period: int = 20, cache_key: Optional[str] = None
    ) -> Dict[str, pd.Series]:
        """
        Create common technical indicators.

        Args:
            size: Number of data points
            rsi_period: RSI calculation period
            sma_period: SMA calculation period
            cache_key: Optional cache key

        Returns:
            Dictionary with indicator Series
        """
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        np.random.seed(self._seed)
        dates = pd.date_range("2024-01-01", periods=size, freq="1h")

        close = pd.Series(np.random.randn(size) * 100 + 50000, index=dates)

        # Simple indicator calculations
        sma = close.rolling(sma_period).mean()
        ema = close.ewm(span=sma_period).mean()

        # RSI calculation (simplified)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # Bollinger Bands
        std = close.rolling(sma_period).std()
        bb_upper = sma + std * 2
        bb_lower = sma - std * 2

        result = {
            "sma": sma,
            "ema": ema,
            "rsi": rsi,
            "bollinger_upper": bb_upper,
            "bollinger_lower": bb_lower,
        }

        if cache_key:
            self._cache[cache_key] = result

        return result

    def create_edge_case_data(self) -> Dict[str, Any]:
        """
        Create edge case test data sets.

        Returns:
            Dictionary with various edge case data
        """
        edge_cases = {}

        # Tiny dataset
        edge_cases["tiny"] = self.create_ohlc_data(size=3, cache_key="tiny")

        # Single data point
        edge_cases["single"] = self.create_ohlc_data(size=1, cache_key="single")

        # Zero volatility
        edge_cases["zero_vol"] = self.create_ohlc_data(size=50, volatility=0.0, cache_key="zero_vol")

        # High volatility
        edge_cases["high_vol"] = self.create_ohlc_data(size=50, volatility=0.10, cache_key="high_vol")

        # Constant price
        edge_cases["constant"] = self.create_ohlc_data(size=50, volatility=0.0, trend=0.0, cache_key="constant")

        return edge_cases


# Global factory instance for convenience
_global_factory = None


def get_global_factory(seed: int = 42) -> TestDataFactory:
    """Get or create global factory instance."""
    global _global_factory
    if _global_factory is None:
        _global_factory = TestDataFactory(seed=seed)
    return _global_factory


def create_test_data(size: int = 100, seed: int = 42) -> pd.DataFrame:
    """Quick helper to create test data."""
    factory = TestDataFactory(seed=seed)
    return factory.create_ohlc_data(size=size, cache_key=f"size_{size}")


def create_fast_ohlc(size: int = 50) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Fast OHLC data creation (high, low, close)."""
    factory = TestDataFactory()
    return factory.create_series_data(size=size, cache_key=f"fast_ohlc_{size}")


# Pre-cached common test data
FAST_DATA_50 = create_test_data(size=50, seed=42)
FAST_DATA_100 = create_test_data(size=100, seed=42)
FAST_DATA_200 = create_test_data(size=200, seed=42)

FAST_OHLC_50 = create_fast_ohlc(size=50)
FAST_OHLC_100 = create_fast_ohlc(size=100)

FAST_INDICATORS_100 = get_global_factory().create_indicators(size=100, cache_key="fast_indicators_100")
