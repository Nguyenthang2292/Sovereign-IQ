"""
Improved fixtures for ultra-fast backtester tests.

Key optimizations:
1. Session-scoped fixtures to cache data across tests
2. Pre-computed mock data to avoid DataFrame generation
3. Simplified mocks using lightweight objects instead of Mock
4. Direct signal mocking to skip signal calculation
"""

import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

# Mock optuna BEFORE any other imports
if "optuna" not in sys.modules:
    optuna_mock = MagicMock()
    sys.modules["optuna"] = optuna_mock
    sys.modules["optuna.exceptions"] = MagicMock()
    sys.modules["optuna.trial"] = MagicMock()
    sys.modules["optuna.study"] = MagicMock()

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ==================== FAST CACHED DATA FIXTURES ====================

# Pre-compute data at session scope
_PRECOMPUTED_DATA = {}


@pytest.fixture(scope="session")
def precomputed_small_df():
    """Pre-computed small DataFrame (50 periods) cached for entire session."""
    if "small" not in _PRECOMPUTED_DATA:
        np.random.seed(42)
        periods = 50
        dates = pd.date_range("2023-01-01", periods=periods, freq="h")
        returns = np.random.randn(periods) * 0.5
        prices = 100 + np.cumsum(returns)
        prices = np.maximum(prices, 10.0)

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.uniform(1000, 10000, periods),
            },
            index=dates,
        )
        _PRECOMPUTED_DATA["small"] = df.copy()

    # Return a copy to prevent test modifications
    return _PRECOMPUTED_DATA["small"].copy()


@pytest.fixture(scope="session")
def precomputed_medium_df():
    """Pre-computed medium DataFrame (150 periods) cached for entire session."""
    if "medium" not in _PRECOMPUTED_DATA:
        np.random.seed(43)
        periods = 150
        dates = pd.date_range("2023-01-01", periods=periods, freq="h")
        returns = np.random.randn(periods) * 0.5
        prices = 100 + np.cumsum(returns)
        prices = np.maximum(prices, 10.0)

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.uniform(1000, 10000, periods),
            },
            index=dates,
        )
        _PRECOMPUTED_DATA["medium"] = df.copy()

    return _PRECOMPUTED_DATA["medium"].copy()


@pytest.fixture(scope="session")
def precomputed_signals():
    """Pre-computed signals array (50 periods) cached for entire session."""
    if "signals" not in _PRECOMPUTED_DATA:
        # Pattern: [1, 0, 0, 1, 0, 0, 1, 0, 0] - signal every 3rd period
        signals = np.array([1 if i % 3 == 0 else 0 for i in range(50)], dtype=np.int64)
        _PRECOMPUTED_DATA["signals"] = signals

    return _PRECOMPUTED_DATA["signals"].copy()


# ==================== LIGHTWEIGHT MOCK OBJECTS ====================


class LightweightSignalCalculator:
    """Lightweight signal calculator without Mock overhead.

    This is ~10x faster than using MagicMock for signal calculation.
    """

    def __init__(self, signal=1, confidence=0.7):
        self.signal = signal
        self.confidence = confidence
        self.call_count = 0

    def calculate_hybrid_signal(self, df, symbol, timeframe, period_index, signal_type, **kwargs):
        """Return fixed signal - no calculation needed."""
        self.call_count += 1
        return (self.signal, self.confidence)

    def calculate_single_signal_highest_confidence(self, df, symbol, timeframe, period_index, **kwargs):
        """Return fixed signal - no calculation needed."""
        self.call_count += 1
        return (self.signal, self.confidence)

    def get_cache_stats(self):
        """Return simple cache stats."""
        return {
            "signal_cache_size": 0,
            "signal_cache_max_size": 1000,
            "cache_hit_rate": 0.0,
        }


class LightweightDataFetcher:
    """Lightweight data fetcher without Mock overhead.

    Returns pre-computed DataFrame - no data generation.
    """

    def __init__(self, df):
        """Initialize with a DataFrame. df is required."""
        if df is None:
            raise ValueError("df is required - pass precomputed_small_df fixture")
        self.df = df

    def fetch_ohlcv_with_fallback_exchange(self, symbol, **kwargs):
        """Return pre-computed DataFrame."""
        limit = kwargs.get("limit", len(self.df))
        if limit <= len(self.df):
            return self.df.iloc[:limit].copy(), "binance"
        return self.df.copy(), "binance"

    def fetch_binance_account_balance(self):
        """Return None - no account balance needed for testing."""
        return None

    @property
    def market_prices(self):
        return {}

    @property
    def _ohlcv_dataframe_cache(self):
        return {}


# ==================== ULTRA-FAST FIXTURES ====================


@pytest.fixture
def fast_df(precomputed_small_df):
    """Fast fixture - returns pre-computed DataFrame."""
    return precomputed_small_df


@pytest.fixture
def fast_signals(precomputed_signals):
    """Fast fixture - returns pre-computed signals."""
    return precomputed_signals


@pytest.fixture
def fast_signal_calculator():
    """Fast fixture - lightweight signal calculator without Mock overhead."""
    return LightweightSignalCalculator(signal=1, confidence=0.7)


@pytest.fixture
def fast_no_signal_calculator():
    """Fast fixture - signal calculator that returns no signals."""
    return LightweightSignalCalculator(signal=0, confidence=0.0)


@pytest.fixture
def fast_data_fetcher(precomputed_small_df):
    """Fast fixture - lightweight data fetcher without Mock overhead."""
    return LightweightDataFetcher(precomputed_small_df)


@pytest.fixture
def fast_data_fetcher_medium(precomputed_medium_df):
    """Fast fixture - data fetcher with medium DataFrame."""
    return LightweightDataFetcher(precomputed_medium_df)


# ==================== MINIMAL MOCK FIXTURES ====================


@pytest.fixture
def minimal_mock_data_fetcher():
    """Minimal mock using SimpleNamespace - fastest option."""

    class MinimalFetcher:
        def fetch_ohlcv_with_fallback_exchange(self, symbol, **kwargs):
            # Return minimal DataFrame - only what's needed
            periods = 20
            dates = pd.date_range("2023-01-01", periods=periods, freq="h")
            prices = np.linspace(100, 105, periods)
            df = pd.DataFrame(
                {
                    "open": prices,
                    "high": prices * 1.01,
                    "low": prices * 0.99,
                    "close": prices,
                    "volume": [5000] * periods,
                },
                index=dates,
            )
            return df, "binance"

        def fetch_binance_account_balance(self):
            return None

        @property
        def market_prices(self):
            return {}

        @property
        def _ohlcv_dataframe_cache(self):
            return {}

    return MinimalFetcher()


@pytest.fixture
def fast_simple_backtester(minimal_mock_data_fetcher):
    """Create backtester with minimal configuration for ultra-fast tests."""
    from modules.backtester.core.backtester import FullBacktester

    # Use lightweight signal calculator
    signal_calc = LightweightSignalCalculator(signal=1, confidence=0.7)

    backtester = FullBacktester(
        data_fetcher=minimal_mock_data_fetcher,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        trailing_stop_pct=0.015,
        max_hold_periods=50,
    )

    # Replace signal calculator (bypasses complex initialization)
    backtester.hybrid_signal_calculator = signal_calc

    return backtester


# ==================== PATCH-LEVEL MOCKS (SKIP CALCULATION) ====================


@pytest.fixture
def skip_heavy_calculations():
    """Fixture to skip heavy calculations at patch level.

    This is ~20x faster than patching individual functions because it
    bypasses all signal calculation entirely.
    """
    # Patch at the highest level - before any calculation happens
    with patch("modules.backtester.core.backtester.calculate_signals", return_value=None):
        with patch("modules.backtester.core.backtester.calculate_single_signals", return_value=None):
            yield


# ==================== BENCHMARK-READY FIXTURES ====================


@pytest.fixture
def benchmark_df():
    """DataFrame optimized for benchmarking (exactly 100 periods)."""
    np.random.seed(999)
    periods = 100
    dates = pd.date_range("2023-01-01", periods=periods, freq="h")
    returns = np.random.randn(periods) * 0.5
    prices = 100 + np.cumsum(returns)
    prices = np.maximum(prices, 10.0)

    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.uniform(1000, 10000, periods),
        },
        index=dates,
    )


@pytest.fixture
def benchmark_signals():
    """Signals optimized for benchmarking (exactly 100 periods)."""
    # Pattern: 10 signals evenly distributed
    return np.array([1 if i % 10 == 0 else 0 for i in range(100)], dtype=np.int64)


# ==================== COMPATIBILITY FIXTURES ====================

# These provide compatibility with existing tests while using faster implementations


@pytest.fixture
def mock_ohlcv_data(precomputed_small_df):
    """Compatibility fixture - wraps precomputed data."""
    cached_df = precomputed_small_df  # Capture the fixture value

    def _generate_data(periods=50, base_price=100.0, volatility=0.5):
        if periods == 50 and base_price == 100.0 and volatility == 0.5:
            return cached_df.copy()

        # Fallback to generation for non-standard sizes
        np.random.seed(42 + periods)
        dates = pd.date_range("2023-01-01", periods=periods, freq="h")
        returns = np.random.randn(periods) * volatility
        prices = base_price + np.cumsum(returns)
        prices = np.maximum(prices, base_price * 0.1)

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.uniform(1000, 10000, periods),
            },
            index=dates,
        )
        return df

    return _generate_data


@pytest.fixture
def mock_medium_ohlcv_data():
    """Compatibility fixture - wraps precomputed medium data."""

    def _generate_data(periods=150, base_price=100.0, volatility=0.5):
        if periods == 150 and base_price == 100.0 and volatility == 0.5:
            return precomputed_medium_df()

        np.random.seed(43 + periods)
        dates = pd.date_range("2023-01-01", periods=periods, freq="h")
        returns = np.random.randn(periods) * volatility
        prices = base_price + np.cumsum(returns)
        prices = np.maximum(prices, base_price * 0.1)

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.uniform(1000, 10000, periods),
            },
            index=dates,
        )
        return df

    return _generate_data


@pytest.fixture
def mock_data_fetcher(precomputed_small_df):
    """Compatibility fixture - uses fast lightweight implementation."""
    return LightweightDataFetcher(precomputed_small_df)
