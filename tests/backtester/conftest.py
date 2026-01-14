"""
Shared fixtures for backtester tests.
"""

import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock optuna BEFORE any other imports
# This must happen before core.signal_calculators is imported
if "optuna" not in sys.modules:
    optuna_mock = MagicMock()
    sys.modules["optuna"] = optuna_mock
    # Also mock submodules used in imports
    sys.modules["optuna.exceptions"] = MagicMock()
    sys.modules["optuna.trial"] = MagicMock()
    sys.modules["optuna.study"] = MagicMock()

# Add project root to path (same as tests/conftest.py)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from unittest.mock import patch

import numpy as np
import pandas as pd

# Import modules for patching


# Global cache for pre-computed data (session-scoped)
_PRECOMPUTED_DATA = {}


def _precompute_small_data():
    """Pre-compute small DataFrame (50 periods) once per session."""
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
    return _PRECOMPUTED_DATA["small"].copy()


def _precompute_medium_data():
    """Pre-compute medium DataFrame (150 periods) once per session."""
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


@pytest.fixture
def mock_ohlcv_data():
    """Generate mock OHLCV data for testing.

    Default is optimized for fast unit tests (50 periods).
    Use mock_medium_ohlcv_data for integration tests.
    """

    def _generate_data(periods=50, base_price=100.0, volatility=0.5):
        # Use pre-computed data for standard sizes
        if periods == 50 and base_price == 100.0 and volatility == 0.5:
            return _precompute_small_data()

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
    """Generate medium mock OHLCV data for integration tests (150 periods)."""

    def _generate_data(periods=150, base_price=100.0, volatility=0.5):
        # Use pre-computed data for standard sizes
        if periods == 150 and base_price == 100.0 and volatility == 0.5:
            return _precompute_medium_data()

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
def mock_data_fetcher(mock_ohlcv_data):
    """Create a fully mocked DataFetcher that doesn't call real APIs.

    Optimized: Uses cached data and SimpleNamespace for speed.
    """

    class FastMockDataFetcher:
        """Lightweight data fetcher without Mock overhead."""

        def __init__(self, data_func):
            self.data_func = data_func
            self.market_prices = {}
            self._ohlcv_dataframe_cache = {}

        def fetch_ohlcv_with_fallback_exchange(self, symbol, **kwargs):
            """Mock fetch function that returns generated data."""
            limit = kwargs.get("limit", 50)
            df = self.data_func(periods=limit)
            return df, "binance"

        def fetch_binance_account_balance(self):
            """Return None - no account balance needed for testing."""
            return None

    return FastMockDataFetcher(mock_ohlcv_data)


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


class LightweightDataFetcher:
    """Lightweight data fetcher without Mock overhead.

    Returns pre-computed DataFrame - no data generation.
    """

    def __init__(self, df=None):
        self.df = df if df is not None else _precompute_small_data()

    def fetch_ohlcv_with_fallback_exchange(self, symbol, **kwargs):
        """Return pre-computed DataFrame."""
        limit = kwargs.get("limit", len(self.df))
        return self.df.head(limit), "binance"


@pytest.fixture
def fast_data_fetcher(precomputed_small_df):
    """Fast fixture - lightweight data fetcher without Mock overhead."""
    return LightweightDataFetcher(precomputed_small_df)


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


@pytest.fixture
def fast_no_signal_calculator():
    """Fast fixture - signal calculator that returns no signals."""
    return LightweightSignalCalculator(signal=0, confidence=0.0)


# Pre-defined signal return values (cached to avoid recreation each test)
_SIGNAL_RETURN_VALUES = {
    "range_oscillator": (1, 0.7),
    "spc": (1, 0.6),
    "xgboost": (1, 0.8),
    "hmm": (1, 0.65),
    "random_forest": (1, 0.75),
}


@pytest.fixture(autouse=True, scope="function")
def auto_mock_signal_calculators():
    """Automatically mock signal calculators for all tests to prevent API calls.

    This is set to autouse=True to ensure no API calls are made by default.
    Individual tests can override this if needed.

    Optimized: Uses pre-computed return values for speed.
    """
    # Import core.signal_calculators first (optuna is already mocked)
    # Then patch functions where they are imported in modules.position_sizing.core.indicator_calculators
    try:
        import core.signal_calculators  # noqa: F401

        # Patch at usage site in IndicatorCalculatorsMixin
        # Use pre-computed values for speed
        with (
            patch(
                "modules.position_sizing.core.indicator_calculators.get_range_oscillator_signal",
                return_value=_SIGNAL_RETURN_VALUES["range_oscillator"],
            ),
            patch(
                "modules.position_sizing.core.indicator_calculators.get_spc_signal",
                return_value=_SIGNAL_RETURN_VALUES["spc"],
            ),
            patch(
                "modules.position_sizing.core.indicator_calculators.get_xgboost_signal",
                return_value=_SIGNAL_RETURN_VALUES["xgboost"],
            ),
            patch(
                "modules.position_sizing.core.indicator_calculators.get_hmm_signal",
                return_value=_SIGNAL_RETURN_VALUES["hmm"],
            ),
            patch(
                "modules.position_sizing.core.indicator_calculators.get_random_forest_signal",
                return_value=_SIGNAL_RETURN_VALUES["random_forest"],
            ),
        ):
            yield
    except (ImportError, AttributeError):
        # If import fails, skip patching (tests will need to handle this)
        yield


@pytest.fixture
def mock_signal_calculators():
    """Mock all signal calculator functions to avoid API calls.

    Use this fixture in tests to prevent API calls to exchanges.
    """
    # Patch at usage site in HybridSignalCalculator (actually IndicatorCalculatorsMixin)
    with (
        patch("modules.position_sizing.core.indicator_calculators.get_range_oscillator_signal") as mock_osc,
        patch("modules.position_sizing.core.indicator_calculators.get_spc_signal") as mock_spc,
        patch("modules.position_sizing.core.indicator_calculators.get_xgboost_signal") as mock_xgb,
        patch("modules.position_sizing.core.indicator_calculators.get_hmm_signal") as mock_hmm,
        patch("modules.position_sizing.core.indicator_calculators.get_random_forest_signal") as mock_rf,
    ):
        # Use pre-computed values for speed
        mock_osc.return_value = _SIGNAL_RETURN_VALUES["range_oscillator"]
        mock_spc.return_value = _SIGNAL_RETURN_VALUES["spc"]
        mock_xgb.return_value = _SIGNAL_RETURN_VALUES["xgboost"]
        mock_hmm.return_value = _SIGNAL_RETURN_VALUES["hmm"]
        mock_rf.return_value = _SIGNAL_RETURN_VALUES["random_forest"]

        yield {
            "range_oscillator": mock_osc,
            "spc": mock_spc,
            "xgboost": mock_xgb,
            "hmm": mock_hmm,
            "random_forest": mock_rf,
        }


@pytest.fixture
def mock_hybrid_signal_calculator():
    """Mock HybridSignalCalculator to return predictable signals."""
    with patch("modules.position_sizing.core.hybrid_signal_calculator.HybridSignalCalculator") as mock_class:
        mock_instance = MagicMock()
        # Return alternating signals for testing
        signals = [1, 0, 1, -1, 0, 1] * 100  # Repeat pattern
        confidences = [0.8, 0.0, 0.7, 0.6, 0.0, 0.75] * 100

        def calculate_signal(df, symbol, timeframe, period_index, signal_type, **kwargs):
            idx = period_index % len(signals)
            return (signals[idx], confidences[idx])

        mock_instance.calculate_hybrid_signal = MagicMock(side_effect=calculate_signal)
        mock_instance.get_cache_stats = MagicMock(
            return_value={
                "signal_cache_size": 0,
                "signal_cache_max_size": 200,
                "data_cache_size": 0,
                "data_cache_max_size": 10,
            }
        )
        mock_instance.enabled_indicators = ["range_oscillator", "spc", "xgboost", "hmm", "random_forest"]
        mock_instance.use_confidence_weighting = True
        mock_instance.min_indicators_agreement = 3

        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_regime_detector():
    """Mock RegimeDetector to return predictable regimes."""
    with patch("modules.position_sizing.core.regime_detector.RegimeDetector") as mock_class:
        mock_instance = MagicMock()
        # Return alternating regimes
        regimes = ["BULLISH", "NEUTRAL", "BEARISH", "NEUTRAL"] * 25

        def detect_regime(symbol, timeframe, limit):
            # Simple hash-based regime selection for consistency
            hash_val = hash(f"{symbol}{timeframe}") % len(regimes)
            return regimes[hash_val]

        mock_instance.detect_regime = MagicMock(side_effect=detect_regime)
        mock_class.return_value = mock_instance
        yield mock_instance


# Pytest hooks for test optimization
def pytest_configure(config):
    """Configure pytest markers and options."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests that measure performance")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on options."""
    if config.getoption("--fast"):
        skip_slow = pytest.mark.skip(reason="Skipped with --fast flag")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Run tests without slow tests",
    )
