"""Test specific fixtures for performance optimization.

These fixtures are specifically designed to work with existing tests
without breaking them while providing significant speed improvements.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# ==========================================================
# 🚀 FAST DATA FIXTURES (Session-scoped)
# ==========================================================


@pytest.fixture(scope="session")
def fast_ohlcv_data():
    """Fast OHLCV data - created once, reused by all tests."""
    np.random.seed(42)
    n = 100  # Optimized size for speed

    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    base_price = 50000.0
    prices = base_price + np.cumsum(np.random.randn(n) * 50)

    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + np.abs(np.random.randn(n) * 20),
            "low": prices - np.abs(np.random.randn(n) * 20),
            "close": prices,
            "volume": np.random.uniform(1000, 5000, n),
        },
        index=dates,
    )


@pytest.fixture(scope="session")
def fast_data_fetcher(fast_ohlcv_data):
    """Fast data fetcher with session-scoped data."""
    from modules.common.core.data_fetcher import DataFetcher
    from modules.common.core.exchange_manager import ExchangeManager

    exchange_manager = Mock(spec=ExchangeManager)
    data_fetcher = DataFetcher(exchange_manager)

    # Pre-cache the data
    data_fetcher.fetch_ohlcv_with_fallback_exchange = Mock(return_value=(fast_ohlcv_data, "binance"))

    return data_fetcher


@pytest.fixture(scope="session")
def fast_indicators():
    """Pre-computed indicators for fast testing."""
    np.random.seed(42)
    n = 50  # Smaller size for indicator tests

    data = pd.Series(np.random.randn(n), name="close")

    return {
        "rsi": data.rolling(14).apply(lambda x: np.mean(x[x > 0]) / (np.mean(np.abs(x)) + 1e-10) * 100),
        "sma": data.rolling(20).mean(),
        "ema": data.ewm(span=20).mean(),
        "bollinger_upper": data.rolling(20).mean() + data.rolling(20).std() * 2,
        "bollinger_lower": data.rolling(20).mean() - data.rolling(20).std() * 2,
    }


# ==========================================================
# 🚀 MOCKED MODEL FIXTURES (Session-scoped)
# ==========================================================


@pytest.fixture(scope="session")
def mocked_random_forest_model():
    """Mock Random Forest model to avoid loading actual model."""
    model = Mock()
    model.predict = Mock(return_value=np.array([1, 0, 1, 0, 1]))
    model.predict_proba = Mock(return_value=np.array([[0.2, 0.8], [0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.1, 0.9]]))
    return model


@pytest.fixture(scope="session")
def mocked_xgboost_model():
    """Mock XGBoost model to avoid training."""
    model = Mock()
    model.predict = Mock(return_value=np.array([0, 1, 0]))
    model.predict_proba = Mock(return_value=np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]]))
    return model


@pytest.fixture(scope="session")
def mocked_hmm_model():
    """Mock HMM model to avoid expensive training."""
    model = Mock()
    model.decode = Mock(return_value=(np.array([0, 1, 0]), np.log(np.array([0.7, 0.2, 0.1]))))
    return model


# ==========================================================
# 🚀 LIGHTWEIGHT CONFIG FIXTURES
# ==========================================================


@pytest.fixture(scope="session")
def fast_config():
    """Lightweight configuration for tests."""
    from types import SimpleNamespace

    return SimpleNamespace(
        symbol="BTC/USDT",
        timeframe="1h",
        limit=50,  # Reduced for speed
        osc_length=20,  # Reduced from 50
        osc_mult=2.0,
        enable_hmm=False,  # Disable expensive HMM by default
        enable_xgboost=False,  # Disable expensive XGBoost by default
        use_mock_models=True,  # Enable mocking by default
    )


# ==========================================================
# 🚀 PARAMETRIZED TEST DATA FIXTURES
# ==========================================================


@pytest.fixture(params=[0.01, 0.02, 0.05])
def test_risk_percentage(request):
    """Parametrized risk percentages."""
    return request.param


@pytest.fixture(params=[10, 20, 50])
def test_sequence_length(request):
    """Parametrized sequence lengths."""
    return request.param


@pytest.fixture(params=["1h", "4h"])
def test_timeframe(request):
    """Parametrized timeframes."""
    return request.param


# ==========================================================
# 🚀 PERFORMANCE MONITORING FIXTURE
# ==========================================================


@pytest.fixture
def performance_tracker():
    """Track performance during test execution."""
    import time

    class Tracker:
        def __init__(self):
            self.start_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            if self.start_time:
                elapsed = time.time() - self.start_time
                print(f"Test execution time: {elapsed:.3f}s")
                return elapsed
            return 0

    return Tracker()


# ==========================================================
# 🚀 UTILITY FIXTURES FOR EXISTING TESTS
# ==========================================================


@pytest.fixture
def sample_ohlc_data():
    """Optimized version of existing sample_ohlc_data fixture."""
    np.random.seed(42)
    n = 100  # Reduced from 200

    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(n) * 50)
    high = close + np.abs(np.random.randn(n) * 25)
    low = close - np.abs(np.random.randn(n) * 25)

    return pd.Series(high, index=dates), pd.Series(low, index=dates), pd.Series(close, index=dates)


@pytest.fixture
def sample_oscillator_data():
    """Optimized oscillator data for testing."""
    np.random.seed(42)
    n = 100  # Reduced size

    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = pd.Series(50000 + np.cumsum(np.random.randn(n) * 50), index=dates)

    oscillator = pd.Series(np.sin(np.linspace(0, 2 * np.pi, n)) * 50, index=close.index)
    ma = close.rolling(20).mean()
    range_atr = pd.Series(np.ones(n) * 500, index=close.index)

    return oscillator, ma, range_atr


# ==========================================================
# 🚀 PATCHED IMPORTS FOR SPEED
# ==========================================================


@pytest.fixture(autouse=True)
def patch_expensive_imports():
    """Patch expensive imports to speed up test loading."""
    with patch.dict(
        "sys.modules",
        {
            # Mock expensive modules that slow down import
            "torch": Mock(),
            "tensorflow": Mock(),
            "sklearn.ensemble._forest": Mock(),
            "xgboost": Mock(),
        },
    ):
        yield


# ==========================================================
# 🚀 TEST MARKER FIXTURES
# ==========================================================


@pytest.fixture
def skip_slow_tests():
    """Skip slow tests by default - use -m 'not slow' to enable."""
    pytest.skip("Slow test - run with -m 'not slow' to skip")


@pytest.fixture
def enable_slow_tests():
    """Enable slow tests when explicitly requested."""
    return True


# ==========================================================
# 🚀 HELPER FUNCTIONS FOR TEST WRITING
# ==========================================================


def create_test_signal_data(size=50):
    """Quick helper to create test data with signals."""
    np.random.seed(42)

    return pd.DataFrame(
        {
            "signal": np.random.choice([-1, 0, 1], size),
            "confidence": np.random.uniform(0, 1, size),
            "timestamp": pd.date_range("2024-01-01", periods=size, freq="1h"),
        }
    )


def assert_signal_result(result):
    """Quick assertion helper for signal results."""
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2
    signal, confidence = result
    assert signal in [-1, 0, 1]
    assert 0 <= confidence <= 1


def create_mock_test_context():
    """Create a mock context for testing."""
    return {"symbol": "BTC/USDT", "timeframe": "1h", "limit": 50, "expected_signal": 1, "expected_confidence": 0.8}
