"""
This conftest_optimized.py file provides pytest fixtures and common test configuration
for the test suite. It is optimized for improved fixture sharing and session-scoped
performance, including efficient synthetic data generation for repeatable and isolated
tests. Warning filters and test utilities are also configured for use across all test modules.
"""

import sys
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Suppress warnings
warnings.filterwarnings("ignore", message=".*The given NumPy array is not writable.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")


# ==========================================================
# 🚀 SESSION-SCOPED FIXTURES (Created once, used for all tests)
# ==========================================================


@pytest.fixture
def session_config():
    """Session configuration object reused across all tests."""
    config = SimpleNamespace(default_symbol="BTC/USDT", default_timeframe="1h", default_limit=100, test_seed=42)
    yield config
    del config


@pytest.fixture
def cached_ohlcv_data():
    """Generate OHLCV data once and cache for all tests."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    # Generate realistic price data
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_price = close + np.random.randn(n) * 25
    volume = np.random.uniform(1000, 5000, n)

    df = pd.DataFrame({"open": open_price, "high": high, "low": low, "close": close, "volume": volume}, index=dates)

    yield df
    del df


@pytest.fixture
def mock_data_fetcher(cached_ohlcv_data):
    """Session-scoped mock data fetcher with cached data."""
    from unittest.mock import Mock

    from modules.common.core.data_fetcher import DataFetcher
    from modules.common.core.exchange_manager import ExchangeManager

    exchange_manager = Mock(spec=ExchangeManager)
    data_fetcher = DataFetcher(exchange_manager)

    # Mock the fetch method to return cached data
    data_fetcher.fetch_ohlcv_with_fallback_exchange = Mock(return_value=(cached_ohlcv_data, "binance"))

    yield data_fetcher
    del data_fetcher


@pytest.fixture
def mock_trained_model():
    """Mock trained model to avoid expensive training."""
    model = Mock()
    model.predict = Mock(return_value=np.array([1, 0, 1, 0, 1]))
    model.predict_proba = Mock(return_value=np.array([[0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.1, 0.9]]))
    yield model
    del model


@pytest.fixture
def precomputed_indicators(cached_ohlcv_data):
    """Precompute common indicators once for all tests."""
    df = cached_ohlcv_data.copy()

    # Add common indicators
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["rsi"] = (
        df["close"]
        .pct_change()
        .rolling(window=14)
        .apply(lambda x: np.mean(x[x > 0]) / (np.mean(np.abs(x)) + 1e-10) * 100)
    )
    df["bollinger_upper"] = df["close"].rolling(window=20).mean() + df["close"].rolling(window=20).std() * 2
    df["bollinger_lower"] = df["close"].rolling(window=20).mean() - df["close"].rolling(window=20).std() * 2
    df["atr"] = df["high"] - df["low"]

    # Remove initial NaN values
    df = df.dropna()

    yield df
    del df


# ==========================================================
# 🚀 MOCKED FIXTURES (Tránh expensive operations)
# ==========================================================


@pytest.fixture
def mock_xgboost_trainer():
    """Mock XGBoost trainer to avoid actual model training."""
    trainer = Mock()
    trainer.train = Mock(return_value={"model": mock_trained_model(), "accuracy": 0.85, "training_time": 0.1})
    yield trainer
    del trainer


@pytest.fixture
def mock_lstm_trainer():
    """Mock LSTM trainer to avoid expensive neural network training."""
    trainer = Mock()
    trainer.train = Mock(return_value={"model": mock_trained_model(), "loss": 0.25, "training_time": 0.5})
    yield trainer
    del trainer


@pytest.fixture
def mock_exchange_manager():
    """Mock exchange manager to avoid network calls."""
    manager = Mock()
    manager.get_markets = Mock(return_value=["BTC/USDT", "ETH/USDT"])
    manager.get_symbol_info = Mock(return_value={"min_amount": 0.001})
    yield manager
    del manager


@pytest.fixture
def mock_api_client():
    """Mock API client to avoid HTTP calls."""
    client = Mock()
    client.get = Mock(return_value={"status": "ok", "data": []})
    client.post = Mock(return_value={"success": True})
    yield client
    del client


# ==========================================================
# 🚀 SMALL DATA FIXTURES (Giảm data size)
# ==========================================================


@pytest.fixture
def small_ohlcv_data():
    """Small dataset for fast tests (100 rows vs 500+)."""
    np.random.seed(42)
    n = 50  # Giảm từ 200 xuống 50
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    base_price = 50000
    prices = base_price + np.cumsum(np.random.randn(n) * 100)

    df = pd.DataFrame(
        {
            "open": prices + np.random.randn(n) * 25,
            "high": prices + np.abs(np.random.randn(n) * 50),
            "low": prices - np.abs(np.random.randn(n) * 50),
            "close": prices,
            "volume": np.random.uniform(1000, 5000, n),
        },
        index=dates,
    )
    yield df
    del df


@pytest.fixture
def tiny_ohlcv_data():
    """Tiny dataset for ultra-fast tests (20 rows)."""
    np.random.seed(42)
    n = 20
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    base_price = 50000
    prices = base_price + np.cumsum(np.random.randn(n) * 50)

    df = pd.DataFrame(
        {
            "open": prices + np.random.randn(n) * 10,
            "high": prices + np.abs(np.random.randn(n) * 20),
            "low": prices - np.abs(np.random.randn(n) * 20),
            "close": prices,
            "volume": np.random.uniform(100, 1000, n),
        },
        index=dates,
    )
    yield df
    del df


# ==========================================================
# 🚀 SHARED FIXTURES (Giúp chia sẻ giữa related tests)
# ==========================================================


@pytest.fixture
def module_test_data():
    """Function-scoped data for tests in the same module."""
    data = {
        "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        "timeframes": ["1h", "4h", "1d"],
        "signals": [-1, 0, 1],
        "confidence_levels": [0.0, 0.5, 1.0],
    }
    yield data
    del data


@pytest.fixture
def common_test_parameters():
    """Function-scoped parameters used across multiple tests."""
    params = SimpleNamespace(
        osc_length=50, osc_mult=2.0, rsi_period=14, sma_period=20, stop_loss_pct=0.02, take_profit_pct=0.04
    )
    yield params
    del params


# ==========================================================
# 🚀 PARAMETRIZED DATA FIXTURES (Tránh lặp code)
# ==========================================================


@pytest.fixture(params=[0.01, 0.02, 0.05])
def risk_percentage(request):
    """Parametrized risk percentage for testing different risk levels."""
    yield request.param


@pytest.fixture(params=[10, 20, 50])
def sequence_length(request):
    """Parametrized sequence length for testing."""
    yield request.param


@pytest.fixture(params=["1h", "4h", "1d"])
def timeframe(request):
    """Parametrized timeframe for testing."""
    yield request.param


# ==========================================================
# 🚀 LEGACY COMPATIBILITY FIXTURES
# ==========================================================


@pytest.fixture
def config_factory():
    """Factory fixture for creating Config instances (original)."""

    def _create_config(**kwargs):
        config = SimpleNamespace()
        config.timeframe = kwargs.get("timeframe", None)
        config.no_menu = kwargs.get("no_menu", False)
        config.enable_spc = kwargs.get("enable_spc", False)
        config.spc_k = kwargs.get("spc_k", None)
        config.enable_xgboost = kwargs.get("enable_xgboost", False)
        config.enable_hmm = kwargs.get("enable_hmm", False)
        config.enable_random_forest = kwargs.get("enable_random_forest", False)
        config.use_decision_matrix = kwargs.get("use_decision_matrix", None)
        config.spc_strategy = kwargs.get("spc_strategy", None)
        config.limit = kwargs.get("limit", None)
        config.max_workers = kwargs.get("max_workers", None)
        return config

    return _create_config


@pytest.fixture
def seeded_random():
    """Seeded NumPy random generator for reproducible tests."""
    rng = np.random.default_rng(42)
    return rng


# ==========================================================
# 🚀 PERFORMANCE MONITORING FIXTURE
# ==========================================================


@pytest.fixture
def performance_monitor():
    """Monitor performance during test execution."""
    import os
    import time

    import psutil

    class Monitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.process = psutil.Process(os.getpid())

        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024

        def stop(self):
            elapsed = time.time() - self.start_time if self.start_time else 0
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_delta = current_memory - self.start_memory if self.start_memory else 0

            return {"elapsed_time": elapsed, "memory_used_mb": current_memory, "memory_delta_mb": memory_delta}

    return Monitor()


# ==========================================================
# 🚀 WINDOWS-SPECIFIC FIXTURES
# ==========================================================


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Configure pytest to avoid capture issues on Windows."""
    if hasattr(config.option, "capture") and config.option.capture == "no":
        if hasattr(sys, "stderr") and sys.stderr is not None:
            try:
                sys.stderr.flush()
            except (ValueError, OSError):
                pass


# ==========================================================
# 🚀 HELPERS FOR TEST WRITING
# ==========================================================


@pytest.fixture
def test_helpers():
    """Helper functions for test writing."""
    return {
        "create_test_signals": lambda: [-1, 0, 1, -1, 0],
        "create_test_confidence": lambda: [0.8, 0.5, 0.9, 0.3, 0.6],
        "assert_signal_valid": lambda s, c: s in [-1, 0, 1] and 0 <= c <= 1,
        "create_ohlc_with_signals": lambda size=100: (
            pd.DataFrame(
                {
                    "open": np.random.randn(size) + 100,
                    "high": np.random.randn(size) + 102,
                    "low": np.random.randn(size) + 98,
                    "close": np.random.randn(size) + 100,
                    "volume": np.random.randint(1000, 10000, size),
                    "signal": np.random.choice([-1, 0, 1], size),
                    "confidence": np.random.uniform(0, 1, size),
                }
            ),
            pd.DataFrame({"open": [100], "high": [102], "low": [98], "close": [100], "volume": [1000]}),
            pd.DataFrame({"signal": [1], "confidence": [0.8]}),
        ),
        "mock_calculator_response": lambda signal=1, confidence=0.8: (signal, confidence),
    }
