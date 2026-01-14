"""
This module configures pytest settings and provides fixtures for testing.
It suppresses certain warnings, handles pytest capture issues on Windows,
and defines reusable fixtures to streamline test setup.
"""

import sys
import tracemalloc
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

# Suppress warnings from pytorch_forecasting about non-writable NumPy arrays
warnings.filterwarnings("ignore", message=".*The given NumPy array is not writable.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_forecasting")


# Workaround for pytest capture bug on Windows with Python 3.12+
# This prevents the "I/O operation on closed file" error
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Configure pytest to avoid capture issues on Windows."""

    # Ensure capture is disabled if not already set
    if hasattr(config.option, "capture") and config.option.capture == "no":
        # Set up safe stderr handling
        if hasattr(sys, "stderr") and sys.stderr is not None:
            try:
                sys.stderr.flush()
            except (ValueError, OSError):
                pass

    # Add memory-related markers
    config.addinivalue_line("markers", "memory_intensive: marks tests that use significant RAM")


@pytest.fixture
def config_factory():
    """Factory fixture for creating Config instances for testing."""

    def _create_config(**kwargs):
        """Create a Config instance with specified attributes."""
        from types import SimpleNamespace

        config = SimpleNamespace()
        # Set default values for all possible attributes
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
    """
    Fixture to provide a seeded NumPy random generator.

    Returns a NumPy Generator instance seeded with 42 for reproducible
    random number generation in tests.

    Usage:
        def test_something(seeded_random):
            # Use the seeded generator
            data = seeded_random.standard_normal(100)
            # ... rest of test
    """
    rng = np.random.default_rng(42)
    return rng


# ==================== SESSION FIXTURES FOR MEMORY OPTIMIZATION ====================


@pytest.fixture(scope="session")
def session_small_df():
    """Session-scoped small DataFrame (50 periods) for memory optimization.

    This fixture shares a single DataFrame across all tests in the session,
    reducing memory usage by ~60-70% for data-heavy tests.
    """
    # Use seeded random for reproducible data
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
    return df


@pytest.fixture(scope="session")
def session_medium_df():
    """Session-scoped medium DataFrame (150 periods) for memory optimization.

    This fixture shares a single DataFrame across all tests in the session,
    reducing memory usage for integration tests.
    """
    # Use seeded random for reproducible data
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
    return df


@pytest.fixture(scope="session")
def session_mock_data_fetcher(session_small_df):
    """Session-scoped data fetcher for memory optimization.

    Use this instead of mock_data_fetcher in new tests to save RAM.
    Shares data across all tests in the session.
    """

    class FastMockDataFetcher:
        """Lightweight data fetcher without Mock overhead."""

        def __init__(self, df):
            self.df = df
            self.market_prices = {}
            self._ohlcv_dataframe_cache = {}

        def fetch_ohlcv_with_fallback_exchange(self, symbol, **kwargs):
            """Mock fetch function that returns generated data."""
            limit = kwargs.get("limit", len(self.df))
            if limit <= len(self.df):
                return self.df.iloc[:limit].copy(), "binance"
            return self.df.copy(), "binance"

        def fetch_binance_account_balance(self):
            """Return None - no account balance needed for testing."""
            return None

    return FastMockDataFetcher(session_small_df)


@pytest.fixture
def optimized_mock_data_fetcher(session_mock_data_fetcher):
    """Function fixture that returns session data fetcher.

    This provides function scope access to session data for memory savings
    while maintaining test isolation through the stateless fetcher.
    """
    return session_mock_data_fetcher


# ==================== LAZY LOADING GENERATORS ====================


def _lazy_data_generator(base_price=100.0, volatility=0.5, max_periods=1000):
    """Lazy generator for OHLCV data - generates data on demand.

    This reduces initial memory usage by generating data only when needed,
    and can handle unlimited periods without preallocating memory.
    """
    np.random.seed(42)  # Deterministic seed
    periods_generated = 0

    while periods_generated < max_periods:
        # Generate data in chunks to avoid large memory allocations
        chunk_size = min(100, max_periods - periods_generated)
        dates = pd.date_range("2023-01-01", periods=chunk_size, freq="h").shift(
            periods_generated
        )  # Shift to continue from last date

        returns = np.random.randn(chunk_size) * volatility
        prices = base_price + np.cumsum(returns)
        prices = np.maximum(prices, base_price * 0.1)

        chunk_df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.uniform(1000, 10000, chunk_size),
            },
            index=dates,
        )

        yield chunk_df
        periods_generated += chunk_size
        base_price = prices[-1]  # Continue from last price


@pytest.fixture
def lazy_data_generator():
    """Lazy data generator fixture for on-demand data creation.

    Usage:
        def test_something(lazy_data_generator):
            gen = lazy_data_generator()
            data_50 = next(gen)  # Get first 50 periods
            data_100 = next(gen)  # Get next 50 periods (51-100)
    """
    return lambda: _lazy_data_generator()


@pytest.fixture
def lazy_mock_data_fetcher(lazy_data_generator):
    """Lazy data fetcher that generates data only when requested.

    This is more memory efficient than pre-generating all data.
    """

    class LazyMockDataFetcher:
        """Data fetcher with lazy loading."""

        def __init__(self, generator_func):
            self.generator_func = generator_func
            self._cache = {}
            self._generator = None

        def _get_generator(self):
            if self._generator is None:
                self._generator = self.generator_func()
            return self._generator

        def fetch_ohlcv_with_fallback_exchange(self, symbol, **kwargs):
            """Fetch data with lazy loading."""
            limit = kwargs.get("limit", 50)
            key = (symbol, limit)

            # Check cache first
            if key in self._cache:
                return self._cache[key].copy(), "binance"

            # Generate required data
            generator = self._get_generator()
            collected_data = []
            total_records = 0

            while total_records < limit:
                try:
                    chunk = next(generator)
                    collected_data.append(chunk)
                    total_records += len(chunk)
                    if total_records >= limit:
                        break
                except StopIteration:
                    break

            if collected_data:
                result_df = pd.concat(collected_data).head(limit)
                self._cache[key] = result_df.copy()
                return result_df, "binance"

            # Fallback to empty data if generator exhausted
            return pd.DataFrame(), "binance"

        def fetch_binance_account_balance(self):
            """Return None - no account balance needed for testing."""
            return None

    return LazyMockDataFetcher(lazy_data_generator)


# ==================== MEMORY MONITORING FIXTURES ====================


@pytest.fixture
def memory_monitor():
    """Fixture for monitoring memory usage during tests.

    Usage:
        def test_something(memory_monitor):
            with memory_monitor.track() as tracker:
                # Your test code
                df = create_large_dataframe()

            print(f"Memory used: {tracker.peak_memory}MB")
            assert tracker.peak_memory < 100  # Max 100MB
    """
    try:
        import psutil

        class MemoryTracker:
            def __init__(self):
                self.process = psutil.Process()
                self.initial_memory = None
                self.peak_memory = 0
                self.final_memory = None

            @contextmanager
            def track(self):
                """Track memory usage. Note: peak_memory tracks Python allocations only (via tracemalloc),
                while initial/final_memory track total process RSS."""
                tracemalloc.start()
                self.initial_memory = self.process.memory_info().rss / 1024 / 1024
                try:
                    yield self
                finally:
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    self.peak_memory = peak / 1024 / 1024  # True peak in MB
                    self.final_memory = self.process.memory_info().rss / 1024 / 1024

            def get_memory_usage(self):
                """Get current memory usage in MB."""
                return self.process.memory_info().rss / 1024 / 1024

            def assert_memory_under(self, limit_mb):
                """Assert that memory usage is under limit."""
                current = self.get_memory_usage()
                assert current < limit_mb, f"Memory usage {current:.1f}MB exceeds limit {limit_mb}MB"

        return MemoryTracker()

    except ImportError:
        # Fallback if psutil not available
        class DummyTracker:
            def __init__(self):
                self.initial_memory = 0
                self.peak_memory = 0
                self.final_memory = 0

            @contextmanager
            def track(self):
                yield self

            def get_memory_usage(self):
                return 0

            def assert_memory_under(self, limit_mb):
                pass

        return DummyTracker()


# ==================== ADVANCED CACHING WITH TTL ====================

import time
from threading import Lock


class TTLCache:
    """Time-based cache with automatic cleanup."""

    def __init__(self, ttl_seconds=300):  # 5 minutes default
        self._cache = {}
        self._timestamps = {}
        self._ttl = ttl_seconds
        self._lock = Lock()

    def get(self, key):
        """Get item from cache, None if expired or not found."""
        with self._lock:
            if key in self._cache:
                if time.time() - self._timestamps[key] < self._ttl:
                    return self._cache[key]
                else:
                    # Expired, remove
                    del self._cache[key]
                    del self._timestamps[key]
            return None

    def set(self, key, value):
        """Set item in cache with current timestamp."""
        with self._lock:
            self._cache[key] = value
            self._timestamps[key] = time.time()

    def clear_expired(self):
        """Manually clear expired items."""
        with self._lock:
            current_time = time.time()
            expired_keys = [key for key, timestamp in self._timestamps.items() if current_time - timestamp >= self._ttl]
            for key in expired_keys:
                del self._cache[key]
                del self._timestamps[key]

    def size(self):
        """Get cache size."""
        with self._lock:
            return len(self._cache)


# Global TTL cache for expensive data
_TTL_CACHE = TTLCache(ttl_seconds=600)  # 10 minutes


@pytest.fixture(scope="session")
def ttl_cache():
    """Session-scoped TTL cache for expensive data."""
    return _TTL_CACHE


@pytest.fixture
def cached_data_factory(ttl_cache):
    """Factory for creating cached data with TTL.

    Usage:
        def test_something(cached_data_factory):
            # Get cached data, or create if not exists
            data = cached_data_factory.get_or_create(
                'large_dataset',
                lambda: create_expensive_data()
            )
    """

    def get_or_create(key, factory_func):
        """Get from cache or create with factory function."""
        cached = ttl_cache.get(key)
        if cached is not None:
            return cached

        # Create new data
        data = factory_func()
        ttl_cache.set(key, data)
        return data

    return get_or_create


# ==================== MEMORY OPTIMIZATION HOOKS ====================


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Automatically cleanup memory after each test to prevent RAM buildup."""
    import gc

    yield  # Test runs here

    # Force garbage collection after each test
    gc.collect()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Memory monitoring setup before each test."""
    if item.config.getoption("--memory-profile"):
        # Store initial memory usage
        try:
            import psutil

            process = psutil.Process()
            item._initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            item._initial_memory = None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    """Additional cleanup between tests for memory optimization."""
    import gc

    # Memory profiling
    if hasattr(item, "_initial_memory") and item._initial_memory is not None:
        try:
            import psutil

            process = psutil.Process()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - item._initial_memory
            threshold = item.config.getoption("--memory-threshold") * 1024  # Convert GB to MB

            if memory_delta > threshold:
                print(f"\n⚠️  HIGH MEMORY USAGE: {item.name} used {memory_delta:.1f}MB (threshold: {threshold:.1f}MB)")

        except ImportError:
            pass

    # Force collection after each test
    gc.collect()

    # Clear matplotlib figures if any
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        pass


def pytest_addoption(parser):
    """Add custom command line options for memory optimization."""
    parser.addoption(
        "--memory-limit",
        action="store",
        default=None,
        help="Skip tests if available memory is below this threshold (in GB)",
    )
    parser.addoption(
        "--skip-memory-intensive",
        action="store_true",
        default=False,
        help="Skip memory-intensive tests",
    )
    parser.addoption(
        "--memory-profile",
        action="store_true",
        default=False,
        help="Enable memory profiling for tests",
    )
    parser.addoption(
        "--memory-threshold",
        action="store",
        default=0.5,
        type=float,
        help="Memory usage threshold for warnings (in GB)",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on memory options."""
    skip_memory = config.getoption("--skip-memory-intensive")

    if skip_memory:
        skip_marker = pytest.mark.skip(reason="Skipped memory-intensive test")
        for item in items:
            if "memory_intensive" in item.keywords:
                item.add_marker(skip_marker)
