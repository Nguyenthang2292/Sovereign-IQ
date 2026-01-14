from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


# Heavy data fixture (function scope) for integration tests
@pytest.fixture
def heavy_position_data():
    n = 5000
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.rand(n) * 1000,
        },
        index=dates,
    )
    yield df
    del df


# Small data fixture (function scope) for fast tests
@pytest.fixture
def small_position_data():
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.rand(n) * 1000,
        },
        index=dates,
    )
    return df


# Mock data fetcher using heavy_position_data by default
@pytest.fixture
def mock_data_fetcher(heavy_position_data):
    fetch_calls = []

    def fake_fetch_ohlcv_with_fallback_exchange(symbol, limit=2160, **kwargs):
        df = heavy_position_data.head(min(limit, len(heavy_position_data)))
        fetch_calls.append({"symbol": symbol, "kwargs": kwargs})
        return df, "binance"

    fetcher = SimpleNamespace()
    fetcher.fetch_ohlcv_with_fallback_exchange = fake_fetch_ohlcv_with_fallback_exchange
    fetcher.fetch_calls = fetch_calls
    return fetcher


# Optional: mock IO to avoid real file/network IO
@pytest.fixture
def mock_io(monkeypatch):
    def _fake_load_csv(*args, **kwargs):
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        return df

    monkeypatch.setattr("position_sizing.io.load_csv", _fake_load_csv, raising=False)
    return True


# Global mock for signal calculators to speed up tests
@pytest.fixture(autouse=True)
def mock_signals(monkeypatch):
    monkeypatch.setattr("core.signal_calculators.get_range_oscillator_signal", lambda *a, **k: (1, 0.7))
    monkeypatch.setattr("core.signal_calculators.get_spc_signal", lambda *a, **k: (1, 0.6))
    monkeypatch.setattr("core.signal_calculators.get_xgboost_signal", lambda *a, **k: (1, 0.8))
    monkeypatch.setattr("core.signal_calculators.get_hmm_signal", lambda *a, **k: (1, 0.65))
    monkeypatch.setattr("core.signal_calculators.get_random_forest_signal", lambda *a, **k: (1, 0.75))


# Optional: mock regime detector to avoid stateful logic
@pytest.fixture
def mock_regime_detector():
    return None
