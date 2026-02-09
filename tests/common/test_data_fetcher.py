"""
Unit tests for the DataFetcher class in modules.common.core.data_fetcher.

This module includes helpers and dummy classes to simulate exchange behavior, and tests
data fetching functionalities, including fallback mechanisms, TTL cache, and parallel
exchange fallback, using pytest and pandas.

Classes:
    DummyExchange: Simulates a crypto exchange's OHLCV fetching API with optional error and call tracking.
    DummyPublic: Mocks a public interface that manages exchange connectivity and response priority.

Functions:
    _build_ohlcv: Builds sample OHLCV lists for testing.
"""

import time
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from modules.common.core.data_fetcher import DataFetcher


def _build_ohlcv(last_timestamp_ms: int):
    step = 60_000
    return [
        [last_timestamp_ms - 2 * step, 1, 2, 0.5, 1.5, 10],
        [last_timestamp_ms - step, 1.6, 2.1, 1.0, 1.8, 11],
        [last_timestamp_ms, 1.9, 2.3, 1.7, 2.0, 12],
    ]


class DummyExchange:
    def __init__(self, data, call_tracker=None, exchange_id=None, call_order_list=None, delay_seconds=0):
        self._data = data
        self._call_tracker = call_tracker
        self._exchange_id = exchange_id
        self._call_order_list = call_order_list
        self._delay_seconds = delay_seconds

    def fetch_ohlcv(self, *args, **kwargs):
        if self._call_order_list is not None and self._exchange_id is not None:
            self._call_order_list.append(self._exchange_id)
        if self._delay_seconds > 0:
            time.sleep(self._delay_seconds)
        if isinstance(self._data, Exception):
            raise self._data
        if self._call_tracker is not None:
            self._call_tracker["calls"] += 1
        return self._data


class DummyPublic:
    def __init__(self, priority, responses):
        self.exchange_priority_for_fallback = priority
        self._responses = responses

    def connect_to_exchange_with_no_credentials(self, exchange_id: str):
        response = self._responses.get(exchange_id)
        if response is None:
            raise ValueError(f"Unknown exchange {exchange_id}")
        if isinstance(response, Exception):
            raise response
        data = response if not isinstance(response, tuple) else response[0]
        extra = response[1] if isinstance(response, tuple) and len(response) > 1 else None
        if isinstance(extra, dict) and ("tracker" in extra or "call_order_list" in extra or "delay_seconds" in extra):
            tracker = extra.get("tracker")
            call_order = extra.get("call_order_list")
            delay = extra.get("delay_seconds", 0)
        else:
            # Backward compat: (data, tracker) where tracker is {"calls": 0}
            tracker = extra
            call_order = None
            delay = 0
        return DummyExchange(
            data,
            call_tracker=tracker,
            exchange_id=exchange_id,
            call_order_list=call_order,
            delay_seconds=delay,
        )

    @staticmethod
    def throttled_call(func, *args, **kwargs):
        return func(*args, **kwargs)


def test_fetch_ohlcv_with_fallback_prefers_fresh_data(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    stale_last = int((now - pd.Timedelta(minutes=180)).timestamp() * 1000)
    fresh_last = int((now - pd.Timedelta(minutes=10)).timestamp() * 1000)

    responses = {
        "binance": _build_ohlcv(stale_last),
        "kraken": _build_ohlcv(fresh_last),
    }
    public = DummyPublic(["binance", "kraken"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    df, exchange_id = fetcher.fetch_ohlcv_with_fallback_exchange(
        "eth/usdt", limit=3, timeframe="1h", check_freshness=True
    )

    assert exchange_id == "kraken"
    assert len(df) == 3
    # Timestamp is set as index, not a column (see data_fetcher.py line 612)
    assert df.index[-1] == pd.to_datetime(fresh_last, unit="ms", utc=True)


def test_fetch_ohlcv_uses_cache_when_not_checking_freshness():
    now = pd.Timestamp.now(tz="UTC")
    last_ts = int(now.timestamp() * 1000)

    tracker = {"calls": 0}
    responses = {
        "binance": (_build_ohlcv(last_ts), tracker),
    }
    public = DummyPublic(["binance"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    df1, _ = fetcher.fetch_ohlcv_with_fallback_exchange("btc/usdt", limit=3, timeframe="1h", check_freshness=False)
    df2, _ = fetcher.fetch_ohlcv_with_fallback_exchange("btc/usdt", limit=3, timeframe="1h", check_freshness=False)

    assert tracker["calls"] == 1  # second call served from cache
    assert df1.equals(df2)


def test_fetch_ohlcv_returns_stale_fallback_when_no_fresh_data():
    now = pd.Timestamp.now(tz="UTC")
    stale_last = int((now - pd.Timedelta(minutes=300)).timestamp() * 1000)
    responses = {
        "binance": _build_ohlcv(stale_last),
    }
    public = DummyPublic(["binance"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    df, exchange_id = fetcher.fetch_ohlcv_with_fallback_exchange(
        "ada/usdt", limit=3, timeframe="1h", check_freshness=True
    )

    assert exchange_id == "binance"
    assert df is not None


def test_fetch_ohlcv_returns_dataframe_only():
    """fetch_ohlcv is a convenience wrapper that returns only the DataFrame."""
    now = pd.Timestamp.now(tz="UTC")
    last_ts = int(now.timestamp() * 1000)
    responses = {"binance": _build_ohlcv(last_ts)}
    public = DummyPublic(["binance"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    df = fetcher.fetch_ohlcv("btc/usdt", timeframe="1h", limit=3)

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    df_tuple, _ = fetcher.fetch_ohlcv_with_fallback_exchange("btc/usdt", limit=3, timeframe="1h")
    assert df.equals(df_tuple)


def test_dataframe_to_close_series_converts_dataframe():
    now = pd.Timestamp.now(tz="UTC")
    data = pd.DataFrame(
        {
            "timestamp": [now - pd.Timedelta(minutes=i) for i in range(3)],
            "close": [10.0, 11.0, 12.0],
        }
    )

    series = DataFetcher.dataframe_to_close_series(data)

    assert series is not None
    assert list(series.values) == [10.0, 11.0, 12.0]


# ==================== TTL CACHE TESTS (Bottleneck #3) ====================


def test_fetch_ohlcv_check_freshness_uses_cache_within_ttl():
    """With check_freshness=True, two consecutive calls for same key hit cache on second (no extra exchange call)."""
    now = pd.Timestamp.now(tz="UTC")
    last_ts = int(now.timestamp() * 1000)
    tracker = {"calls": 0}
    responses = {"binance": (_build_ohlcv(last_ts), {"tracker": tracker})}
    public = DummyPublic(["binance"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    fetcher.fetch_ohlcv_with_fallback_exchange(
        "btc/usdt", limit=3, timeframe="1h", check_freshness=True
    )
    assert tracker["calls"] == 1

    fetcher.fetch_ohlcv_with_fallback_exchange(
        "btc/usdt", limit=3, timeframe="1h", check_freshness=True
    )
    assert tracker["calls"] == 1, "Second call within TTL should be served from cache"


def test_fetch_ohlcv_check_freshness_refetches_when_cache_expired():
    """When cache is expired (timestamp old), next fetch with check_freshness=True refetches from exchange."""
    now = pd.Timestamp.now(tz="UTC")
    last_ts = int(now.timestamp() * 1000)
    tracker = {"calls": 0}
    responses = {"binance": (_build_ohlcv(last_ts), {"tracker": tracker})}
    public = DummyPublic(["binance"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    fetcher.fetch_ohlcv_with_fallback_exchange(
        "btc/usdt", limit=3, timeframe="1h", check_freshness=True
    )
    assert tracker["calls"] == 1

    # Simulate expired cache: set timestamp to 2 hours ago (TTL for 1h is 1h = 3600s)
    cache_key = ("BTC/USDT", "1h", 3)
    fetcher._ohlcv_cache_timestamps[cache_key] = time.time() - 7200

    fetcher.fetch_ohlcv_with_fallback_exchange(
        "btc/usdt", limit=3, timeframe="1h", check_freshness=True
    )
    assert tracker["calls"] == 2, "Expired cache should trigger refetch"


def test_fetch_ohlcv_cache_ttl_multiplier_extends_cache_window():
    """cache_ttl_multiplier extends TTL; cache older than 1x TTL but within 2x TTL still hits when multiplier=2."""
    now = pd.Timestamp.now(tz="UTC")
    last_ts = int(now.timestamp() * 1000)
    tracker = {"calls": 0}
    responses = {"binance": (_build_ohlcv(last_ts), {"tracker": tracker})}
    public = DummyPublic(["binance"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    # Populate cache via public API
    fetcher.fetch_ohlcv_with_fallback_exchange(
        "btc/usdt", limit=3, timeframe="1h", check_freshness=True
    )
    assert tracker["calls"] == 1

    # Set cache age to 4000s (between 1h=3600 and 2h=7200)
    cache_key = ("BTC/USDT", "1h", 3)
    fetcher._ohlcv_cache_timestamps[cache_key] = time.time() - 4000

    # Call via _ohlcv with multiplier 2.0 -> TTL = 7200s, age 4000 < 7200 -> cache hit
    df, ex = fetcher._ohlcv.fetch_ohlcv_with_fallback_exchange(
        "btc/usdt", 3, "1h", True, None, cache_ttl_multiplier=2.0
    )
    assert df is not None
    assert tracker["calls"] == 1, "With multiplier 2.0, 4000s age is within TTL"


def test_fetch_ohlcv_ttl_boundary_just_under_ttl_hits_cache():
    """Cache age just under TTL returns cache hit."""
    now = pd.Timestamp.now(tz="UTC")
    last_ts = int(now.timestamp() * 1000)
    tracker = {"calls": 0}
    responses = {"binance": (_build_ohlcv(last_ts), {"tracker": tracker})}
    public = DummyPublic(["binance"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    fetcher.fetch_ohlcv_with_fallback_exchange(
        "btc/usdt", limit=3, timeframe="1h", check_freshness=True
    )
    # Set age to 3599s (just under 3600s TTL for 1h)
    cache_key = ("BTC/USDT", "1h", 3)
    fetcher._ohlcv_cache_timestamps[cache_key] = time.time() - 3599

    fetcher.fetch_ohlcv_with_fallback_exchange(
        "btc/usdt", limit=3, timeframe="1h", check_freshness=True
    )
    assert tracker["calls"] == 1


def test_fetch_ohlcv_ttl_boundary_just_over_ttl_refetches():
    """Cache age just over TTL triggers refetch."""
    now = pd.Timestamp.now(tz="UTC")
    last_ts = int(now.timestamp() * 1000)
    tracker = {"calls": 0}
    responses = {"binance": (_build_ohlcv(last_ts), {"tracker": tracker})}
    public = DummyPublic(["binance"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    fetcher.fetch_ohlcv_with_fallback_exchange(
        "btc/usdt", limit=3, timeframe="1h", check_freshness=True
    )
    cache_key = ("BTC/USDT", "1h", 3)
    fetcher._ohlcv_cache_timestamps[cache_key] = time.time() - 3601

    fetcher.fetch_ohlcv_with_fallback_exchange(
        "btc/usdt", limit=3, timeframe="1h", check_freshness=True
    )
    assert tracker["calls"] == 2


# ==================== PARALLEL EXCHANGE FALLBACK TESTS (Bottleneck #2) ====================


def test_fetch_ohlcv_parallel_probe_first_success_wins():
    """When probing exchanges in parallel, first successful response wins (order may vary)."""
    now = pd.Timestamp.now(tz="UTC")
    fresh_last = int((now - pd.Timedelta(minutes=10)).timestamp() * 1000)
    call_order = []
    # First exchange slow, second fast -> second may complete first
    responses = {
        "binance": (_build_ohlcv(fresh_last), {"call_order_list": call_order, "delay_seconds": 0.15}),
        "kraken": (_build_ohlcv(fresh_last), {"call_order_list": call_order, "delay_seconds": 0}),
        "kucoin": (_build_ohlcv(fresh_last), {"call_order_list": call_order, "delay_seconds": 0.1}),
    }
    public = DummyPublic(["binance", "kraken", "kucoin"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    df, exchange_id = fetcher.fetch_ohlcv_with_fallback_exchange(
        "eth/usdt", limit=3, timeframe="1h", check_freshness=True
    )

    assert df is not None
    assert exchange_id in ("binance", "kraken", "kucoin")
    # Parallel probe: at least 2 exchanges may be invoked (we don't cancel others on first success)
    assert len(call_order) >= 1


def test_fetch_ohlcv_parallel_probe_first_fails_second_succeeds():
    """First exchange fails, second succeeds; result is from second (parallel probe)."""
    now = pd.Timestamp.now(tz="UTC")
    fresh_last = int((now - pd.Timedelta(minutes=5)).timestamp() * 1000)
    responses = {
        "binance": ConnectionError("timeout"),
        "kraken": _build_ohlcv(fresh_last),
        "kucoin": _build_ohlcv(fresh_last),
    }
    public = DummyPublic(["binance", "kraken", "kucoin"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    df, exchange_id = fetcher.fetch_ohlcv_with_fallback_exchange(
        "sol/usdt", limit=3, timeframe="1h", check_freshness=True
    )

    assert df is not None
    assert exchange_id in ("kraken", "kucoin")


def test_fetch_ohlcv_parallel_probe_all_fail_then_sequential_fallback():
    """First 3 (probe) fail, 4th exchange succeeds via sequential fallback."""
    now = pd.Timestamp.now(tz="UTC")
    fresh_last = int((now - pd.Timedelta(minutes=5)).timestamp() * 1000)
    responses = {
        "binance": ConnectionError("timeout"),
        "kraken": ConnectionError("timeout"),
        "kucoin": ConnectionError("timeout"),
        "gate": _build_ohlcv(fresh_last),
    }
    public = DummyPublic(["binance", "kraken", "kucoin", "gate"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    df, exchange_id = fetcher.fetch_ohlcv_with_fallback_exchange(
        "btc/usdt", limit=3, timeframe="1h", check_freshness=True
    )

    assert df is not None
    assert exchange_id == "gate"


def test_fetch_ohlcv_parallel_probe_all_exchanges_fail_returns_none():
    """When all exchanges fail, returns (None, None)."""
    responses = {
        "binance": ConnectionError("timeout"),
        "kraken": ConnectionError("timeout"),
        "kucoin": ValueError("empty"),
    }
    public = DummyPublic(["binance", "kraken", "kucoin"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    df, exchange_id = fetcher.fetch_ohlcv_with_fallback_exchange(
        "btc/usdt", limit=3, timeframe="1h", check_freshness=True
    )

    assert df is None
    assert exchange_id is None
