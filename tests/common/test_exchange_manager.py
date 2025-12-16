import pytest
import ccxt
import time

from modules.common.core.exchange_manager import ExchangeManager, PublicExchangeManager


class DummyExchange:
    def __init__(self):
        self.calls = 0

    def ping(self):
        self.calls += 1
        return "pong"


def test_public_manager_caches_instances(monkeypatch):
    public = PublicExchangeManager()

    class DummyCCXTModule:
        def __init__(self, params=None):
            pass

    # Mock hasattr and getattr for ccxt module
    original_hasattr = hasattr
    original_getattr = getattr
    
    def mock_hasattr(obj, name):
        if obj is ccxt and name == "dummy":
            return True
        return original_hasattr(obj, name)
    
    def mock_getattr(obj, name, default=None):
        if obj is ccxt and name == "dummy":
            return DummyCCXTModule
        return original_getattr(obj, name, default)
    
    monkeypatch.setattr("builtins.hasattr", mock_hasattr)
    monkeypatch.setattr("builtins.getattr", mock_getattr)
    
    ex1 = public.connect_to_exchange_with_no_credentials("dummy")
    ex2 = public.connect_to_exchange_with_no_credentials("dummy")
    
    assert ex1 is ex2

    ex1 = public.connect_to_exchange_with_no_credentials("dummy")
    ex2 = public.connect_to_exchange_with_no_credentials("dummy")

    assert ex1 is ex2


def test_throttled_call_enforces_wait(monkeypatch):
    public = PublicExchangeManager(request_pause=0.01)
    exchange = DummyExchange()

    called = []

    def fake_time():
        return len(called) * 0.02

    monkeypatch.setattr(time, "time", fake_time)
    result = public.throttled_call(exchange.ping)
    called.append(1)
    result2 = public.throttled_call(exchange.ping)

    assert result == "pong" and result2 == "pong"
    assert exchange.calls == 2


def test_exchange_manager_normalizes_symbols():
    manager = ExchangeManager()
    assert manager.normalize_symbol("BTC/USDT:USDT") == "BTC/USDT"


def test_public_manager_rejects_unknown_exchange(monkeypatch):
    public = PublicExchangeManager()

    # Mock getattr to raise AttributeError for unknown exchange
    original_getattr = getattr
    def mock_getattr(obj, name, default=None):
        if obj is ccxt and name == "not_real":
            raise AttributeError(f"module 'ccxt' has no attribute '{name}'")
        return original_getattr(obj, name, default)
    
    monkeypatch.setattr("builtins.getattr", mock_getattr)

    with pytest.raises(ValueError):
        public.connect_to_exchange_with_no_credentials("not_real")
