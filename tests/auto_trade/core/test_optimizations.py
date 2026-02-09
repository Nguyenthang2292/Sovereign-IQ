"""Tests for optimization modules."""

import time

import pytest

from modules.auto_trade.legacy.caching import Cache
from modules.auto_trade.core.circuit_breaker import CircuitBreaker, CircuitState
from modules.auto_trade.core.health import HealthRegistry, HealthStatus


class TestCache:
    def test_cache_set_get(self):
        cache = Cache()
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_cache_expiry(self):
        cache = Cache()
        cache.set("key", "value", ttl=0.1)  # type: ignore[arg-type]
        time.sleep(0.2)
        assert cache.get("key") is None

    def test_cache_cleanup(self):
        cache = Cache()
        cache.set("key", "value", ttl=0.1)  # type: ignore[arg-type]
        time.sleep(0.2)
        cache.cleanup()
        assert "key" not in cache._cache


class TestCircuitBreaker:
    def test_circuit_state_transitions(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Should be CLOSED initially
        assert cb.state == CircuitState.CLOSED

        # Test success
        def success_func():
            return "ok"

        assert cb.call(success_func) == "ok"

        # Test failure
        def fail_func():
            raise ValueError("fail")

        try:
            cb.call(fail_func)
        except ValueError:
            pass
        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED

        try:
            cb.call(fail_func)
        except ValueError:
            pass
        assert cb.state == CircuitState.OPEN

        # Test blocking
        with pytest.raises(Exception, match=r"Circuit .* is OPEN"):
            cb.call(success_func)

        # Wait for recovery (recovery_timeout=1, so wait 1.2s)
        time.sleep(1.2)

        # Should transition to HALF_OPEN on next call
        assert cb.call(success_func) == "ok"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestHealthRegistry:
    def test_health_check(self):
        registry = HealthRegistry()

        def healthy_check():
            return HealthStatus.HEALTHY, "All good"

        def unhealthy_check():
            return HealthStatus.UNHEALTHY, "Bad"

        registry.register_check("test_ok", healthy_check)
        assert registry.is_healthy() is True

        registry.register_check("test_bad", unhealthy_check)
        assert registry.is_healthy() is False

        results = registry.check_health()
        assert results["test_ok"]["status"] == "HEALTHY"
        assert results["test_bad"]["status"] == "UNHEALTHY"
