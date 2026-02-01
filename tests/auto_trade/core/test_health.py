"""Tests for Health Check Module."""

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from modules.auto_trade.core.health import HealthRegistry, HealthStatus


class TestHealthRegistry:
    def test_empty_registry(self):
        """Test behavior with empty registry."""
        registry = HealthRegistry()

        assert registry.check_health() == {}
        assert registry.is_healthy() is True
        assert registry.list_checks() == []

    def test_register_and_check_single(self):
        """Test registering and checking a single health check."""
        registry = HealthRegistry()

        def my_check():
            return HealthStatus.HEALTHY, "All systems operational"

        registry.register_check("database", my_check)

        results = registry.check_health()
        assert len(results) == 1
        assert results["database"]["status"] == "HEALTHY"
        assert results["database"]["details"] == "All systems operational"
        assert "timestamp" in results["database"]

    def test_multiple_checks(self):
        """Test multiple health checks with different statuses."""
        registry = HealthRegistry()

        registry.register_check("database", lambda: (HealthStatus.HEALTHY, "OK"))
        registry.register_check("cache", lambda: (HealthStatus.DEGRADED, "Slow response"))
        registry.register_check("api", lambda: (HealthStatus.HEALTHY, "OK"))

        results = registry.check_health()
        assert len(results) == 3
        assert results["database"]["status"] == "HEALTHY"
        assert results["cache"]["status"] == "DEGRADED"
        assert results["api"]["status"] == "HEALTHY"

    def test_unhealthy_check(self):
        """Test that unhealthy check is detected."""
        registry = HealthRegistry()

        registry.register_check("failing_service", lambda: (HealthStatus.UNHEALTHY, "Service down"))

        assert registry.is_healthy() is False

        results = registry.check_health()
        assert results["failing_service"]["status"] == "UNHEALTHY"

    def test_exception_handling(self):
        """Test that exceptions in check functions are handled gracefully."""
        registry = HealthRegistry()

        def failing_check():
            raise ValueError("Something went wrong")

        registry.register_check("broken", failing_check)

        results = registry.check_health()
        assert len(results) == 1
        assert results["broken"]["status"] == "UNHEALTHY"
        assert "Check failed:" in results["broken"]["details"]
        assert "Something went wrong" in results["broken"]["details"]

        assert registry.is_healthy() is False

    def test_check_single(self):
        """Test checking a single health check."""
        registry = HealthRegistry()

        registry.register_check("database", lambda: (HealthStatus.HEALTHY, "OK"))
        registry.register_check("cache", lambda: (HealthStatus.DEGRADED, "Slow"))

        result = registry.check_single("database")
        assert result["status"] == "HEALTHY"
        assert result["details"] == "OK"
        assert "timestamp" in result

    def test_check_single_not_found(self):
        """Test that check_single raises KeyError for unknown check."""
        registry = HealthRegistry()

        with pytest.raises(KeyError, match="Health check 'unknown' not found"):
            registry.check_single("unknown")

    def test_unregister_check(self):
        """Test unregistering a health check."""
        registry = HealthRegistry()

        registry.register_check("database", lambda: (HealthStatus.HEALTHY, "OK"))
        registry.register_check("cache", lambda: (HealthStatus.HEALTHY, "OK"))

        assert len(registry.list_checks()) == 2

        registry.unregister_check("cache")

        assert len(registry.list_checks()) == 1
        assert "database" in registry.list_checks()
        assert "cache" not in registry.list_checks()

        results = registry.check_health()
        assert "database" in results
        assert "cache" not in results

    def test_unregister_nonexistent_check(self):
        """Test that unregistering non-existent check doesn't raise error."""
        registry = HealthRegistry()

        registry.register_check("database", lambda: (HealthStatus.HEALTHY, "OK"))

        registry.unregister_check("nonexistent")

        assert len(registry.list_checks()) == 1

    def test_list_checks(self):
        """Test listing all registered checks."""
        registry = HealthRegistry()

        registry.register_check("check1", lambda: (HealthStatus.HEALTHY, "OK"))
        registry.register_check("check2", lambda: (HealthStatus.HEALTHY, "OK"))
        registry.register_check("check3", lambda: (HealthStatus.HEALTHY, "OK"))

        checks = registry.list_checks()
        assert len(checks) == 3
        assert "check1" in checks
        assert "check2" in checks
        assert "check3" in checks

    def test_check_timeout(self):
        """Test that checks timeout correctly."""
        registry = HealthRegistry(default_timeout=0.1)

        def slow_check():
            time.sleep(5)
            return HealthStatus.HEALTHY, "Should timeout"

        registry.register_check("slow", slow_check)

        results = registry.check_health()
        assert results["slow"]["status"] == "UNHEALTHY"
        assert "timed out" in results["slow"]["details"]
        assert "0.1s" in results["slow"]["details"]

    def test_check_timeout_override(self):
        """Test that timeout can be overridden per call."""
        registry = HealthRegistry(default_timeout=0.1)

        def slow_check():
            time.sleep(0.05)
            return HealthStatus.HEALTHY, "OK"

        registry.register_check("slow", slow_check)

        results = registry.check_health(timeout=1.0)
        assert results["slow"]["status"] == "HEALTHY"
        assert results["slow"]["details"] == "OK"

    def test_check_single_timeout(self):
        """Test that check_single does not timeout."""
        registry = HealthRegistry(default_timeout=0.1)

        def slow_check():
            time.sleep(0.05)
            return HealthStatus.HEALTHY, "OK"

        registry.register_check("slow", slow_check)

        result = registry.check_single("slow")
        assert result["status"] == "HEALTHY"

    def test_is_healthy_optimization(self):
        """Test that is_healthy short-circuits on first failure."""
        registry = HealthRegistry()

        check_order = []

        def check1():
            check_order.append(1)
            return HealthStatus.HEALTHY, "OK"

        def check2():
            check_order.append(2)
            return HealthStatus.UNHEALTHY, "Failed"

        def check3():
            check_order.append(3)
            return HealthStatus.HEALTHY, "OK"

        registry.register_check("check1", check1)
        registry.register_check("check2", check2)
        registry.register_check("check3", check3)

        is_healthy = registry.is_healthy()

        assert is_healthy is False
        assert check_order == [1, 2]
        assert 3 not in check_order

    def test_concurrent_registration(self):
        """Test concurrent registration is thread-safe."""
        registry = HealthRegistry()

        def register_check(i):
            registry.register_check(f"check_{i}", lambda: (HealthStatus.HEALTHY, "OK"))

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(register_check, range(100)))

        assert len(registry.list_checks()) == 100
        assert registry.is_healthy() is True

    def test_concurrent_unregistration(self):
        """Test concurrent unregistration is thread-safe."""
        registry = HealthRegistry()

        for i in range(50):
            registry.register_check(f"check_{i}", lambda: (HealthStatus.HEALTHY, "OK"))

        def unregister_check(i):
            registry.unregister_check(f"check_{i}")

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(unregister_check, range(50)))

        assert len(registry.list_checks()) == 0

    def test_concurrent_health_checks(self):
        """Test concurrent health check execution."""
        registry = HealthRegistry()

        for i in range(20):
            registry.register_check(f"check_{i}", lambda: (HealthStatus.HEALTHY, "OK"))

        results_list = []

        def run_check():
            results_list.append(registry.check_health())

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(lambda _: run_check(), range(10)))

        for results in results_list:
            assert len(results) == 20
            for result in results.values():
                assert result["status"] == "HEALTHY"

    def test_mixed_statuses_is_healthy(self):
        """Test is_healthy with mixed HEALTHY and DEGRADED statuses."""
        registry = HealthRegistry()

        registry.register_check("check1", lambda: (HealthStatus.HEALTHY, "OK"))
        registry.register_check("check2", lambda: (HealthStatus.DEGRADED, "Warning"))
        registry.register_check("check3", lambda: (HealthStatus.HEALTHY, "OK"))

        assert registry.is_healthy() is True

    def test_type_safety_literal(self):
        """Test that status field uses Literal type."""
        registry = HealthRegistry()

        registry.register_check("test", lambda: (HealthStatus.HEALTHY, "OK"))
        results = registry.check_health()

        assert results["test"]["status"] in ["HEALTHY", "DEGRADED", "UNHEALTHY"]

    def test_logging_on_failure(self):
        """Test that failures are logged."""
        registry = HealthRegistry()

        def failing_check():
            raise ValueError("Test error")

        registry.register_check("failing", failing_check)

        with patch("modules.auto_trade.core.health.logger") as mock_logger:
            registry.check_health()

            assert mock_logger.error.called
            error_call = mock_logger.error.call_args
            assert "failing" in str(error_call)

    def test_logging_on_unhealthy(self):
        """Test that unhealthy checks are logged."""
        registry = HealthRegistry()

        registry.register_check("unhealthy", lambda: (HealthStatus.UNHEALTHY, "Service down"))

        with patch("modules.auto_trade.core.health.logger") as mock_logger:
            registry.check_health()

            assert mock_logger.warning.called
            warning_call = mock_logger.warning.call_args
            assert "unhealthy" in str(warning_call)
            assert "UNHEALTHY" in str(warning_call)

    def test_logging_on_degraded(self):
        """Test that degraded checks are logged at INFO level."""
        registry = HealthRegistry()

        registry.register_check("degraded", lambda: (HealthStatus.DEGRADED, "Slow response"))

        with patch("modules.auto_trade.core.health.logger") as mock_logger:
            registry.check_health()

            assert mock_logger.info.called
            info_call = mock_logger.info.call_args
            assert "degraded" in str(info_call)
            assert "DEGRADED" in str(info_call)
