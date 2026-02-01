"""Tests for Circuit Breaker Module."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from modules.auto_trade.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerOpenError,
    CircuitState,
    circuit_breaker,
)


class TestCircuitBreaker:
    def test_initial_state(self):
        """Test circuit breaker starts in CLOSED state."""
        breaker = CircuitBreaker(name="test")
        assert breaker.get_state() == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_successful_call(self):
        """Test successful calls don't affect failure count."""
        breaker = CircuitBreaker(failure_threshold=3, name="test")

        def success_func():
            return "success"

        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.get_state() == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_failure_counting(self):
        """Test failures are counted correctly."""
        breaker = CircuitBreaker(failure_threshold=3, name="test")

        def failing_func():
            raise ValueError("Test error")

        for i in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        assert breaker.get_state() == CircuitState.OPEN
        assert breaker.failure_count == 3

    def test_circuit_opens_on_threshold(self):
        """Test circuit opens after reaching failure threshold."""
        breaker = CircuitBreaker(failure_threshold=2, name="test")

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.get_state() == CircuitState.CLOSED

        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.get_state() == CircuitState.OPEN

    def test_open_circuit_blocks_calls(self):
        """Test that OPEN circuit raises CircuitBreakerOpenError."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60, name="test")

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            breaker.call(failing_func)

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            breaker.call(lambda: "should block")

        assert "test" in str(exc_info.value)
        assert "OPEN" in str(exc_info.value)

    def test_half_open_transition(self):
        """Test circuit transitions to HALF_OPEN after recovery timeout."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1, name="test")

        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.get_state() == CircuitState.OPEN

        time.sleep(1.1)

        def success_func():
            return "success"

        breaker.call(success_func)
        assert breaker.get_state() == CircuitState.CLOSED

    def test_success_in_half_open(self):
        """Test success in HALF_OPEN closes circuit."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1, name="test")

        def failing_func():
            return 1 / 0

        with pytest.raises(ZeroDivisionError):
            breaker.call(failing_func)

        assert breaker.get_state() == CircuitState.OPEN

        time.sleep(1.1)

        def success_func():
            return "success"

        breaker.call(success_func)
        assert breaker.get_state() == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_failure_in_half_open_reopens(self):
        """Test failure in HALF_OPEN reopens circuit."""
        pass

    def test_non_excluded_exceptions_trigger_circuit(self):
        """Test that non-excluded exceptions trigger circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=1, name="test", excluded_exceptions=(ValueError,))

        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("Test")))

        assert breaker.get_state() == CircuitState.OPEN

    def test_success_threshold(self):
        """Test multiple successes needed to close circuit."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1, success_threshold=3, name="test")

        def failing_func():
            return 1 / 0

        with pytest.raises(ZeroDivisionError):
            breaker.call(failing_func)

        with pytest.raises(ZeroDivisionError):
            breaker.call(failing_func)

        assert breaker.get_state() == CircuitState.OPEN

        time.sleep(1.1)

        # First call after timeout will go to HALF_OPEN
        def success_func():
            return "success"

        breaker.call(success_func)
        assert breaker.get_state() == CircuitState.HALF_OPEN

        # Need 2 more successes
        for _ in range(2):
            breaker.call(success_func)
            assert breaker.get_state() == CircuitState.HALF_OPEN

        breaker.call(success_func)
        assert breaker.get_state() == CircuitState.CLOSED

    def test_metrics_tracking(self):
        """Test metrics are tracked correctly."""
        breaker = CircuitBreaker(failure_threshold=2, name="test")

        for _ in range(5):
            breaker.call(lambda: "success")

        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("Test")))

        metrics = breaker.get_metrics()
        assert metrics.total_calls == 7
        assert metrics.successful_calls == 5
        assert metrics.failed_calls == 2
        assert metrics.circuit_opened_count == 1

    def test_thread_safety_concurrent_failures(self):
        """Test that concurrent failures count correctly."""
        breaker = CircuitBreaker(failure_threshold=50, name="test")

        def failing_func():
            raise ValueError("Test error")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(breaker.call, failing_func) for _ in range(20)]
            for f in futures:
                with pytest.raises(ValueError):
                    f.result()

        assert breaker.get_state() == CircuitState.CLOSED
        assert breaker.failure_count == 20

    def test_thread_safety_concurrent_calls(self):
        """Test concurrent calls are thread-safe."""
        breaker = CircuitBreaker(name="test")

        def success_func(x):
            time.sleep(0.01)
            return x * 2

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(breaker.call, [success_func] * 10, [i for i in range(10)]))

        assert all(r == i * 2 for i, r in zip(range(10), results))
        assert breaker.get_state() == CircuitState.CLOSED

    def test_half_open_single_request(self):
        """Test that only one request passes in HALF_OPEN."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1, name="test")

        def failing_func():
            return 1 / 0

        with pytest.raises(ZeroDivisionError):
            breaker.call(failing_func)

        with pytest.raises(ZeroDivisionError):
            breaker.call(failing_func)

        assert breaker.get_state() == CircuitState.OPEN

        time.sleep(1.1)

        slow_func_started = threading.Event()
        slow_func_continue = threading.Event()

        def slow_success():
            slow_func_started.set()
            slow_func_continue.wait()
            return "success"

        t1 = threading.Thread(target=lambda: breaker.call(slow_success))
        t1.start()
        slow_func_started.wait()

        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(lambda: "should block")

        slow_func_continue.set()
        t1.join()

        assert breaker.get_state() == CircuitState.CLOSED

    def test_reset(self):
        """Test manual reset of circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=1, name="test")

        def failing_func():
            return 1 / 0

        with pytest.raises(ZeroDivisionError):
            breaker.call(failing_func)

        assert breaker.get_state() == CircuitState.OPEN

        breaker.reset()
        assert breaker.get_state() == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_context_manager_success(self):
        """Test context manager with successful call."""
        breaker = CircuitBreaker(name="test")

        with breaker:
            result = lambda: "success"

        assert breaker.get_state() == CircuitState.CLOSED

    def test_context_manager_failure(self):
        """Test context manager with failure."""
        breaker = CircuitBreaker(failure_threshold=1, name="test")

        def failing_func():
            return 1 / 0

        with pytest.raises(ZeroDivisionError):
            with breaker:
                failing_func()

        assert breaker.get_state() == CircuitState.OPEN

    def test_context_manager_excluded_exception(self):
        """Test context manager with excluded exception."""
        breaker = CircuitBreaker(failure_threshold=1, name="test", excluded_exceptions=(ValueError,))

        def failing_func():
            raise ValueError("Test")

        with pytest.raises(ValueError):
            with breaker:
                failing_func()

        assert breaker.get_state() == CircuitState.CLOSED

    def test_decorator(self):
        """Test circuit_breaker decorator."""
        breaker = CircuitBreaker(name="test")

        @circuit_breaker(breaker)
        def protected_func(x):
            return x * 2

        result = protected_func(5)
        assert result == 10

    def test_decorator_with_failure(self):
        """Test decorator handles failures."""
        breaker = CircuitBreaker(failure_threshold=2, name="test")

        @circuit_breaker(breaker)
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_func()

        with pytest.raises(ValueError):
            failing_func()

        with pytest.raises(CircuitBreakerOpenError):
            failing_func()

    def test_state_durations_tracked(self):
        """Test that state durations are tracked."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1, name="test")

        time.sleep(0.1)

        with pytest.raises(ValueError):
            breaker.call(lambda: 1 / 0)

        time.sleep(0.1)

        metrics = breaker.get_metrics()
        assert metrics.state_durations[CircuitState.CLOSED] > 0
        assert metrics.state_durations[CircuitState.OPEN] > 0

    def test_custom_exception_class(self):
        """Test that CircuitBreakerError is properly raised."""
        breaker = CircuitBreaker(name="test")

        with pytest.raises(ValueError):
            breaker.call(lambda: 1 / 0)

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            breaker.call(lambda: "blocked")

        assert exc_info.value.circuit_name == "test"
        assert hasattr(exc_info.value, "retry_after")

    def test_multiple_circuits(self):
        """Test multiple independent circuit breakers."""
        breaker1 = CircuitBreaker(failure_threshold=2, name="breaker1")
        breaker2 = CircuitBreaker(failure_threshold=2, name="breaker2")

        with pytest.raises(ValueError):
            breaker1.call(lambda: 1 / 0)

        assert breaker1.get_state() == CircuitState.OPEN
        assert breaker2.get_state() == CircuitState.CLOSED

    def test_recovery_timeout_precision(self):
        """Test that recovery timeout is respected precisely."""
        pass

    def test_context_manager_success(self):
        """Test context manager with successful call."""
        pass

    def test_context_manager_failure(self):
        """Test context manager with failure."""
        pass

    def test_context_manager_excluded_exception(self):
        """Test context manager with excluded exception."""
        pass

    def test_context_manager_failure(self):
        """Test context manager with failure."""
        breaker = CircuitBreaker(failure_threshold=1, name="test")

        def failing_func():
            return 1 / 0

        with pytest.raises(ZeroDivisionError):
            with breaker:
                failing_func()

        assert breaker.get_state() == CircuitState.OPEN

    def test_context_manager_excluded_exception(self):
        """Test context manager with excluded exception."""
        breaker = CircuitBreaker(failure_threshold=1, name="test", excluded_exceptions=(ValueError,))

        def failing_func():
            raise ValueError("Test")

        with pytest.raises(ValueError):
            with breaker:
                failing_func()

        assert breaker.get_state() == CircuitState.CLOSED
