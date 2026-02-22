import pytest
import time
import threading
from unittest.mock import MagicMock

from modules.auto_trade.core.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerOpenError,
    circuit_breaker,
)


def test_circuit_breaker_closed_to_open():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

    # 2 failures, still closed
    for _ in range(2):
        with pytest.raises(ValueError):
            cb.call(lambda: int("bad"))
    assert cb.state == CircuitState.CLOSED

    # 3rd failure opens circuit
    with pytest.raises(ValueError):
        cb.call(lambda: int("bad"))
    assert cb.state == CircuitState.OPEN


def test_circuit_breaker_open_to_half_open_to_closed():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1, success_threshold=2)

    for _ in range(2):
        with pytest.raises(ValueError):
            cb.call(lambda: int("bad"))

    assert cb.state == CircuitState.OPEN

    # Still open before timeout
    with pytest.raises(CircuitBreakerOpenError):
        cb.call(lambda: 1)

    # Wait for timeout (1 second + small padding)
    time.sleep(1.1)

    # First success puts it in HALF_OPEN
    assert cb.call(lambda: 1) == 1
    assert cb.state == CircuitState.HALF_OPEN

    # Second success closes it
    assert cb.call(lambda: 2) == 2
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_thread_safety():
    cb = CircuitBreaker(failure_threshold=100, recovery_timeout=1)

    def worker():
        try:
            cb.call(lambda: int("bad"))
        except ValueError:
            pass

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert cb.metrics.failed_calls == 50
    assert cb.failure_count == 50
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_decorator():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)

    @circuit_breaker(cb)
    def my_func(fail=False):
        if fail:
            raise ValueError("fail")
        return "ok"

    assert my_func() == "ok"

    with pytest.raises(ValueError):
        my_func(fail=True)

    assert cb.state == CircuitState.OPEN

    with pytest.raises(CircuitBreakerOpenError):
        my_func()


def test_circuit_breaker_context_manager():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)

    with pytest.raises(ValueError):
        with cb:
            raise ValueError("fail")

    assert cb.state == CircuitState.OPEN

    with pytest.raises(CircuitBreakerOpenError):
        with cb:
            if cb.state == CircuitState.OPEN:
                cb.call(lambda: 1)


def test_circuit_breaker_sanitize_errors():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1, sanitize_errors=True)
    with pytest.raises(ValueError):
        cb.call(lambda: int("bad"))
    assert cb.state == CircuitState.OPEN
