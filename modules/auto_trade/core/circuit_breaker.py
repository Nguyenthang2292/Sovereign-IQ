"""
Circuit Breaker Module.

Implements the Circuit Breaker pattern to prevent cascading failures
when external services (APIs) are down or unstable.
"""

import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional

from modules.common.ui.logging import log_error, log_info, log_warn


class CircuitState(Enum):
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, block requests
    HALF_OPEN = "HALF_OPEN"  # Testing if service is back


class CircuitBreaker:
    """
    Circuit Breaker implementation.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        name: str = "default",
    ) -> None:
        """
        Args:
            failure_threshold: Number of failures before opening circuit.
            recovery_timeout: Seconds to wait before attempting half-open.
            name: Name of the circuit breaker for logging.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the function with circuit breaker protection.
        """
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self._transition_to_half_open()
            else:
                raise Exception(f"Circuit {self.name} is OPEN. Call blocked.")

        try:
            result = func(*args, **kwargs)
            if self.state == CircuitState.HALF_OPEN:
                self._transition_to_closed()
            else:
                self._reset_failure_count()
            return result
        except Exception as e:
            self._handle_failure(e)
            raise e

    def _handle_failure(self, error: Exception) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        log_warn(f"Circuit {self.name} failure ({self.failure_count}/{self.failure_threshold}): {error}")

        if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
            self._transition_to_open()

    def _transition_to_open(self) -> None:
        self.state = CircuitState.OPEN
        log_error(f"Circuit {self.name} opened! Blocking calls for {self.recovery_timeout}s.")

    def _transition_to_half_open(self) -> None:
        self.state = CircuitState.HALF_OPEN
        log_info(f"Circuit {self.name} half-open. Testing service...")

    def _transition_to_closed(self) -> None:
        self.state = CircuitState.CLOSED
        self._reset_failure_count()
        log_info(f"Circuit {self.name} closed. Service recovered.")

    def _reset_failure_count(self) -> None:
        self.failure_count = 0


def circuit_breaker(breaker: CircuitBreaker):
    """
    Decorator for easy use of CircuitBreaker.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator
