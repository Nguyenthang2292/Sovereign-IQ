"""
Circuit Breaker Module.

Implements the Circuit Breaker pattern to prevent cascading failures
when external services (APIs) are down or unstable.
Features:
- Thread-safe circuit breaking
- Custom exception hierarchy
- Comprehensive metrics tracking
- Configurable success/failure thresholds
- Context manager support
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from threading import RLock
from types import TracebackType
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar

from modules.common.ui.logging import log_error, log_info, log_warn

F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""

    pass


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when circuit is open and blocking requests."""

    def __init__(self, circuit_name: str, retry_after: float):
        self.circuit_name = circuit_name
        self.retry_after = retry_after
        super().__init__(f"Circuit '{circuit_name}' is OPEN. Retry after {retry_after:.1f}s.")


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker observability."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    circuit_opened_count: int = 0
    state_durations: Dict[CircuitState, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.state_durations:
            self.state_durations = {state: 0.0 for state in CircuitState}


class CircuitBreaker:
    """
    Thread-safe Circuit Breaker implementation with comprehensive metrics.

    The circuit breaker monitors failures to external services and opens
    the circuit when the failure threshold is reached, preventing cascading failures.
    """

    failure_threshold: int
    recovery_timeout: float
    success_threshold: int
    name: str
    excluded_exceptions: Tuple[Type[Exception], ...]
    on_open: Optional[Callable[[str], None]]
    on_close: Optional[Callable[[str], None]]
    sanitize_errors: bool
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: float
    _lock: RLock
    _half_open_request_in_flight: bool
    metrics: CircuitBreakerMetrics
    _state_enter_time: float

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 1,
        name: str = "default",
        excluded_exceptions: Tuple[Type[Exception], ...] = (),
        on_open: Optional[Callable[[str], None]] = None,
        on_close: Optional[Callable[[str], None]] = None,
        sanitize_errors: bool = False,
    ) -> None:
        """
        Initialize Circuit Breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening circuit (minimum 1).
            recovery_timeout: Seconds to wait in OPEN state before attempting HALF_OPEN (minimum 1).
            success_threshold: Number of consecutive successes needed in HALF_OPEN to close circuit (minimum 1).
            name: Name of the circuit breaker for logging and metrics.
            excluded_exceptions: Exceptions that should not count as failures.
            on_open: Optional callback triggered when circuit opens (receives name).
            on_close: Optional callback triggered when circuit closes (receives name).
            sanitize_errors: If True, log generic error messages instead of exceptions to prevent data leaks.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if recovery_timeout < 1:
            raise ValueError("recovery_timeout must be at least 1")
        if success_threshold < 1:
            raise ValueError("success_threshold must be at least 1")
        if not name or not name.strip():
            raise ValueError("name cannot be empty")

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.name = name.strip()
        self.excluded_exceptions = excluded_exceptions
        self.on_open = on_open
        self.on_close = on_close
        self.sanitize_errors = sanitize_errors

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self._lock = RLock()
        self._half_open_request_in_flight = False

        self.metrics = CircuitBreakerMetrics()
        self._state_enter_time = time.time()

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute with circuit breaking.
            *args: Positional arguments to pass to function.
            **kwargs: Keyword arguments to pass to function.

        Returns:
            Result of function execution.

        Raises:
            CircuitBreakerOpenError: If circuit is OPEN.
            Exception: If function raises an exception (re-raised).
        """
        self.metrics.total_calls += 1

        with self._lock:
            if self.state == CircuitState.OPEN:
                time_since_failure = time.time() - self.last_failure_time
                if time_since_failure > self.recovery_timeout:
                    self._transition_to_half_open()
                else:
                    retry_after = max(0, self.recovery_timeout - time_since_failure)
                    raise CircuitBreakerOpenError(self.name, retry_after)

            if self.state == CircuitState.HALF_OPEN:
                if self._half_open_request_in_flight:
                    retry_after = self.recovery_timeout - (time.time() - self.last_failure_time)
                    raise CircuitBreakerOpenError(self.name, retry_after)
                self._half_open_request_in_flight = True

        try:
            result = func(*args, **kwargs)
            with self._lock:
                self.metrics.successful_calls += 1

                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    # Request completed, clear in-flight flag
                    self._half_open_request_in_flight = False

                    if self.success_count >= self.success_threshold:
                        self._transition_to_closed()
                        self.success_count = 0
                else:
                    self._reset_failure_count()
            return result

        except self.excluded_exceptions:
            with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    self._half_open_request_in_flight = False
            raise

        except Exception as e:
            with self._lock:
                # print(f"DEBUG: Caught exception {type(e)} in call")
                self.metrics.failed_calls += 1

                if self.state == CircuitState.HALF_OPEN:
                    self._half_open_request_in_flight = False
                    self.success_count = 0

                self._handle_failure(e)
            raise

    def _handle_failure(self, error: Exception) -> None:
        """
        Handle a function failure.

        Caller must hold lock.

        Args:
            error: Exception that was raised.
        """
        self.failure_count += 1
        self.last_failure_time = time.time()
        # print(f"DEBUG: handle_failure count={self.failure_count} thresh={self.failure_threshold} state={self.state}")

        if self.sanitize_errors:
            msg = f"Circuit {self.name} failure ({self.failure_count}/{self.failure_threshold})"
        else:
            msg = f"Circuit {self.name} failure ({self.failure_count}/{self.failure_threshold}): {error}"

        log_warn(msg)

        if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
            self._transition_to_open()

    def _transition_to_open(self) -> None:
        """Transition to OPEN state. Caller must hold lock."""
        if self.state != CircuitState.OPEN:
            self._update_state_duration()
            self.state = CircuitState.OPEN
            self.metrics.circuit_opened_count += 1
            self.success_count = 0
            log_error(f"Circuit {self.name} opened! Blocking calls for {self.recovery_timeout}s.")

            # Fire callback if configured
            if self.on_open:
                try:
                    self.on_open(self.name)
                except Exception as e:
                    log_error(f"Error in on_open callback for {self.name}: {e}")

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state. Caller must hold lock."""
        if self.state != CircuitState.HALF_OPEN:
            self._update_state_duration()
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            log_info(f"Circuit {self.name} half-open. Testing service...")

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state. Caller must hold lock."""
        if self.state != CircuitState.CLOSED:
            self._update_state_duration()
            self.state = CircuitState.CLOSED
            self._reset_failure_count()
            self.success_count = 0
            log_info(f"Circuit {self.name} closed. Service recovered.")

            # Fire callback if configured
            if self.on_close:
                try:
                    self.on_close(self.name)
                except Exception as e:
                    log_error(f"Error in on_close callback for {self.name}: {e}")

    def _reset_failure_count(self) -> None:
        """Reset failure count. Caller must hold lock."""
        self.failure_count = 0

    def _update_state_duration(self) -> None:
        """Update duration spent in current state. Caller must hold lock."""
        duration = time.time() - self._state_enter_time
        self.metrics.state_durations[self.state] += duration
        self._state_enter_time = time.time()

    def get_state(self) -> CircuitState:
        """
        Get current circuit state thread-safely.

        Returns:
            Current CircuitState.
        """
        with self._lock:
            return self.state

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate (0.0 to 1.0)."""
        with self._lock:
            total = self.metrics.total_calls
            if total == 0:
                return 0.0
            return self.metrics.failed_calls / total

    def get_metrics(self) -> CircuitBreakerMetrics:
        """
        Get current metrics snapshot thread-safely.

        Returns:
            CircuitBreakerMetrics copy.
        """
        with self._lock:
            self._update_state_duration()
            return CircuitBreakerMetrics(
                total_calls=self.metrics.total_calls,
                successful_calls=self.metrics.successful_calls,
                failed_calls=self.metrics.failed_calls,
                circuit_opened_count=self.metrics.circuit_opened_count,
                state_durations=dict(self.metrics.state_durations),
            )

    def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state thread-safely."""
        with self._lock:
            self._transition_to_closed()
            self._half_open_request_in_flight = False
            self._reset_failure_count()
            log_info(f"Circuit {self.name} manually reset.")

    def __enter__(self) -> "CircuitBreaker":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        _exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        _exc_tb: Optional[TracebackType],
    ) -> None:
        """
        Context manager exit.

        Args:
            _exc_type: Exception type if exception was raised.
            exc_val: Exception value if exception was raised.
            _exc_tb: Exception traceback if exception was raised.
        """
        if exc_val is not None and not isinstance(exc_val, self.excluded_exceptions):
            with self._lock:
                self.metrics.failed_calls += 1
                if isinstance(exc_val, Exception):
                    self._handle_failure(exc_val)
                else:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    if self.failure_count >= self.failure_threshold:
                        self._transition_to_open()


def circuit_breaker(breaker: CircuitBreaker) -> Callable[[F], F]:
    """
    Decorator for easy use of CircuitBreaker.

    Args:
        breaker: CircuitBreaker instance to use.

    Returns:
        Decorator function.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return breaker.call(func, *args, **kwargs)

        return wrapper  # type: ignore

    return decorator
