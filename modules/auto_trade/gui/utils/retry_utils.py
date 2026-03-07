"""
Retry utilities for handling transient network errors
Implements exponential backoff strategy
"""

import functools
import time
from typing import Any, Callable, Optional, Tuple, Type

import ccxt

from modules.common.ui.logging import log_error, log_warn


def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (ccxt.NetworkError, ccxt.RequestTimeout, ConnectionError),
):
    """
    Decorator that retries a function with exponential backoff on specific exceptions

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function that retries on specified exceptions

    Example:
        @retry_with_exponential_backoff(max_retries=3)
        def fetch_data():
            return api.get_data()
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    # Don't retry on last attempt
                    if attempt == max_retries:
                        break

                    # Calculate delay with exponential backoff
                    delay: float = min(base_delay * (backoff_factor**attempt), max_delay)

                    log_warn("Attempt %d/%d failed: %s", attempt + 1, max_retries + 1, str(e))
                    log_warn("Retrying in %.2f seconds...", delay)

                    time.sleep(delay)

            # If we get here, all retries failed
            log_error("All %d attempts failed", max_retries + 1)
            if last_exception is not None:
                raise last_exception
            raise RuntimeError(f"All {max_retries + 1} attempts failed")

        return wrapper

    return decorator


def retry_async_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (ccxt.NetworkError, ccxt.RequestTimeout, ConnectionError),
) -> Callable:
    """
    Async version of retry decorator with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to retry on

    Returns:
        Decorated async function that retries on specified exceptions
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            import asyncio

            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    # Don't retry on last attempt
                    if attempt == max_retries:
                        break

                    # Calculate delay with exponential backoff
                    delay: float = min(base_delay * (backoff_factor**attempt), max_delay)

                    log_warn("Attempt %d/%d failed: %s", attempt + 1, max_retries + 1, str(e))
                    log_warn("Retrying in %.2f seconds...", delay)

                    await asyncio.sleep(delay)

            # If we get here, all retries failed
            log_error("All %d attempts failed", max_retries + 1)
            if last_exception is not None:
                raise last_exception
            raise RuntimeError(f"All {max_retries + 1} attempts failed")

        return wrapper

    return decorator


class RetryableOperation:
    """
    Context manager for retryable operations with exponential backoff

    Example:
        operation = RetryableOperation(max_retries=3)
        for attempt in operation:
            try:
                result = api.fetch_data()
                operation.success()
                break
            except NetworkError as e:
                operation.failed(e)
    """

    def __init__(
        self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0, backoff_factor: float = 2.0
    ) -> None:
        self.max_retries: int = max_retries
        self.base_delay: float = base_delay
        self.max_delay: float = max_delay
        self.backoff_factor: float = backoff_factor
        self.attempt: int = 0
        self._success: bool = False
        self.last_exception: Optional[Exception] = None

    def __iter__(self) -> "RetryableOperation":
        return self

    def __next__(self) -> int:
        if self._success:
            raise StopIteration

        if self.attempt > self.max_retries:
            raise StopIteration

        self.attempt += 1
        return self.attempt

    def success(self) -> None:
        """Mark operation as successful"""
        self._success = True

    def failed(self, exception: Exception) -> None:
        """
        Mark operation as failed and sleep if more retries available

        Args:
            exception: The exception that occurred
        """
        self.last_exception = exception

        if self.attempt < self.max_retries + 1:
            delay: float = min(self.base_delay * (self.backoff_factor ** (self.attempt - 1)), self.max_delay)
            log_warn("Attempt %d/%d failed: %s", self.attempt, self.max_retries + 1, str(exception))
            log_warn("Retrying in %.2f seconds...", delay)
            time.sleep(delay)
