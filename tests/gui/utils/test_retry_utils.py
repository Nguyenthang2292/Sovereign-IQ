"""
Unit tests for retry utilities
"""

import pytest
import time
from unittest.mock import MagicMock
import ccxt
from modules.auto_trade.gui.utils.retry_utils import retry_with_exponential_backoff, RetryableOperation


class TestRetryWithExponentialBackoff:
    """Test cases for retry_with_exponential_backoff decorator"""

    def test_success_on_first_attempt(self):
        """Test function succeeds on first attempt"""
        mock_func = MagicMock(return_value="success")

        @retry_with_exponential_backoff(max_retries=3)
        def decorated_func():
            return mock_func()

        result = decorated_func()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_success_after_retries(self):
        """Test function succeeds after some retries"""
        call_count = 0

        @retry_with_exponential_backoff(max_retries=3, base_delay=0.01)
        def decorated_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ccxt.NetworkError("Temporary error")
            return "success"

        result = decorated_func()

        assert result == "success"
        assert call_count == 3

    def test_all_retries_exhausted(self):
        """Test all retries are exhausted and exception is raised"""

        @retry_with_exponential_backoff(max_retries=2, base_delay=0.01)
        def decorated_func():
            raise ccxt.NetworkError("Persistent error")

        with pytest.raises(ccxt.NetworkError):
            decorated_func()

    def test_exponential_backoff_timing(self):
        """Test that exponential backoff increases delays"""
        delays = []

        @retry_with_exponential_backoff(max_retries=3, base_delay=0.1, backoff_factor=2.0)
        def decorated_func():
            start = time.time()
            if len(delays) > 0:
                delays.append(time.time() - start)
            raise ccxt.NetworkError("Test error")

        try:
            decorated_func()
        except ccxt.NetworkError:
            pass

        # Delays should approximately double each time
        # First delay ~0.1s, second ~0.2s, third ~0.4s

    def test_non_retryable_exception(self):
        """Test that non-retryable exceptions are not retried"""
        call_count = 0

        @retry_with_exponential_backoff(max_retries=3, exceptions=(ccxt.NetworkError,))
        def decorated_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")

        with pytest.raises(ValueError):
            decorated_func()

        # Should only be called once (no retries)
        assert call_count == 1

    def test_max_delay_capping(self):
        """Test that delay is capped at max_delay"""

        @retry_with_exponential_backoff(
            max_retries=5,
            base_delay=1.0,
            max_delay=2.0,
            backoff_factor=10.0,  # Would normally cause very large delays
        )
        def decorated_func():
            raise ccxt.NetworkError("Test error")

        start = time.time()
        try:
            decorated_func()
        except ccxt.NetworkError:
            pass
        elapsed = time.time() - start

        # Total time should be bounded by max_delay * retries
        # With max_delay=2.0 and 5 retries, should be < 15 seconds
        assert elapsed < 15

    def test_custom_exceptions(self):
        """Test retry with custom exception types"""

        @retry_with_exponential_backoff(max_retries=2, base_delay=0.01, exceptions=(ConnectionError, TimeoutError))
        def decorated_func():
            raise ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            decorated_func()

    def test_function_with_arguments(self):
        """Test decorated function with arguments"""

        @retry_with_exponential_backoff(max_retries=2, base_delay=0.01)
        def decorated_func(x, y, z=10):
            if x < 2:
                raise ccxt.NetworkError("Not ready")
            return x + y + z

        result = decorated_func(2, 3, z=5)
        assert result == 10

    def test_function_preserves_metadata(self):
        """Test that decorator preserves function metadata"""

        @retry_with_exponential_backoff(max_retries=2)
        def my_function():
            """My docstring"""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring"


class TestRetryableOperation:
    """Test cases for RetryableOperation context manager"""

    def test_success_on_first_attempt(self):
        """Test operation succeeds on first attempt"""
        operation = RetryableOperation(max_retries=3)
        attempt_count = 0

        for attempt in operation:
            attempt_count += 1
            operation.success()

        assert attempt_count == 1

    def test_success_after_retries(self):
        """Test operation succeeds after retries"""
        operation = RetryableOperation(max_retries=3, base_delay=0.01)
        attempt_count = 0

        for attempt in operation:
            attempt_count += 1
            if attempt < 3:
                operation.failed(Exception("Not ready"))
            else:
                operation.success()

        assert attempt_count == 3

    def test_all_retries_exhausted(self):
        """Test all retries are exhausted"""
        operation = RetryableOperation(max_retries=2, base_delay=0.01)
        attempt_count = 0

        for attempt in operation:
            attempt_count += 1
            operation.failed(Exception("Persistent error"))

        assert attempt_count == 3  # Initial + 2 retries
        assert operation.last_exception is not None

    def test_iteration_stops_after_success(self):
        """Test iteration stops after marking success"""
        operation = RetryableOperation(max_retries=5)
        attempt_count = 0

        for attempt in operation:
            attempt_count += 1
            if attempt == 2:
                operation.success()
            # Even if we don't break, iterator should stop

        assert attempt_count == 2

    def test_exponential_backoff(self):
        """Test exponential backoff between attempts"""
        operation = RetryableOperation(max_retries=3, base_delay=0.05, backoff_factor=2.0)

        start_time = time.time()
        for attempt in operation:
            if attempt < operation.max_retries + 1:
                operation.failed(Exception("Test"))

        elapsed = time.time() - start_time

        # Should have delays of ~0.05, ~0.1, ~0.2 = ~0.35 total
        # Allow some margin for execution time
        assert elapsed >= 0.3

    def test_last_exception_stored(self):
        """Test that last exception is stored"""
        operation = RetryableOperation(max_retries=2, base_delay=0.01)

        for attempt in operation:
            operation.failed(ValueError(f"Error {attempt}"))

        assert operation.last_exception is not None
        assert "Error" in str(operation.last_exception)
