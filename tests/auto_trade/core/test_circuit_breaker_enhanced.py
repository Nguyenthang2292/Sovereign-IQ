"""Tests for Circuit Breaker enhancements."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from modules.auto_trade.core.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreakerEnhancements:
    def test_alerting_callbacks(self):
        """Test on_open and on_close callbacks."""
        mock_on_open = MagicMock()
        mock_on_close = MagicMock()

        breaker = CircuitBreaker(
            failure_threshold=1, recovery_timeout=1, on_open=mock_on_open, on_close=mock_on_close, name="alert_test"
        )

        # Trigger open
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("Fail")))

        mock_on_open.assert_called_once_with("alert_test")
        mock_on_close.assert_not_called()

        # Reset to trigger close
        breaker.reset()
        mock_on_close.assert_called_once_with("alert_test")

    @patch("modules.auto_trade.core.circuit_breaker.log_warn")
    def test_sanitize_errors(self, mock_log):
        """Test error message sanitization."""
        breaker = CircuitBreaker(failure_threshold=5, sanitize_errors=True, name="sanitize_test")

        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("SENSITIVE_DATA")))

        mock_log.assert_called_once()
        args, _ = mock_log.call_args
        log_msg = args[0]

        assert "SENSITIVE_DATA" not in log_msg
        assert "Circuit sanitize_test failure" in log_msg

    @patch("modules.auto_trade.core.circuit_breaker.log_warn")
    def test_unsanitized_errors(self, mock_log):
        """Test that errors are logged fully when sanitization is off."""
        breaker = CircuitBreaker(
            failure_threshold=5,
            sanitize_errors=False,  # Default
            name="unsanitize_test",
        )

        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("PUBLIC_DATA")))

        mock_log.assert_called_once()
        args, _ = mock_log.call_args
        log_msg = args[0]

        assert "PUBLIC_DATA" in log_msg

    def test_failure_rate_metric(self):
        """Test failure rate calculation."""
        breaker = CircuitBreaker(failure_threshold=10)

        # 5 successes
        for _ in range(5):
            breaker.call(lambda: "success")

        # 5 failures
        for _ in range(5):
            try:
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("Fail")))
            except ValueError:
                pass

        assert breaker.metrics.total_calls == 10
        assert breaker.metrics.failed_calls == 5
        assert breaker.failure_rate == 0.5

        # 1 more failure
        try:
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("Fail")))
        except ValueError:
            pass

        assert breaker.metrics.total_calls == 11
        assert breaker.failure_rate == 6 / 11
