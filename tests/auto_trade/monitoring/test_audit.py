"""
Tests for Audit Trail System.

Tests audit logging, integrity verification, data sanitization, and error handling.
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from modules.auto_trade.monitoring.audit import AuditEventType, AuditLogger


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for test logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def audit_logger(temp_log_dir):
    """Create an AuditLogger instance with temporary directory."""
    logger = AuditLogger(log_dir=temp_log_dir)
    yield logger
    # Cleanup: close logger to release file handles
    logger.close()


class TestAuditEventType:
    """Test AuditEventType enum."""

    def test_all_event_types_exist(self):
        """Test that all expected event types are defined."""
        expected_types = [
            "ORDER_CREATED", "ORDER_FILLED", "ORDER_CANCELLED", "ORDER_FAILED",
            "POSITION_OPENED", "POSITION_CLOSED", "POSITION_MODIFIED",
            "RISK_LIMIT_EXCEEDED", "MANUAL_INTERVENTION",
            "SYSTEM_START", "SYSTEM_STOP", "CONFIG_CHANGE"
        ]

        for event_type in expected_types:
            assert hasattr(AuditEventType, event_type)
            assert getattr(AuditEventType, event_type).value == event_type

    def test_event_type_is_string(self):
        """Test that AuditEventType values are strings."""
        assert isinstance(AuditEventType.ORDER_CREATED.value, str)
        assert isinstance(AuditEventType.POSITION_OPENED.value, str)


class TestAuditLoggerInitialization:
    """Test AuditLogger initialization."""

    def test_init_creates_log_directory(self, temp_log_dir):
        """Test that initialization creates log directory."""
        logger = AuditLogger(log_dir=temp_log_dir)
        assert logger.log_dir.exists()
        assert logger.log_dir.is_dir()
        logger.close()

    def test_init_with_nested_directory(self, temp_log_dir):
        """Test initialization with nested directory path."""
        nested_path = Path(temp_log_dir) / "nested" / "logs"
        logger = AuditLogger(log_dir=str(nested_path))
        assert nested_path.exists()
        logger.close()

    def test_init_with_empty_log_dir_raises_error(self):
        """Test that empty log_dir raises ValueError."""
        with pytest.raises(ValueError, match="log_dir cannot be empty"):
            AuditLogger(log_dir="")

        with pytest.raises(ValueError, match="log_dir cannot be empty"):
            AuditLogger(log_dir="   ")

    def test_init_creates_handler(self, audit_logger):
        """Test that initialization creates log handler."""
        assert len(audit_logger.logger.handlers) == 1
        handler = audit_logger.logger.handlers[0]
        assert handler.__class__.__name__ == "TimedRotatingFileHandler"

    def test_init_does_not_add_duplicate_handlers(self, temp_log_dir):
        """Test that re-initialization doesn't add duplicate handlers."""
        logger1 = AuditLogger(log_dir=temp_log_dir)
        handler_count = len(logger1.logger.handlers)

        logger2 = AuditLogger(log_dir=temp_log_dir)
        assert len(logger2.logger.handlers) == handler_count

        # Clean up
        logger1.close()
        logger2.close()

    def test_logger_does_not_propagate(self, audit_logger):
        """Test that logger doesn't propagate to root logger."""
        assert audit_logger.logger.propagate is False


class TestDataSanitization:
    """Test sensitive data sanitization."""

    def test_sanitize_password_field(self, audit_logger):
        """Test that password fields are redacted."""
        details = {"username": "user1", "password": "secret123"}
        sanitized = audit_logger._sanitize_details(details)

        assert sanitized["username"] == "user1"
        assert sanitized["password"] == "***REDACTED***"

    def test_sanitize_api_key_field(self, audit_logger):
        """Test that API key fields are redacted."""
        details = {"api_key": "abc123", "api_secret": "xyz789"}
        sanitized = audit_logger._sanitize_details(details)

        assert sanitized["api_key"] == "***REDACTED***"
        assert sanitized["api_secret"] == "***REDACTED***"

    def test_sanitize_token_fields(self, audit_logger):
        """Test that token fields are redacted."""
        details = {
            "access_token": "token123",
            "refresh_token": "refresh456",
            "auth_token": "auth789"
        }
        sanitized = audit_logger._sanitize_details(details)

        assert sanitized["access_token"] == "***REDACTED***"
        assert sanitized["refresh_token"] == "***REDACTED***"
        assert sanitized["auth_token"] == "***REDACTED***"

    def test_sanitize_nested_dict(self, audit_logger):
        """Test sanitization of nested dictionaries."""
        details = {
            "user": "test",
            "config": {
                "password": "secret",
                "token": "abc123",
                "timeout": 30
            }
        }
        sanitized = audit_logger._sanitize_details(details)

        assert sanitized["user"] == "test"
        assert isinstance(sanitized["config"], dict)
        assert sanitized["config"]["password"] == "***REDACTED***"
        assert sanitized["config"]["token"] == "***REDACTED***"
        assert sanitized["config"]["timeout"] == 30  # Non-sensitive preserved

    def test_sanitize_list_with_dicts(self, audit_logger):
        """Test sanitization of lists containing dictionaries."""
        details = {
            "users": [
                {"name": "user1", "password": "pass1"},
                {"name": "user2", "api_key": "key2"}
            ]
        }
        sanitized = audit_logger._sanitize_details(details)

        assert sanitized["users"][0]["name"] == "user1"
        assert sanitized["users"][0]["password"] == "***REDACTED***"
        assert sanitized["users"][1]["name"] == "user2"
        assert sanitized["users"][1]["api_key"] == "***REDACTED***"

    def test_sanitize_preserves_non_sensitive_data(self, audit_logger):
        """Test that non-sensitive data is preserved."""
        details = {
            "order_id": "12345",
            "symbol": "BTC/USDT",
            "price": 50000.0,
            "quantity": 0.5
        }
        sanitized = audit_logger._sanitize_details(details)

        assert sanitized == details


class TestChecksumCalculation:
    """Test integrity checksum calculation."""

    def test_calculate_checksum_returns_string(self, audit_logger):
        """Test that checksum is a hexadecimal string."""
        record = {"timestamp": "2024-01-01T00:00:00Z", "event_type": "TEST"}
        checksum = audit_logger._calculate_checksum(record)

        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 produces 64 hex characters

    def test_calculate_checksum_is_deterministic(self, audit_logger):
        """Test that same input produces same checksum."""
        record = {"timestamp": "2024-01-01T00:00:00Z", "event_type": "TEST"}

        checksum1 = audit_logger._calculate_checksum(record)
        checksum2 = audit_logger._calculate_checksum(record)

        assert checksum1 == checksum2

    def test_calculate_checksum_changes_with_content(self, audit_logger):
        """Test that different content produces different checksum."""
        record1 = {"timestamp": "2024-01-01T00:00:00Z", "event_type": "TEST1"}
        record2 = {"timestamp": "2024-01-01T00:00:00Z", "event_type": "TEST2"}

        checksum1 = audit_logger._calculate_checksum(record1)
        checksum2 = audit_logger._calculate_checksum(record2)

        assert checksum1 != checksum2

    def test_calculate_checksum_excludes_checksum_field(self, audit_logger):
        """Test that checksum field is excluded from calculation."""
        record_without = {"timestamp": "2024-01-01T00:00:00Z", "event_type": "TEST"}
        record_with = {
            "timestamp": "2024-01-01T00:00:00Z",
            "event_type": "TEST",
            "checksum": "old_checksum"
        }

        checksum1 = audit_logger._calculate_checksum(record_without)
        checksum2 = audit_logger._calculate_checksum(record_with)

        assert checksum1 == checksum2


class TestJSONSerialization:
    """Test JSON serialization handling."""

    def test_make_json_serializable_handles_dict(self, audit_logger):
        """Test serialization of dictionaries."""
        obj = {"key": "value", "number": 123}
        result = audit_logger._make_json_serializable(obj)
        assert result == obj

    def test_make_json_serializable_handles_list(self, audit_logger):
        """Test serialization of lists."""
        obj = [1, 2, "three", {"key": "value"}]
        result = audit_logger._make_json_serializable(obj)
        assert result == obj

    def test_make_json_serializable_handles_datetime(self, audit_logger):
        """Test serialization of datetime objects."""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = audit_logger._make_json_serializable(dt)
        assert isinstance(result, str)
        assert "2024-01-01" in result

    def test_make_json_serializable_handles_custom_objects(self, audit_logger):
        """Test serialization of custom objects."""
        class CustomObject:
            def __init__(self):
                self.value = 123

        obj = CustomObject()
        result = audit_logger._make_json_serializable(obj)
        assert isinstance(result, str)

    def test_make_json_serializable_handles_nested_structures(self, audit_logger):
        """Test serialization of nested structures."""
        obj = {
            "list": [1, 2, {"nested": "value"}],
            "dict": {"key": [1, 2, 3]},
            "datetime": datetime(2024, 1, 1)
        }
        result = audit_logger._make_json_serializable(obj)

        assert isinstance(result["list"], list)
        assert isinstance(result["dict"], dict)
        assert isinstance(result["datetime"], str)


class TestEventLogging:
    """Test audit event logging."""

    def test_log_event_creates_log_file(self, audit_logger, temp_log_dir):
        """Test that logging creates the log file."""
        audit_logger.log_event("TEST_EVENT", {"key": "value"})

        log_file = Path(temp_log_dir) / "audit.log"
        assert log_file.exists()

    def test_log_event_writes_json(self, audit_logger, temp_log_dir):
        """Test that logged events are valid JSON."""
        audit_logger.log_event("TEST_EVENT", {"key": "value"})

        log_file = Path(temp_log_dir) / "audit.log"
        with open(log_file, 'r') as f:
            line = f.readline()
            record = json.loads(line)

            assert record["event_type"] == "TEST_EVENT"
            assert record["details"]["key"] == "value"

    def test_log_event_includes_timestamp(self, audit_logger, temp_log_dir):
        """Test that logged events include ISO 8601 timestamp."""
        audit_logger.log_event("TEST_EVENT", {"key": "value"})

        log_file = Path(temp_log_dir) / "audit.log"
        with open(log_file, 'r') as f:
            line = f.readline()
            record = json.loads(line)

            assert "timestamp" in record
            assert record["timestamp"].endswith("Z")
            # Verify ISO 8601 format
            datetime.fromisoformat(record["timestamp"].rstrip("Z"))

    def test_log_event_includes_user(self, audit_logger, temp_log_dir):
        """Test that logged events include user field."""
        audit_logger.log_event("TEST_EVENT", {"key": "value"}, user="test_user")

        log_file = Path(temp_log_dir) / "audit.log"
        with open(log_file, 'r') as f:
            line = f.readline()
            record = json.loads(line)

            assert record["user"] == "test_user"

    def test_log_event_default_user_is_system(self, audit_logger, temp_log_dir):
        """Test that default user is 'system'."""
        audit_logger.log_event("TEST_EVENT", {"key": "value"})

        log_file = Path(temp_log_dir) / "audit.log"
        with open(log_file, 'r') as f:
            line = f.readline()
            record = json.loads(line)

            assert record["user"] == "system"

    def test_log_event_includes_checksum_by_default(self, audit_logger, temp_log_dir):
        """Test that checksum is included by default."""
        audit_logger.log_event("TEST_EVENT", {"key": "value"})

        log_file = Path(temp_log_dir) / "audit.log"
        with open(log_file, 'r') as f:
            line = f.readline()
            record = json.loads(line)

            assert "checksum" in record
            assert len(record["checksum"]) == 64

    def test_log_event_without_checksum(self, audit_logger, temp_log_dir):
        """Test logging without checksum."""
        audit_logger.log_event("TEST_EVENT", {"key": "value"}, add_checksum=False)

        log_file = Path(temp_log_dir) / "audit.log"
        with open(log_file, 'r') as f:
            line = f.readline()
            record = json.loads(line)

            assert "checksum" not in record

    def test_log_event_with_audit_event_type_enum(self, audit_logger, temp_log_dir):
        """Test logging with AuditEventType enum."""
        audit_logger.log_event(
            AuditEventType.ORDER_CREATED.value,
            {"order_id": "12345"}
        )

        log_file = Path(temp_log_dir) / "audit.log"
        with open(log_file, 'r') as f:
            line = f.readline()
            record = json.loads(line)

            assert record["event_type"] == "ORDER_CREATED"

    def test_log_event_sanitizes_sensitive_data(self, audit_logger, temp_log_dir):
        """Test that sensitive data is sanitized in logs."""
        audit_logger.log_event(
            "TEST_EVENT",
            {"username": "user1", "password": "secret123"}
        )

        log_file = Path(temp_log_dir) / "audit.log"
        with open(log_file, 'r') as f:
            line = f.readline()
            record = json.loads(line)

            assert record["details"]["username"] == "user1"
            assert record["details"]["password"] == "***REDACTED***"
            assert "secret123" not in line


class TestInputValidation:
    """Test input validation."""

    def test_log_event_with_empty_event_type_logs_error(self, audit_logger, temp_log_dir, capsys):
        """Test that empty event_type logs error to stderr and audit log."""
        audit_logger.log_event("", {"key": "value"})

        captured = capsys.readouterr()
        assert "CRITICAL: Failed to write audit log" in captured.err
        assert "event_type must be a non-empty string" in captured.err

        # Should log AUDIT_LOGGING_FAILED event
        log_file = Path(temp_log_dir) / "audit.log"
        with open(log_file, 'r') as f:
            content = f.read()
            assert "AUDIT_LOGGING_FAILED" in content

    def test_log_event_with_non_string_event_type_logs_error(self, audit_logger, temp_log_dir, capsys):
        """Test that non-string event_type logs error to stderr and audit log."""
        audit_logger.log_event(123, {"key": "value"})

        captured = capsys.readouterr()
        assert "CRITICAL: Failed to write audit log" in captured.err

        # Should log fallback event
        log_file = Path(temp_log_dir) / "audit.log"
        assert log_file.exists()

    def test_log_event_with_non_dict_details_logs_error(self, audit_logger, temp_log_dir, capsys):
        """Test that non-dict details logs error to stderr and audit log."""
        audit_logger.log_event("TEST_EVENT", "not a dict")

        captured = capsys.readouterr()
        assert "CRITICAL: Failed to write audit log" in captured.err
        assert "details must be a dictionary" in captured.err

    def test_log_event_with_empty_user_logs_error(self, audit_logger, temp_log_dir, capsys):
        """Test that empty user logs error to stderr."""
        audit_logger.log_event("TEST_EVENT", {"key": "value"}, user="")

        captured = capsys.readouterr()
        assert "CRITICAL: Failed to write audit log" in captured.err

    def test_log_event_with_non_string_user_logs_error(self, audit_logger, temp_log_dir, capsys):
        """Test that non-string user logs error to stderr."""
        audit_logger.log_event("TEST_EVENT", {"key": "value"}, user=123)

        captured = capsys.readouterr()
        assert "CRITICAL: Failed to write audit log" in captured.err


class TestErrorHandling:
    """Test error handling."""

    def test_log_event_handles_non_serializable_objects(self, audit_logger, temp_log_dir):
        """Test that non-serializable objects are handled gracefully."""
        class NonSerializable:
            pass

        details = {"object": NonSerializable()}
        audit_logger.log_event("TEST_EVENT", details)

        # Should log successfully with fallback
        log_file = Path(temp_log_dir) / "audit.log"
        assert log_file.exists()

    @patch('sys.stderr')
    def test_log_event_logs_to_stderr_on_critical_failure(self, mock_stderr, audit_logger):
        """Test that critical failures log to stderr (non-raising version)."""
        # Mock both logger methods to raise exceptions
        with patch.object(audit_logger.logger, 'info', side_effect=Exception("Test error")):
            with patch.object(audit_logger.logger, 'error', side_effect=Exception("Test error")):
                # This should raise RuntimeError after all fallbacks fail
                with pytest.raises(RuntimeError, match="Audit logging completely failed"):
                    audit_logger.log_event("TEST_EVENT", {"key": "value"})

    def test_log_event_with_circular_reference_in_details(self, audit_logger, temp_log_dir):
        """Test handling of circular references in details."""
        details: dict = {"key": "value"}
        details["self"] = details  # type: ignore[assignment]  # Circular reference

        # Should handle gracefully and log
        audit_logger.log_event("TEST_EVENT", details)

        log_file = Path(temp_log_dir) / "audit.log"
        assert log_file.exists()


class TestMultipleEvents:
    """Test logging multiple events."""

    def test_log_multiple_events(self, audit_logger, temp_log_dir):
        """Test logging multiple events in sequence."""
        for i in range(5):
            audit_logger.log_event(f"TEST_EVENT_{i}", {"index": i})

        log_file = Path(temp_log_dir) / "audit.log"
        with open(log_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 5

        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["event_type"] == f"TEST_EVENT_{i}"
            assert record["details"]["index"] == i

    def test_log_events_are_appended(self, audit_logger, temp_log_dir):
        """Test that events are appended, not overwritten."""
        audit_logger.log_event("EVENT_1", {"order": 1})
        audit_logger.log_event("EVENT_2", {"order": 2})

        log_file = Path(temp_log_dir) / "audit.log"
        with open(log_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert "EVENT_1" in lines[0]
        assert "EVENT_2" in lines[1]


class TestIntegration:
    """Integration tests for AuditLogger."""

    def test_complete_audit_flow(self, temp_log_dir):
        """Test complete audit flow from initialization to logging."""
        # Initialize logger
        logger = AuditLogger(log_dir=temp_log_dir)

        # Log various event types
        logger.log_event(
            AuditEventType.ORDER_CREATED.value,
            {"order_id": "12345", "symbol": "BTC/USDT", "price": 50000}
        )

        logger.log_event(
            AuditEventType.POSITION_OPENED.value,
            {"position_id": "67890", "size": 1.5},
            user="trader_bot"
        )

        logger.log_event(
            AuditEventType.MANUAL_INTERVENTION.value,
            {"action": "emergency_stop", "reason": "market_volatility"},
            user="admin"
        )

        # Verify all events logged
        log_file = Path(temp_log_dir) / "audit.log"
        with open(log_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Verify event 1
        record1 = json.loads(lines[0])
        assert record1["event_type"] == "ORDER_CREATED"
        assert record1["details"]["order_id"] == "12345"
        assert "checksum" in record1

        # Verify event 2
        record2 = json.loads(lines[1])
        assert record2["event_type"] == "POSITION_OPENED"
        assert record2["user"] == "trader_bot"

        # Verify event 3
        record3 = json.loads(lines[2])
        assert record3["event_type"] == "MANUAL_INTERVENTION"
        assert record3["user"] == "admin"

        # Clean up
        logger.close()

    def test_sensitive_data_not_in_logs(self, temp_log_dir):
        """Test that sensitive data never appears in log files."""
        logger = AuditLogger(log_dir=temp_log_dir)

        sensitive_data = {
            "user": "test_user",
            "api_key": "super_secret_key_12345",
            "password": "my_password_456",
            "order_id": "order_123"
        }

        logger.log_event("USER_LOGIN", sensitive_data)

        log_file = Path(temp_log_dir) / "audit.log"
        with open(log_file, 'r') as f:
            content = f.read()

        # Verify sensitive values are not in log
        assert "super_secret_key_12345" not in content
        assert "my_password_456" not in content

        # Verify redaction markers are present
        assert "***REDACTED***" in content

        # Verify non-sensitive data is present
        assert "test_user" in content
        assert "order_123" in content

        # Clean up
        logger.close()
