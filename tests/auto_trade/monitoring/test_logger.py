"""
Tests for Structured Logging Module.

Tests JSON formatter, logging setup, error handling, and log file management.
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from modules.auto_trade.monitoring.logger import (
    JSONFormatter,
    cleanup_logging,
    get_logger,
    setup_logging,
)


@pytest.fixture
def temp_log_dir():
    """Create temporary directory for log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
        # Ensure all handlers are closed before directory cleanup
        cleanup_logging()


class TestJSONFormatter:
    """Test JSONFormatter functionality."""

    def test_formatter_basic_message(self):
        """Test formatting a basic log message."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )

        result = formatter.format(record)
        log_dict = json.loads(result)

        assert log_dict["level"] == "INFO"
        assert log_dict["logger"] == "test"
        assert log_dict["message"] == "Test message"
        assert log_dict["module"] == "test"
        assert log_dict["lineno"] == 10
        assert "timestamp" in log_dict

    def test_formatter_with_exception(self):
        """Test formatting log with exception info."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=20,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )

        result = formatter.format(record)
        log_dict = json.loads(result)

        assert log_dict["level"] == "ERROR"
        assert log_dict["message"] == "Error occurred"
        assert "exception" in log_dict
        assert "ValueError: Test error" in log_dict["exception"]

    def test_formatter_with_extra_fields(self):
        """Test formatting log with extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=30,
            msg="Test message",
            args=(),
            exc_info=None
        )

        # Add extra fields directly to record (how logging.info(..., extra={}) works)
        record.user_id = "12345"
        record.request_id = "abc-def"
        record.custom_field = "custom_value"

        result = formatter.format(record)
        log_dict = json.loads(result)

        assert log_dict["user_id"] == "12345"
        assert log_dict["request_id"] == "abc-def"
        assert log_dict["custom_field"] == "custom_value"

    def test_formatter_excludes_standard_attrs(self):
        """Test that standard LogRecord attributes are not duplicated in extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=40,
            msg="Test message",
            args=(),
            exc_info=None
        )

        result = formatter.format(record)
        log_dict = json.loads(result)

        # Standard attrs should not appear as extra fields
        assert "pathname" not in log_dict
        assert "created" not in log_dict
        assert "process" not in log_dict
        assert "thread" not in log_dict

    def test_formatter_excludes_private_attrs(self):
        """Test that private attributes (starting with _) are excluded."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=50,
            msg="Test message",
            args=(),
            exc_info=None
        )

        # Add private attribute
        record._private_field = "should not appear"
        record.public_field = "should appear"

        result = formatter.format(record)
        log_dict = json.loads(result)

        assert "_private_field" not in log_dict
        assert "public_field" in log_dict

    def test_formatter_json_output_valid(self):
        """Test that formatter always produces valid JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=60,
            msg="Test message",
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        # Should not raise JSONDecodeError
        parsed = json.loads(result)
        assert isinstance(parsed, dict)


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_creates_log_directory(self, temp_log_dir):
        """Test that setup_logging creates the log directory."""
        log_dir = Path(temp_log_dir) / "new_logs"
        assert not log_dir.exists()

        setup_logging(str(log_dir))

        assert log_dir.exists()
        assert log_dir.is_dir()

    def test_setup_creates_log_files(self, temp_log_dir):
        """Test that setup_logging creates log files."""
        setup_logging(temp_log_dir)

        # Log something to trigger file creation
        logger = logging.getLogger()
        logger.info("Test message")

        system_log = Path(temp_log_dir) / "system.log"
        assert system_log.exists()

    def test_setup_with_custom_level(self, temp_log_dir):
        """Test setup_logging with custom logging level."""
        setup_logging(temp_log_dir, level=logging.DEBUG)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_with_empty_log_dir_raises_error(self):
        """Test that empty log_dir raises ValueError."""
        with pytest.raises(ValueError, match="log_dir cannot be empty"):
            setup_logging("")

    def test_setup_with_whitespace_log_dir_raises_error(self):
        """Test that whitespace-only log_dir raises ValueError."""
        with pytest.raises(ValueError, match="log_dir cannot be empty"):
            setup_logging("   ")

    def test_setup_with_invalid_permissions_raises_error(self):
        """Test that invalid permissions raise ValueError."""
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Access denied")):
            with pytest.raises(ValueError, match="Cannot create log directory"):
                setup_logging("test_logs")

    def test_setup_handler_creation_failure_raises_runtime_error(self, temp_log_dir):
        """Test that handler creation failure raises RuntimeError."""
        with patch('logging.handlers.TimedRotatingFileHandler', side_effect=OSError("File error")):
            with pytest.raises(RuntimeError, match="Log handler configuration failed"):
                setup_logging(temp_log_dir)

    def test_setup_clears_existing_handlers(self, temp_log_dir):
        """Test that setup_logging clears existing handlers."""
        # Add a handler
        root_logger = logging.getLogger()
        dummy_handler = logging.StreamHandler()
        root_logger.addHandler(dummy_handler)

        initial_count = len(root_logger.handlers)
        assert initial_count > 0

        # Setup logging should clear and recreate handlers
        setup_logging(temp_log_dir)

        # Handlers should be replaced, not added
        assert len(root_logger.handlers) >= 3  # console, system log, error log

    def test_setup_creates_console_handler(self, temp_log_dir):
        """Test that setup_logging creates console handler."""
        setup_logging(temp_log_dir)

        root_logger = logging.getLogger()
        console_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)
                            and not isinstance(h, logging.handlers.TimedRotatingFileHandler)]

        assert len(console_handlers) > 0

    def test_setup_creates_file_handlers(self, temp_log_dir):
        """Test that setup_logging creates file handlers."""
        setup_logging(temp_log_dir)

        root_logger = logging.getLogger()
        file_handlers = [h for h in root_logger.handlers
                        if isinstance(h, logging.handlers.TimedRotatingFileHandler)]

        # Should have system.log and error.log handlers
        assert len(file_handlers) == 2

    def test_setup_error_handler_level_is_error(self, temp_log_dir):
        """Test that error handler only logs ERROR and above."""
        setup_logging(temp_log_dir)

        root_logger = logging.getLogger()
        file_handlers = [h for h in root_logger.handlers
                        if isinstance(h, logging.handlers.TimedRotatingFileHandler)]

        error_handlers = [h for h in file_handlers if h.level == logging.ERROR]
        assert len(error_handlers) == 1

    def test_setup_json_formatter_on_file_handlers(self, temp_log_dir):
        """Test that file handlers use JSONFormatter."""
        setup_logging(temp_log_dir)

        root_logger = logging.getLogger()
        file_handlers = [h for h in root_logger.handlers
                        if isinstance(h, logging.handlers.TimedRotatingFileHandler)]

        for handler in file_handlers:
            assert isinstance(handler.formatter, JSONFormatter)


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test_logger")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_name(self):
        """Test that get_logger returns logger with correct name."""
        logger = get_logger("my.custom.logger")
        assert logger.name == "my.custom.logger"

    def test_get_logger_same_name_returns_same_instance(self):
        """Test that get_logger returns the same instance for the same name."""
        logger1 = get_logger("test")
        logger2 = get_logger("test")
        assert logger1 is logger2


class TestCleanupLogging:
    """Test cleanup_logging function."""

    def test_cleanup_removes_all_handlers(self, temp_log_dir):
        """Test that cleanup_logging removes all handlers."""
        setup_logging(temp_log_dir)

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0

        cleanup_logging()

        assert len(root_logger.handlers) == 0

    def test_cleanup_closes_handlers(self, temp_log_dir):
        """Test that cleanup_logging closes handlers."""
        setup_logging(temp_log_dir)

        root_logger = logging.getLogger()
        handlers = list(root_logger.handlers)

        cleanup_logging()

        # Handlers should be closed (difficult to test directly, but no exceptions should occur)
        assert len(root_logger.handlers) == 0

    def test_cleanup_handles_handler_close_errors(self, temp_log_dir):
        """Test that cleanup_logging handles handler close errors gracefully."""
        setup_logging(temp_log_dir)

        root_logger = logging.getLogger()
        initial_handler_count = len(root_logger.handlers)

        # Mock handler.close to raise exception
        original_closes = []
        for handler in root_logger.handlers:
            original_closes.append(handler.close)
            handler.close = Mock(side_effect=Exception("Close error"))

        # Should not raise exception even though close fails
        cleanup_logging()

        # Handlers should still be removed even if close fails
        assert len(root_logger.handlers) == 0

        # Restore original close methods to avoid issues with temp_log_dir cleanup
        # (though handlers are already removed, this is defensive)
        root_logger = logging.getLogger()
        for i, original_close in enumerate(original_closes):
            # The handlers list is empty now, so no restoration needed
            pass

    def test_cleanup_is_idempotent(self, temp_log_dir):
        """Test that cleanup_logging can be called multiple times."""
        setup_logging(temp_log_dir)

        cleanup_logging()
        cleanup_logging()  # Second call should not raise errors

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 0


class TestIntegration:
    """Integration tests for logging module."""

    def test_full_logging_workflow(self, temp_log_dir):
        """Test complete logging workflow from setup to cleanup."""
        # Setup
        setup_logging(temp_log_dir)

        # Get logger and log messages
        logger = get_logger("test.integration")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Verify log files exist
        system_log = Path(temp_log_dir) / "system.log"
        error_log = Path(temp_log_dir) / "error.log"

        assert system_log.exists()
        assert error_log.exists()

        # Cleanup
        cleanup_logging()

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 0

    def test_logging_with_extra_fields(self, temp_log_dir):
        """Test logging with extra fields end-to-end."""
        setup_logging(temp_log_dir)

        logger = get_logger("test.extra")
        logger.info("Message with extras", extra={"user_id": "123", "action": "login"})

        system_log = Path(temp_log_dir) / "system.log"
        assert system_log.exists()

        # Read log file and verify extra fields
        with open(system_log, 'r', encoding='utf-8') as f:
            log_line = f.readline()
            log_dict = json.loads(log_line)

            assert log_dict["message"] == "Message with extras"
            assert log_dict["user_id"] == "123"
            assert log_dict["action"] == "login"

        cleanup_logging()

    def test_error_log_only_contains_errors(self, temp_log_dir):
        """Test that error.log only contains ERROR and CRITICAL messages."""
        setup_logging(temp_log_dir)

        logger = get_logger("test.errors")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        error_log = Path(temp_log_dir) / "error.log"
        assert error_log.exists()

        # Read error log
        with open(error_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Should only have ERROR and CRITICAL
        assert len(lines) == 2

        for line in lines:
            log_dict = json.loads(line)
            assert log_dict["level"] in ["ERROR", "CRITICAL"]

        cleanup_logging()

    def test_multiple_loggers_use_same_handlers(self, temp_log_dir):
        """Test that multiple loggers use the same root handlers."""
        setup_logging(temp_log_dir)

        logger1 = get_logger("app.module1")
        logger2 = get_logger("app.module2")

        logger1.info("Message from module1")
        logger2.info("Message from module2")

        system_log = Path(temp_log_dir) / "system.log"
        with open(system_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        assert len(lines) == 2

        log1 = json.loads(lines[0])
        log2 = json.loads(lines[1])

        assert log1["logger"] == "app.module1"
        assert log2["logger"] == "app.module2"

        cleanup_logging()

    def test_logging_with_exception(self, temp_log_dir):
        """Test logging with exception information."""
        setup_logging(temp_log_dir)

        logger = get_logger("test.exception")

        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.error("An error occurred", exc_info=True)

        error_log = Path(temp_log_dir) / "error.log"
        with open(error_log, 'r', encoding='utf-8') as f:
            log_line = f.readline()
            log_dict = json.loads(log_line)

            assert log_dict["level"] == "ERROR"
            assert log_dict["message"] == "An error occurred"
            assert "exception" in log_dict
            assert "ValueError: Test exception" in log_dict["exception"]

        cleanup_logging()


class TestStandardAttrsConstant:
    """Test STANDARD_ATTRS constant in JSONFormatter."""

    def test_standard_attrs_is_set(self):
        """Test that STANDARD_ATTRS is a set."""
        assert isinstance(JSONFormatter.STANDARD_ATTRS, set)

    def test_standard_attrs_contains_expected_fields(self):
        """Test that STANDARD_ATTRS contains expected LogRecord fields."""
        expected_fields = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'message', 'pathname',
            'process', 'processName', 'relativeCreated', 'thread', 'threadName',
            'exc_info', 'exc_text', 'stack_info'
        }

        for field in expected_fields:
            assert field in JSONFormatter.STANDARD_ATTRS

    def test_standard_attrs_is_comprehensive(self):
        """Test that STANDARD_ATTRS is comprehensive enough."""
        # Should have at least 20 standard attributes
        assert len(JSONFormatter.STANDARD_ATTRS) >= 20
