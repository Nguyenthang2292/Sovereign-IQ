"""
Tests for CLILogger (web/utils/cli_logger.py).

Tests cover:
- Starting/stopping logger
- Capturing stdout
- Capturing stderr
- Capturing Python logging
- Context manager usage
- Thread safety
"""

import logging
import sys
from unittest.mock import patch

import pytest

from web.utils.cli_logger import CLILogger
from web.utils.log_manager import LogFileManager


@pytest.fixture
def tmp_logs_dir(tmp_path):
    """Create temporary logs directory."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    return logs_dir


@pytest.fixture
def log_manager(tmp_logs_dir):
    """Create LogFileManager instance for testing."""
    with patch("web.utils.cli_logger.get_log_manager") as mock_get:
        manager = LogFileManager(logs_dir=str(tmp_logs_dir))
        mock_get.return_value = manager
        yield manager


class TestCLILogger:
    """Test CLILogger class."""

    def test_init(self, log_manager):
        """Test CLILogger initialization."""
        logger = CLILogger("test-session", "scan")
        assert logger.session_id == "test-session"
        assert logger.command_type == "scan"
        assert not logger._active

    def test_start_stop(self, log_manager, tmp_logs_dir):
        """Test starting and stopping logger."""
        logger = CLILogger("test-session", "scan")
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        logger.start()
        assert logger._active
        assert sys.stdout != original_stdout
        assert sys.stderr != original_stderr

        logger.stop()
        assert not logger._active
        assert sys.stdout == original_stdout
        assert sys.stderr == original_stderr

    def test_capture_stdout(self, log_manager, tmp_logs_dir):
        """Test capturing stdout output."""
        logger = CLILogger("test-session", "scan")

        with logger.capture_output():
            print("Test stdout message")

        # Check log file
        log_path = tmp_logs_dir / "scan_test-session.log"
        assert log_path.exists(), "Log file should be created"
        content = log_path.read_text(encoding="utf-8")
        assert "Test stdout message" in content

    def test_capture_stderr(self, log_manager, tmp_logs_dir):
        """Test capturing stderr output."""
        logger = CLILogger("test-session", "scan")

        with logger.capture_output():
            print("Test stderr message", file=sys.stderr)

        log_path = tmp_logs_dir / "scan_test-session.log"
        assert log_path.exists(), "Log file should be created"
        content = log_path.read_text(encoding="utf-8")
        assert "Test stderr message" in content

    def test_capture_python_logging(self, log_manager, tmp_logs_dir):
        """Test capturing Python logging output."""
        logger = CLILogger("test-session", "scan")
        test_logger = logging.getLogger("test_logger")
        test_logger.setLevel(logging.INFO)

        with logger.capture_output():
            test_logger.info("Test logging message")

        # Check log file
        log_path = tmp_logs_dir / "scan_test-session.log"
        assert log_path.exists(), "Log file should be created"
        content = log_path.read_text(encoding="utf-8")
        assert "Test logging message" in content

    def test_context_manager(self, log_manager):
        """Test using CLILogger as context manager."""
        logger = CLILogger("test-session", "scan")
        original_stdout = sys.stdout

        with logger.capture_output():
            assert logger._active
            assert sys.stdout != original_stdout

        assert not logger._active
        assert sys.stdout == original_stdout

    def test_multiple_start_stop(self, log_manager):
        """Test multiple start/stop calls."""
        logger = CLILogger("test-session", "scan")

        logger.start()
        logger.start()  # Should be idempotent
        assert logger._active

        logger.stop()
        logger.stop()  # Should be idempotent
        assert not logger._active

    def test_write_while_inactive(self, log_manager, tmp_logs_dir):
        """Test that writes don't happen when logger is inactive."""
        logger = CLILogger("test-session", "scan")

        # Write without starting
        logger._write_to_log("Should not be written", "output")

        log_path = tmp_logs_dir / "scan_test-session.log"
        assert not log_path.exists() or "Should not be written" not in log_path.read_text(encoding="utf-8"), (
            "Log file should not exist, or the message should not be present if the file exists"
        )

    def test_different_command_types(self, log_manager, tmp_logs_dir):
        """Test logger with different command types."""
        scan_logger = CLILogger("session-1", "scan")
        analyze_logger = CLILogger("session-1", "analyze")

        with scan_logger.capture_output():
            print("Scan message")

        with analyze_logger.capture_output():
            print("Analyze message")

        scan_log = tmp_logs_dir / "scan_session-1.log"
        analyze_log = tmp_logs_dir / "analyze_session-1.log"

        assert scan_log.exists(), "Scan log file should be created"
        assert "Scan message" in scan_log.read_text(encoding="utf-8")

        assert analyze_log.exists(), "Analyze log file should be created"
        assert "Analyze message" in analyze_log.read_text(encoding="utf-8")
