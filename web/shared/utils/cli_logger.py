"""
CLI Logger for capturing stdout/stderr and Python logging output to log files.
"""

import logging
import sys
import threading
from contextlib import contextmanager
from datetime import datetime

from modules.common.ui.logging import log_debug, log_error

from .log_manager import get_log_manager

# Note: logging module is still needed for intercepting Python logging
logger = logging.getLogger(__name__)


class CLILogger:
    """Captures stdout/stderr and Python logging output, writes to log file."""

    def __init__(self, session_id: str, command_type: str = "scan"):
        """
        Initialize CLILogger.

        Args:
            session_id: Unique session identifier
            command_type: Type of command ('scan' or 'analyze')
        """
        self.session_id = session_id
        self.command_type = command_type
        self.log_manager = get_log_manager()

        self._original_stdout = None
        self._original_stderr = None
        self._active = False
        self._lock = threading.Lock()

        self._file_log_handler = None
        self._original_log_handlers = None
        self._logging_failed = False

    def _write_to_log(self, text: str, stream_type: str = "output"):
        """Write text to log file."""
        if not self._active:
            return

        try:
            # Add timestamp prefix for better readability
            # Use dependency injection for time source to decouple from global datetime operations.
            if hasattr(self, "_now_func") and callable(self._now_func):
                now = self._now_func()
            else:
                now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] {text}"

            self.log_manager.write_log(self.session_id, log_message, self.command_type)
        except Exception as e:
            # Prevent recursion: if logging already failed, don't try to write to stderr
            if not self._logging_failed:
                self._logging_failed = True
                try:
                    # Write minimal fallback report to stderr
                    error_msg = f"CLILogger error (session={self.session_id}, cmd={self.command_type}): {str(e)}\n"
                    sys.stderr.write(error_msg)
                    sys.stderr.flush()
                except Exception:
                    # If even stderr write fails, give up silently
                    pass

    def _setup_log_interception(self):
        """Setup Python logging interception."""
        root_logger = logging.getLogger()

        # Save original handlers
        self._original_log_handlers = {"handlers": root_logger.handlers[:], "level": root_logger.level}

        # Create custom handler that writes to log file
        class FileLogHandler(logging.Handler):
            def __init__(self, cli_logger_instance):
                super().__init__()
                self.cli_logger = cli_logger_instance

            def emit(self, record):
                if self.cli_logger._active:
                    try:
                        message = self.format(record)
                        self.cli_logger._write_to_log(message, "info")
                    except Exception as e:
                        log_error(f"Error in FileLogHandler.emit: {e}")

        self._file_log_handler = FileLogHandler(self)
        self._file_log_handler.setLevel(logging.INFO)

        # Add handler to root logger
        root_logger.addHandler(self._file_log_handler)
        root_logger.setLevel(logging.INFO)

    def _restore_log_interception(self):
        """Restore original logging configuration."""
        root_logger = logging.getLogger()

        if self._file_log_handler:
            root_logger.removeHandler(self._file_log_handler)

        if self._original_log_handlers:
            for handler in self._original_log_handlers["handlers"]:
                root_logger.addHandler(handler)

            root_logger.setLevel(self._original_log_handlers["level"])

    def start(self):
        """Start capturing output."""
        with self._lock:
            if self._active:
                return

            # Save original streams if not already saved
            if self._original_stdout is None:
                self._original_stdout = sys.stdout
            if self._original_stderr is None:
                self._original_stderr = sys.stderr

            self._active = True

            # Create custom file-like objects that write to both original stream and log file
            class TeeOutput:
                def __init__(self, original_stream, cli_logger, stream_type):
                    self.original = original_stream
                    self.cli_logger = cli_logger
                    self.stream_type = stream_type

                def write(self, data):
                    if isinstance(data, bytes):
                        data = data.decode("utf-8", errors="replace")

                    # Write to original stream
                    try:
                        self.original.write(data)
                        self.original.flush()
                    except (ValueError, OSError, AttributeError):
                        # Ignore expected stream errors (closed, mocked, etc.)
                        pass

                    # Write to log file (only if data is not empty)
                    # Use try-except to prevent recursion when _write_to_log triggers imports
                    if self.cli_logger._active and data:
                        try:
                            self.cli_logger._write_to_log(data.rstrip("\n\r"), self.stream_type)
                        except (RecursionError, ImportError, AttributeError):
                            # Silently ignore recursion/import errors to prevent infinite loops
                            # This can happen when __import__ is mocked in tests
                            pass
                    return len(data) if hasattr(data, "__len__") else len(str(data))

                def flush(self):
                    self.original.flush()

                def __getattr__(self, name):
                    return getattr(self.original, name)

            # Redirect stdout and stderr
            sys.stdout = TeeOutput(self._original_stdout, self, "output")
            sys.stderr = TeeOutput(self._original_stderr, self, "error")

            # Setup logging interception
            self._setup_log_interception()

            log_debug(f"CLILogger started for session {self.session_id}")

    def stop(self):
        """Stop capturing output."""
        with self._lock:
            if not self._active:
                return

            self._active = False

            # Restore stdout and stderr
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr

            # Restore logging
            self._restore_log_interception()

            log_debug(f"CLILogger stopped for session {self.session_id}")

    @contextmanager
    def capture_output(self):
        """Context manager for capturing output."""
        self.start()
        try:
            yield
        finally:
            self.stop()
