"""
Structured Logging Module.

Configures Python logging to output JSON formatted logs for machine readability,
and manages log files for different components.
"""

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Set

# Constants
LOG_DIR = Path("logs")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for logs.
    """

    # Standard LogRecord attributes to exclude from extra fields
    STANDARD_ATTRS: Set[str] = {
        'name', 'msg', 'args', 'created', 'filename', 'funcName', 'levelname',
        'levelno', 'lineno', 'module', 'msecs', 'message', 'pathname', 'process',
        'processName', 'relativeCreated', 'thread', 'threadName', 'exc_info',
        'exc_text', 'stack_info', 'getMessage', 'asctime'
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.

        Args:
            record: LogRecord instance to format

        Returns:
            JSON-formatted string
        """
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "lineno": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Add extra fields if passed via logging.info(..., extra={})
        # Extra fields are merged directly into record.__dict__
        for key, value in record.__dict__.items():
            if key not in self.STANDARD_ATTRS and not key.startswith('_'):
                log_record[key] = value

        return json.dumps(log_record)


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """
    Configure system-wide logging.

    Args:
        log_dir: Directory to store log files.
        level: Minimum logging level.

    Raises:
        ValueError: If log_dir is empty or cannot be created
        RuntimeError: If handler configuration fails
    """
    # Validate log_dir
    if not log_dir or not log_dir.strip():
        raise ValueError("log_dir cannot be empty")

    # Create logs directory with error handling
    log_path = Path(log_dir)
    try:
        log_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise ValueError(f"Cannot create log directory '{log_dir}': {e}")

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # JSON Formatter
    json_formatter = JSONFormatter()

    try:
        # 1. Console Handler (Standard Text)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(console_handler)

        # 2. Daily Rotating File Handler for General Logs
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_path / "system.log",
            when="midnight",
            interval=1,
            backupCount=30,  # Keep 30 days
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)

        # 3. Error Log (Separate file for errors)
        error_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_path / "error.log",
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(json_formatter)
        root_logger.addHandler(error_handler)

    except (OSError, PermissionError) as e:
        # Critical fallback: log to stderr if file handlers fail
        error_msg = f"CRITICAL: Failed to create log handlers: {e}"
        print(error_msg, file=sys.stderr)

        # Keep console handler only
        raise RuntimeError(f"Log handler configuration failed: {e}")

    # Configure specific loggers if needed
    logging.getLogger("modules.auto_trade").setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def cleanup_logging() -> None:
    """
    Close all log handlers and release file handles.

    Call this method when shutting down to ensure proper cleanup
    and prevent memory leaks on Windows.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:  # Copy list to avoid modification during iteration
        try:
            handler.close()
        except Exception:
            pass  # Ignore errors during close
        finally:
            # Always remove handler even if close failed
            try:
                root_logger.removeHandler(handler)
            except Exception:
                pass  # Ignore errors during removal
