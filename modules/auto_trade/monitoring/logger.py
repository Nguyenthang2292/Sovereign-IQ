"""
Structured Logging Module.

Configures Python logging to output JSON formatted logs for machine readability,
and manages log files for different components.
"""

import json
import logging
import logging.handlers
from pathlib import Path

# Constants
LOG_DIR = Path("logs")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for logs.
    """

    def format(self, record: logging.LogRecord) -> str:
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

        # Add extra fields if passed
        if hasattr(record, "extra"):
            log_record.update(record.extra)

        return json.dumps(log_record)


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """
    Configure system-wide logging.

    Args:
        log_dir: Directory to store log files.
        level: Minimum logging level.
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # JSON Formatter
    json_formatter = JSONFormatter()

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
        filename=log_path / "error.log", when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter)
    root_logger.addHandler(error_handler)

    # Configure specific loggers if needed
    logging.getLogger("modules.auto_trade").setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)
