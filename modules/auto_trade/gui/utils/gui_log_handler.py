"""
GUI Logging Handler

Custom logging handler that sends log records to a queue for GUI display.
"""

import logging
import queue
from datetime import datetime


class GUILogHandler(logging.Handler):
    """Custom logging handler that sends logs to a queue for GUI display."""

    def __init__(self, log_queue: queue.Queue):
        """
        Initialize the GUI log handler.

        Args:
            log_queue: Queue to send log records to
        """
        super().__init__()
        self.log_queue = log_queue
        self.setLevel(logging.INFO)  # Default to INFO level

        # Format logs
        formatter = logging.Formatter("%(message)s")
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord):
        """
        Emit a log record to the queue.

        Args:
            record: Log record to emit
        """
        try:
            # Create a dictionary with the log info
            log_dict = {
                "level": record.levelname,
                "message": self.format(record),
                "timestamp": datetime.fromtimestamp(record.created),
                "logger": record.name,
            }

            # Put in queue (non-blocking)
            try:
                self.log_queue.put_nowait(log_dict)
            except queue.Full:
                # If queue is full, drop oldest message and try again
                try:
                    self.log_queue.get_nowait()
                    self.log_queue.put_nowait(log_dict)
                except queue.Empty:
                    pass

        except Exception:
            # Don't raise exceptions in logging handler
            self.handleError(record)


def setup_gui_logging(log_queue: queue.Queue, logger_names: list = None):
    """
    Set up GUI logging for specified loggers.

    Args:
        log_queue: Queue to send log records to
        logger_names: List of logger names to attach handler to.
                     If None, attaches to root logger.
    """
    handler = GUILogHandler(log_queue)

    if logger_names is None:
        # Attach to root logger
        logging.getLogger().addHandler(handler)
    else:
        # Attach to specific loggers
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    return handler
