"""
Stdout Redirector for GUI Logging

Captures stdout (print statements) and redirects to GUI log queue.
"""

import queue
import sys
from datetime import datetime
from io import StringIO


class StdoutRedirector:
    """Redirects stdout to both console and GUI log queue."""

    def __init__(self, log_queue: queue.Queue, original_stdout=None):
        """
        Initialize stdout redirector.

        Args:
            log_queue: Queue to send log messages to
            original_stdout: Original stdout to also write to (for console output)
        """
        self.log_queue = log_queue
        self.original_stdout = original_stdout or sys.stdout
        self.buffer = StringIO()

        # Store encoding and other attributes from original stdout
        self.encoding = getattr(self.original_stdout, 'encoding', 'utf-8')
        self.errors = getattr(self.original_stdout, 'errors', 'replace')

    def write(self, message):
        """Write message to both original stdout and log queue."""
        # Handle both string and bytes
        if isinstance(message, bytes):
            try:
                message = message.decode(self.encoding, errors=self.errors)
            except Exception:
                message = str(message)

        # Ensure message is string
        if not isinstance(message, str):
            message = str(message)

        # Write to original stdout (console) - with error handling
        try:
            self.original_stdout.write(message)
            self.original_stdout.flush()
        except Exception as e:
            # If console output fails, continue with GUI logging
            pass

        # Add to buffer
        self.buffer.write(message)

        # If we have a complete line, send to log queue
        if "\n" in message:
            self._flush_buffer()

    def _flush_buffer(self):
        """Flush buffered content to log queue."""
        content = self.buffer.getvalue()
        if content.strip():  # Only send non-empty messages
            # Split into lines
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    self._send_to_queue(line)

        # Clear buffer
        self.buffer = StringIO()

    def _send_to_queue(self, message: str):
        """Send message to log queue."""
        # Determine log level based on message content
        if "❌" in message or "ERROR" in message.upper() or "Error" in message:
            level = "ERROR"
        elif "⚠️" in message or "WARNING" in message.upper() or "WARN" in message.upper():
            level = "WARNING"
        else:
            level = "INFO"

        log_dict = {"level": level, "message": message, "timestamp": datetime.now(), "logger": "stdout"}

        try:
            self.log_queue.put_nowait(log_dict)
        except queue.Full:
            # If queue is full, drop oldest and try again
            try:
                self.log_queue.get_nowait()
                self.log_queue.put_nowait(log_dict)
            except (queue.Empty, Exception):
                pass

    def flush(self):
        """Flush any remaining buffer."""
        self._flush_buffer()
        try:
            self.original_stdout.flush()
        except Exception:
            pass

    def isatty(self):
        """Return whether this is a tty."""
        try:
            return self.original_stdout.isatty()
        except Exception:
            return False

    def fileno(self):
        """Return file descriptor number."""
        try:
            return self.original_stdout.fileno()
        except Exception:
            return -1

    @property
    def closed(self):
        """Check if stream is closed."""
        try:
            return self.original_stdout.closed
        except Exception:
            return False
