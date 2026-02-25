"""
Stdout Redirector for GUI Logging

Captures stdout (print statements) and redirects to GUI log queue.
Keeps writing to the original console simultaneously (tee behaviour).
"""

import queue
import sys
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Union


class _BinaryBufferProxy:
    """Binary-compatible proxy for sys.stdout.buffer / sys.stderr.buffer.

    Some libraries (colorama, tqdm, etc.) call sys.stdout.buffer.write(bytes)
    directly.  This proxy delegates those byte-writes to the parent redirector
    so they appear in the GUI log as decoded text.
    """

    def __init__(self, redirector: "StdoutRedirector") -> None:
        self._redirector = redirector

    def write(self, data: bytes) -> int:
        if isinstance(data, (bytes, bytearray)):
            text = data.decode(self._redirector.encoding, errors=self._redirector.errors)
        else:
            text = str(data)
        self._redirector.write(text)
        return len(data)

    def flush(self) -> None:
        self._redirector.flush()

    def fileno(self) -> int:
        return self._redirector.fileno()

    @property
    def closed(self) -> bool:
        return self._redirector.closed


class StdoutRedirector:
    """Redirects stdout to both console and GUI log queue (tee).

    Usage::
        original = sys.stdout
        sys.stdout = StdoutRedirector(log_queue, original_stdout=original)
    """

    def __init__(self, log_queue: queue.Queue, original_stdout: Any = None) -> None:
        self.log_queue: queue.Queue = log_queue
        self.original_stdout: Any = original_stdout or sys.stdout

        # Store encoding/errors from original stream
        self.encoding: str = getattr(self.original_stdout, "encoding", "utf-8") or "utf-8"
        self.errors: str = getattr(self.original_stdout, "errors", "replace") or "replace"

        # Internal line buffer (text-mode)
        self._line_buf: StringIO = StringIO()

        # Binary buffer proxy — satisfies sys.stdout.buffer accesses
        self.buffer: _BinaryBufferProxy = _BinaryBufferProxy(self)

    # ── Text-mode interface ───────────────────────────────────────────────────

    def write(self, message: Union[str, bytes]) -> None:
        """Write message to both original stdout and log queue."""
        # Decode bytes if necessary
        if isinstance(message, (bytes, bytearray)):
            try:
                message = message.decode(self.encoding, errors=self.errors)
            except Exception:
                message = repr(message)

        if not isinstance(message, str):
            message = str(message)

        # Mirror to original console (best-effort)
        try:
            self.original_stdout.write(message)
            self.original_stdout.flush()
        except Exception:
            pass

        # Buffer until newline, then flush entire line(s) to the queue
        self._line_buf.write(message)
        if "\n" in message:
            self._flush_line_buf()

    def _flush_line_buf(self) -> None:
        """Send complete lines from the internal buffer to the log queue."""
        content: str = self._line_buf.getvalue()
        self._line_buf = StringIO()

        if not content.strip():
            return

        for raw_line in content.split("\n"):
            line = raw_line.strip()
            if line:
                self._enqueue(line)

    def _enqueue(self, message: str) -> None:
        """Put a log dict into the queue (non-blocking, drop-oldest on full)."""
        # Infer level from message content
        msg_upper = message.upper()
        if "ERROR" in msg_upper or "❌" in message:
            level = "ERROR"
        elif "WARN" in msg_upper or "⚠" in message:
            level = "WARNING"
        elif "DEBUG" in msg_upper:
            level = "DEBUG"
        else:
            level = "INFO"

        log_dict: Dict[str, Any] = {
            "level": level,
            "message": message,
            "timestamp": datetime.now(),
            "logger": "stdout",
        }

        try:
            self.log_queue.put_nowait(log_dict)
        except queue.Full:
            try:
                self.log_queue.get_nowait()
                self.log_queue.put_nowait(log_dict)
            except (queue.Empty, Exception):
                pass

    def flush(self) -> None:
        """Flush remaining buffer."""
        self._flush_line_buf()
        try:
            self.original_stdout.flush()
        except Exception:
            pass

    # ── Standard stream interface ─────────────────────────────────────────────

    def isatty(self) -> bool:
        try:
            return bool(self.original_stdout.isatty())
        except Exception:
            return False

    def fileno(self) -> int:
        try:
            return int(self.original_stdout.fileno())
        except Exception:
            return -1

    @property
    def closed(self) -> bool:
        try:
            return bool(self.original_stdout.closed)
        except Exception:
            return False
