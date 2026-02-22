"""
Threading Utilities Module

Provides thread-safe utilities for periodic updates and background tasks.
"""

import threading
import time
from typing import Callable, Optional


class PeriodicUpdater:
    """
    Executes a callback function periodically in a background thread.

    Overlap-safe: if *callback* takes longer than *interval*, the next
    scheduled tick is skipped rather than spawning a second concurrent
    execution.  Stop signals are honoured within 0.5 s.
    """

    def __init__(self, callback: Callable[[], None], interval: int = 30) -> None:
        """
        Initialize periodic updater.

        Args:
            callback: Function to call periodically
            interval: Interval in seconds between calls (default: 30)
        """
        self.callback: Callable[[], None] = callback
        self.interval: int = interval
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        # Prevent overlapping executions when callback takes > interval seconds
        self._executing = threading.Lock()

    def start(self) -> None:
        """Start the periodic updater thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        """Stop the periodic updater thread."""
        self.running = False

    def trigger(self) -> None:
        """Manually trigger the callback immediately (non-blocking)."""
        t = threading.Thread(target=self._safe_call, daemon=True)
        t.start()

    def _safe_call(self) -> None:
        """Call callback if not already executing."""
        if self._executing.acquire(blocking=False):
            try:
                self.callback()
            except Exception as exc:
                print(f"Error in periodic update ({self.callback.__name__}): {exc}")
            finally:
                self._executing.release()
        # else: already running, skip this tick silently

    def _run(self) -> None:
        """Internal loop that runs in background thread."""
        _TICK = 0.5  # check stop() responsiveness
        elapsed = 0.0
        first_run = True
        while self.running:
            if first_run or elapsed >= self.interval:
                first_run = False
                elapsed = 0.0
                self._safe_call()
            time.sleep(_TICK)
            elapsed += _TICK
