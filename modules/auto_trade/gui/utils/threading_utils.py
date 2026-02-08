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

    Useful for updating GUI elements with fresh data at regular intervals.
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

    def start(self) -> None:
        """Start the periodic updater thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        """Stop the periodic updater thread."""
        self.running = False

    def _run(self) -> None:
        """Internal loop that runs in background thread."""
        while self.running:
            try:
                self.callback()
            except Exception as e:
                print(f"Error in periodic update: {e}")
            time.sleep(self.interval)
