"""
Market Scanner Scheduler Module

Schedules market scans and signal generation when no positions are open.
Triggers signal pipeline every 5 minutes and executes orders if signals are found.
"""

import time
from datetime import datetime
from threading import Event, Thread
from typing import Callable, Optional

from modules.common.ui.logging import log_error, log_info, log_warn


class ScannerScheduler:
    """
    Schedules periodic market scans when no positions are open.

    Example:
        >>> scheduler = ScannerScheduler(
        ...     scan_callback=run_signal_pipeline,
        ...     execute_callback=execute_order,
        ...     position_check_callback=check_positions,
        ...     scan_interval=300
        ... )
        >>> scheduler.start()
    """

    def __init__(
        self,
        scan_callback: Callable[[], Optional[object]],
        execute_callback: Callable[[object], bool],
        position_check_callback: Callable[[], bool],
        scan_interval: float = 300.0,  # 5 minutes
        enabled: bool = True,
    ):
        """
        Initialize ScannerScheduler.

        Args:
            scan_callback: Function to run signal pipeline, returns signal or None
            execute_callback: Function to execute signal, returns True if successful
            position_check_callback: Function to check if positions are open, returns True if open
            scan_interval: Scan interval in seconds (default: 300 = 5 minutes)
            enabled: Start enabled or disabled
        """
        self.scan_callback = scan_callback
        self.execute_callback = execute_callback
        self.position_check_callback = position_check_callback
        self.scan_interval = scan_interval
        self.enabled = enabled

        self._running = False
        self._stop_event = Event()
        self._scheduler_thread: Optional[Thread] = None
        self._last_scan_time: Optional[datetime] = None
        self._scan_count = 0
        self._signal_count = 0
        self._execution_count = 0

        log_info(f"ScannerScheduler initialized (interval={scan_interval}s, enabled={enabled})")

    def start(self):
        """Start the scheduler."""
        if self._running:
            log_warn("ScannerScheduler is already running")
            return

        self._running = True
        self._stop_event.clear()
        self._scheduler_thread = Thread(target=self._schedule_loop, daemon=True)
        self._scheduler_thread.start()
        log_info("✅ ScannerScheduler started")

    def stop(self):
        """Stop the scheduler."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=10)
        log_info("⏹️ ScannerScheduler stopped")

    def enable(self):
        """Enable scheduled scans."""
        self.enabled = True
        log_info("ScannerScheduler enabled")

    def disable(self):
        """Disable scheduled scans."""
        self.enabled = False
        log_info("ScannerScheduler disabled")

    def trigger_scan_now(self):
        """Manually trigger a scan immediately."""
        log_info("Manual scan triggered")
        self._perform_scan()

    def _schedule_loop(self):
        """Main scheduling loop."""
        log_info("Scheduler loop started")

        while self._running and not self._stop_event.is_set():
            try:
                # Check if scan should run
                if self.enabled and self._should_scan():
                    self._perform_scan()

            except Exception as e:
                log_error(f"Error in scheduler loop: {e}", exc_info=True)

            # Wait for next interval
            self._stop_event.wait(timeout=self.scan_interval)

        log_info("Scheduler loop stopped")

    def _should_scan(self) -> bool:
        """
        Check if a scan should be performed.

        Returns:
            True if scan should run, False otherwise
        """
        # Check if positions are open
        try:
            has_positions = self.position_check_callback()
            if has_positions:
                log_info("Positions are open, skipping scan")
                return False
        except Exception as e:
            log_error(f"Error checking positions: {e}")
            return False

        # Check if enough time has passed since last scan
        if self._last_scan_time:
            elapsed = (datetime.now() - self._last_scan_time).total_seconds()
            if elapsed < self.scan_interval:
                return False

        return True

    def _perform_scan(self):
        """Perform a market scan and execute if signal found."""
        self._scan_count += 1
        self._last_scan_time = datetime.now()

        log_info(f"🔍 Running market scan #{self._scan_count}...")

        try:
            # Run signal pipeline
            signal = self.scan_callback()

            if signal:
                self._signal_count += 1
                log_info(f"✅ Signal found: {signal}")

                # Execute signal
                log_info("Executing signal...")
                success = self.execute_callback(signal)

                if success:
                    self._execution_count += 1
                    log_info(f"✅ Signal executed successfully (#{self._execution_count})")
                else:
                    log_error("❌ Signal execution failed")
            else:
                log_info("No signal generated from scan")

        except Exception as e:
            log_error(f"Error during scan: {e}", exc_info=True)

    def get_stats(self) -> dict:
        """
        Get scheduler statistics.

        Returns:
            Dict with scan count, signal count, execution count, etc.
        """
        return {
            "total_scans": self._scan_count,
            "signals_found": self._signal_count,
            "executions": self._execution_count,
            "last_scan": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "enabled": self.enabled,
            "running": self._running,
            "scan_interval_seconds": self.scan_interval,
        }

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    @property
    def is_enabled(self) -> bool:
        """Check if scheduler is enabled."""
        return self.enabled

    @property
    def last_scan_time(self) -> Optional[datetime]:
        """Get timestamp of last scan."""
        return self._last_scan_time
