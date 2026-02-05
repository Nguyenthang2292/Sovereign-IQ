"""Periodic updater and threading management."""

import queue

from gui.utils.threading_utils import PeriodicUpdater


class UpdaterManager:
    """Manages periodic updaters and update queue processing."""

    def __init__(self, parent):
        self.parent = parent
        self.updaters = {}

    def setup_updaters(self):
        """Initialize and start all periodic updaters."""
        self.parent._update_queue = queue.Queue()

        def refresh_all():
            self.parent.refresh_signals()
            self.parent.refresh_positions()
            self.parent.refresh_account()
            self.parent.refresh_stats()
            self.parent._update_timestamp()

        # PeriodicUpdater runs in background thread; callbacks use queue
        self.updaters["signal"] = PeriodicUpdater(self.parent._thread_refresh_signals, interval=30)
        self.updaters["stats"] = PeriodicUpdater(self.parent._thread_refresh_stats, interval=60)

        refresh_all()

        self.updaters["signal"].start()
        self.updaters["stats"].start()
        self.parent.after(100, self._drain_update_queue)

        # Start log streaming updater
        self.parent.after(100, self._drain_log_queue)

    def _drain_update_queue(self):
        """Process UI updates from background thread (must run on main thread)."""
        try:
            while True:
                kind, data = self.parent._update_queue.get_nowait()
                if kind == "signals":
                    self.parent.signals_frame.update_signals(data)
                elif kind == "positions":
                    self.parent.positions_frame.update_positions(data)
                elif kind == "account" and data:
                    self.parent.account_frame.update_data(data)
                elif kind == "stats" and data:
                    self.parent.stats_frame.update_data(data)
                elif kind == "scanner_done":
                    if hasattr(self.parent, "scanner_control"):
                        self.parent.scanner_control.update_last_scan_time()
        except queue.Empty:
            pass
        self.parent.after(100, self._drain_update_queue)

    def _drain_log_queue(self):
        """Process log messages from log_queue and display in logs_viewer."""
        try:
            if hasattr(self.parent, "logs_viewer") and hasattr(self.parent, "log_queue"):
                while not self.parent.log_queue.empty():
                    try:
                        log_record = self.parent.log_queue.get_nowait()
                        # Format log message
                        log_msg = f"[{log_record.levelname}] {log_record.getMessage()}"
                        self.parent.logs_viewer.append_log(log_msg)
                    except queue.Empty:
                        break
        except Exception as e:
            print(f"Error draining log queue: {e}")

        # Schedule next check
        self.parent.after(100, self._drain_log_queue)

    def stop_all(self):
        """Stop all periodic updaters."""
        for updater in self.updaters.values():
            updater.stop()

    def create_auto_trade_updater(self, callback, interval=60):
        """Create and start auto-trade updater."""
        updater = PeriodicUpdater(callback, interval=interval)
        updater.start()
        self.updaters["auto_trade"] = updater
        return updater

    def create_scanner_updater(self, callback, interval=300):
        """Create and start scanner updater."""
        updater = PeriodicUpdater(callback, interval=interval)
        updater.start()
        self.updaters["scanner"] = updater
        return updater

    def stop_updater(self, name):
        """Stop a specific updater by name."""
        if name in self.updaters:
            self.updaters[name].stop()
