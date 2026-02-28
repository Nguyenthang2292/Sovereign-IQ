"""Periodic updater and threading management."""

import queue
import threading
import time

from modules.auto_trade.gui.utils.threading_utils import PeriodicUpdater
from modules.common.ui.logging import log_error, log_info


class UpdaterManager:
    """Manages periodic updaters and update queue processing."""

    def __init__(self, parent):
        self.parent = parent
        self.updaters = {}
        # Track next scheduled scan time for countdown display
        self._next_scan_time: float = 0.0
        # Track current known open positions count for status label
        self._open_positions_count: int = 0

    def setup_updaters(self):
        """Initialize and start all periodic updaters."""
        self.parent._update_queue = queue.Queue()

        # PeriodicUpdater runs in background thread; callbacks use queue
        self.updaters["signal"] = PeriodicUpdater(self.parent._thread_refresh_signals, interval=30)
        self.updaters["positions"] = PeriodicUpdater(self.parent._thread_refresh_positions, interval=10)
        self.updaters["account"] = PeriodicUpdater(self.parent._thread_refresh_account, interval=15)
        self.updaters["stats"] = PeriodicUpdater(self.parent._thread_refresh_stats, interval=60)

        # NOTE: PeriodicUpdater._run() fires the callback immediately on first tick,
        # so we do NOT call refresh_all() here — that would block the main thread.
        # Each updater runs its first callback in its own background thread.
        self.updaters["signal"].start()
        self.updaters["positions"].start()
        self.updaters["account"].start()
        self.updaters["stats"].start()
        self.parent.after(100, self._drain_update_queue)

        # Start log streaming updater
        self.parent.after(100, self._drain_log_queue)

        # Start 1-second ticker for scanner countdown
        self.parent.after(1000, self._update_scanner_status_tick)

        # One-shot startup reconcile: close stale OPEN positions in DB that
        # are no longer on Binance.  Runs in a background thread with a short
        # delay so the GUI paints first.  This is independent of auto-trade.
        if getattr(self.parent, "mode", "DRY_RUN") != "DRY_RUN":
            self.parent.after(3000, self._startup_reconcile)

    def _drain_update_queue(self):
        """Process UI updates from background thread (must run on main thread)."""
        try:
            while True:
                kind, data = self.parent._update_queue.get_nowait()
                if kind == "signals":
                    self.parent.signals_frame.update_signals(data)
                elif kind == "positions":
                    print(f"[UpdateQueue] Processing 'positions' update: {len(data) if data else 0} items")
                    self.parent.positions_frame.update_positions(data)
                    # Update open position count and refresh scanner status in real-time
                    new_count = len(data) if data else 0
                    if new_count != self._open_positions_count:
                        self._open_positions_count = new_count
                        self._refresh_scanner_status_label()
                elif kind == "account" and data:
                    self.parent.account_frame.update_data(data)
                elif kind == "stats" and data:
                    self.parent.stats_frame.update_data(data)
                elif kind == "scanner_done":
                    if hasattr(self.parent, "scanner_control"):
                        self.parent.scanner_control.update_last_scan_time()
                    # Record the time when next scan will run
                    sm = getattr(self.parent, "scanner_manager", None)
                    if sm and sm.updater:
                        config = self.parent.settings_manager.get("scanner", {})
                        interval_s = config.get("scan_interval", 5) * 60
                        self._next_scan_time = time.monotonic() + interval_s
                    else:
                        self._next_scan_time = 0.0
                    # Refresh scanner status label with the current state
                    self._refresh_scanner_status_label()
        except queue.Empty:
            pass
        self.parent.after(100, self._drain_update_queue)

    def _refresh_scanner_status_label(self):
        """Update scanner_status_label to reflect current positions / scanner state."""
        if not hasattr(self.parent, "scanner_status_label"):
            return
        sm = getattr(self.parent, "scanner_manager", None)
        is_running = sm is not None and sm.updater is not None
        if not is_running:
            self.parent.scanner_status_label.configure(text="🔴 Scanner: STOPPED", text_color="gray")
            return
        n = self._open_positions_count
        if n > 0:
            pos_word = "position" if n == 1 else "positions"
            self.parent.scanner_status_label.configure(
                text=f"🟢 Scanner: RUNNING (scan skipped – {n} open {pos_word})",
                text_color="#00ff88",
            )
        else:
            self.parent.scanner_status_label.configure(
                text="🟢 Scanner: RUNNING",
                text_color="#00ff88",
            )

    def _update_scanner_status_tick(self):
        """1-second ticker: refresh scanner status label + countdown to next scan."""
        try:
            self._refresh_scanner_status_label()
            # Update countdown label
            if hasattr(self.parent, "scanner_countdown_label"):
                sm = getattr(self.parent, "scanner_manager", None)
                is_running = sm is not None and sm.updater is not None
                if is_running and self._next_scan_time > 0:
                    remaining = self._next_scan_time - time.monotonic()
                    if remaining > 0:
                        mins, secs = divmod(int(remaining), 60)
                        if mins > 0:
                            countdown_text = f" Next scan in: {mins}m {secs:02d}s"
                        else:
                            countdown_text = f" Next scan in: {secs}s"
                        self.parent.scanner_countdown_label.configure(text=countdown_text, text_color="#aaaaaa")
                    else:
                        self.parent.scanner_countdown_label.configure(text="Scanning now...", text_color="#00ff88")
                elif is_running:
                    # Scanner running but we don't know next scan time yet
                    self.parent.scanner_countdown_label.configure(text="", text_color="#aaaaaa")
                else:
                    self.parent.scanner_countdown_label.configure(text="", text_color="#aaaaaa")
        except Exception as e:
            print(f"Error in scanner status tick: {e}")
        self.parent.after(1000, self._update_scanner_status_tick)

    MAX_LOG_LINES = 500

    def _drain_log_queue(self):
        """Process log messages from log_queue and display in logs_viewer or logs_textbox."""
        try:
            if not hasattr(self.parent, "log_queue"):
                self.parent.after(100, self._drain_log_queue)
                return
            while not self.parent.log_queue.empty():
                try:
                    log_record = self.parent.log_queue.get_nowait()
                    if isinstance(log_record, dict):
                        level = log_record.get("level", "INFO")
                        msg = log_record.get("message", "")
                        log_msg = f"[{level}] {msg}"
                    else:
                        # Fallback for raw LogRecord objects
                        log_msg = (
                            f"[{getattr(log_record, 'levelname', 'INFO')}] {getattr(log_record, 'getMessage', lambda: (
                                    str(log_record)
                                ))()}"
                        )
                    if hasattr(self.parent, "logs_viewer"):
                        self.parent.logs_viewer.append_log(log_msg)
                    elif hasattr(self.parent, "logs_textbox"):
                        self._append_log_to_textbox(log_msg)
                except queue.Empty:
                    break
        except Exception as e:
            print(f"Error draining log queue: {e}")

        self.parent.after(100, self._drain_log_queue)

    def _append_log_to_textbox(self, log_message: str):
        """Append log to parent.logs_textbox (used when layout has no LogsViewer)."""
        try:
            tb = self.parent.logs_textbox
            tb.configure(state="normal")
            tb.insert("end", log_message + "\n")
            lines = int(tb.index("end-1c").split(".")[0])
            if lines > self.MAX_LOG_LINES:
                tb.delete("1.0", f"{lines - self.MAX_LOG_LINES}.0")
            tb.see("end")
            tb.configure(state="disabled")
        except Exception as e:
            print(f"Error appending log to textbox: {e}")

    def _startup_reconcile(self):
        """One-shot reconcile: sync Binance positions and close stale DB entries.

        Runs in a background thread so the GUI stays responsive.
        """

        def _run():
            try:
                from modules.auto_trade.execution.binance_client import BinanceClient
                from modules.auto_trade.gui.utils.position_sync_service import PositionSyncService

                ds = self.parent.data_service
                api_key = getattr(ds, "api_key", "") or ""
                api_secret = getattr(ds, "api_secret", "") or ""
                if not api_key or not api_secret:
                    return

                testnet = getattr(ds, "testnet", False)
                client = BinanceClient(
                    api_key=api_key,
                    api_secret=api_secret,
                    testnet=testnet,
                    dry_run=False,
                )
                stats = PositionSyncService.sync_all_positions(client)
                closed = stats.get("closed", 0)
                if closed:
                    log_info(f"[Startup] Reconcile closed {closed} stale DB position(s)")
                    # Refresh positions frame so the GUI removes stale entries
                    if hasattr(self.parent, "updater_manager") and "positions" in self.updaters:
                        self.updaters["positions"].trigger()
            except Exception as exc:
                log_error(f"[Startup] Reconcile error: {exc}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

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

    def create_reconcile_updater(self, callback, interval=3600):
        """Create and start Binance↔DB reconcile updater."""
        updater = PeriodicUpdater(callback, interval=interval)
        updater.start()
        self.updaters["reconcile"] = updater
        return updater

    def stop_updater(self, name):
        """Stop a specific updater by name."""
        if name in self.updaters:
            self.updaters[name].stop()

    def create_trailing_stop_updater(self, callback, interval=30):
        """Create and start trailing stop updater."""
        updater = PeriodicUpdater(callback, interval=interval)
        updater.start()
        self.updaters["trailing_stop"] = updater
        return updater

    def create_negative_breakeven_updater(self, callback, interval=30):
        """Create and start negative breakeven updater."""
        updater = PeriodicUpdater(callback, interval=interval)
        updater.start()
        self.updaters["negative_breakeven"] = updater
        return updater

    def create_ensure_tp_sl_updater(self, callback, interval=60):
        """Create and start ensure TP/SL updater (add missing TP/SL for open AUTO positions)."""
        updater = PeriodicUpdater(callback, interval=interval)
        updater.start()
        self.updaters["ensure_tp_sl"] = updater
        return updater

    def create_auto_close_timer_updater(self, callback, interval=30):
        """Create and start auto-close timer updater."""
        updater = PeriodicUpdater(callback, interval=interval)
        updater.start()
        self.updaters["auto_close_timer"] = updater
        return updater
