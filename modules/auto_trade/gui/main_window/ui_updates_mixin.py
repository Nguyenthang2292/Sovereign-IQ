"""UI update and refresh helpers for Auto Trade Dashboard."""

from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.modes import TradingMode
from modules.common.ui.logging import log_debug


class UIUpdatesMixin:
    """Provide UI update helpers and non-blocking refresh triggers."""

    def _update_mode_display(self):
        """Update mode indicator in stats frame and header."""
        mode_colors = {
            TradingMode.PRODUCTION: Colors.PRODUCTION,
            TradingMode.DEMO: Colors.DEMO,
            TradingMode.DRY_RUN: Colors.DRY_RUN,
        }

        mode_color = mode_colors.get(self.mode, Colors.DRY_RUN)
        mode_text = self.mode.replace("_", " ")

        if hasattr(self, "stats_frame"):
            self.stats_frame.mode_indicator.destroy()
            from modules.auto_trade.gui.components.stats_frame import ModeIndicator

            self.stats_frame.mode_indicator = ModeIndicator(self.stats_frame, self.mode)
            self.stats_frame.mode_indicator.pack(pady=(0, 10))

        if hasattr(self, "header_mode_label") and self.header_mode_label is not None:
            self.header_mode_label.configure(text=f"[{mode_text}]", text_color=mode_color)

    def _update_timestamp(self):
        """Update last update timestamp."""
        from datetime import datetime

        timestamp = datetime.now()
        time_str = timestamp.strftime("%H:%M:%S")

        def _do_update():
            # self.last_update_label is declared as None in __init__ and may never
            # be assigned if the layout doesn't create a standalone label.
            # The actual widget lives inside StatusBar, so check both locations.
            lbl = self.last_update_label
            if lbl is None and hasattr(self, "status_bar"):
                lbl = getattr(self.status_bar, "last_update_label", None)
            if lbl is not None:
                lbl.configure(text=f"Last update: {time_str}")

        self.after(0, _do_update)

        if hasattr(self, "status_bar"):
            self.after(0, lambda: self.status_bar.set_last_update(timestamp))

    def _thread_refresh_signals(self):
        """Thread-safe signal refresh."""
        signals = self.data_service.get_signals()
        self._update_queue.put(("signals", signals))

    def _thread_refresh_positions(self):
        """Thread-safe positions refresh (runs in PeriodicUpdater background thread)."""
        log_debug("[MainWindow] _thread_refresh_positions called")
        positions = self.data_service.get_positions()
        log_debug(f"[MainWindow] get_positions() returned {len(positions) if positions else 0} positions")
        self._update_queue.put(("positions", positions))

    def _thread_refresh_account(self):
        """Thread-safe account refresh."""
        data = self.data_service.get_account_data()
        self._update_queue.put(("account", data))

    def _thread_refresh_stats(self):
        """Thread-safe stats refresh."""
        stats = self.data_service.get_quick_stats()
        self._update_queue.put(("stats", stats))

    def _update_connection_status(self):
        """Update status bar connection status based on WebSocket state."""
        if hasattr(self, "status_bar") and hasattr(self, "ws_data_service"):
            is_connected = self.ws_data_service.is_connected
            self.status_bar.set_connection_status(is_connected)

    def refresh_signals(self):
        """Trigger background signal refresh (non-blocking, result delivered via update queue)."""
        if hasattr(self, "updater_manager") and "signal" in self.updater_manager.updaters:
            self.updater_manager.updaters["signal"].trigger()
        self._update_connection_status()

    def refresh_positions(self):
        """Trigger background positions refresh (non-blocking, result delivered via update queue)."""
        if hasattr(self, "updater_manager") and "positions" in self.updater_manager.updaters:
            self.updater_manager.updaters["positions"].trigger()
        self._update_connection_status()

    def refresh_account(self):
        """Trigger background account refresh (non-blocking, result delivered via update queue)."""
        if hasattr(self, "updater_manager") and "account" in self.updater_manager.updaters:
            self.updater_manager.updaters["account"].trigger()

    def refresh_stats(self):
        """Trigger background stats refresh (non-blocking, result delivered via update queue)."""
        if hasattr(self, "updater_manager") and "stats" in self.updater_manager.updaters:
            self.updater_manager.updaters["stats"].trigger()
