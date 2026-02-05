"""Main Auto Trade Dashboard Window - orchestrates all components."""

import logging
import logging.handlers
import queue
import sys
from pathlib import Path
from typing import Optional

import customtkinter as ctk

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from gui.utils.colors import Colors
from gui.utils.data_service import DataService
from gui.utils.modes import TradingMode
from gui.utils.settings_manager import SettingsManager
from gui.utils.websocket_data_service import WebSocketDataService

from .auto_trade import AutoTradeManager
from .layout import LayoutManager
from .position_actions import PositionActionHandler
from .risk_manager import RiskManager
from .scanner import ScannerManager
from .settings_handler import SettingsHandler
from .updaters import UpdaterManager
from .websocket_handler import WebSocketHandler


class AutoTradeDashboard(ctk.CTk):
    """Main Auto Trade Dashboard application window."""

    def __init__(self):
        super().__init__()

        self.settings_manager = SettingsManager()
        self.settings_manager.load()

        self.mode = self.settings_manager.get("api.mode", TradingMode.DRY_RUN)

        # Initialize data services
        self.data_service = DataService(mode=self.mode)
        self.ws_data_service = WebSocketDataService(mode=self.mode, settings_manager=self.settings_manager)

        self.title(f"Auto Trade Dashboard - [{self.mode}]")
        self.geometry("1200x800")
        self.minsize(800, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._update_queue = queue.Queue()

        # Create log queue for GUI logging (stream to textbox)
        self.log_queue = queue.Queue(maxsize=500)

        # Set up file-based logging for GUI (use absolute path)
        self.log_file_path = Path("logs/auto_trade_gui.log").absolute()
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._setup_file_logging()

        # Initialize managers
        self.layout_manager = LayoutManager(self)
        self.updater_manager = UpdaterManager(self)
        self.websocket_handler = WebSocketHandler(self)
        self.auto_trade_manager = AutoTradeManager(self)
        self.risk_manager = RiskManager(self)
        self.settings_handler = SettingsHandler(self)
        self.scanner_manager = ScannerManager(self)
        self.position_action_handler = PositionActionHandler(self)

        # Create UI and start services
        self.layout_manager.create_layout()

        # Test log message after UI creation
        logging.info("GUI layout created successfully")
        logging.info("LogsViewer should now be active and reading from log file")

        self.updater_manager.setup_updaters()
        self.websocket_handler.register_callbacks()
        self.settings_handler.apply_settings()

        # Start WebSocket service
        if self.mode != TradingMode.DRY_RUN:
            self.ws_data_service.start()
            logging.info(f"WebSocket service started (mode={self.mode})")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _setup_file_logging(self):
        """Set up file-based logging that captures all logs from all modules."""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Remove existing file handlers
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                root_logger.removeHandler(handler)

        # Create file handler
        file_handler = logging.FileHandler(self.log_file_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(file_handler)

        # Create queue handler for GUI streaming
        queue_handler = logging.handlers.QueueHandler(self.log_queue)
        queue_handler.setLevel(logging.INFO)  # Only stream INFO and above to GUI
        root_logger.addHandler(queue_handler)

        logging.info("=" * 60)
        logging.info("AUTO TRADE DASHBOARD")
        logging.info(f"Mode: {self.mode}")
        logging.info(f"Log file: {self.log_file_path}")
        logging.info("=" * 60)

    # ==================== UI Update Methods ====================

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
            from gui.components.stats_frame import ModeIndicator

            self.stats_frame.mode_indicator = ModeIndicator(self.stats_frame, self.mode)
            self.stats_frame.mode_indicator.pack(pady=(0, 10))

        if hasattr(self, "header_mode_label"):
            self.header_mode_label.configure(text=f"[{mode_text}]", text_color=mode_color)

    def _update_timestamp(self):
        """Update last update timestamp."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.after(0, lambda: self.last_update_label.configure(text=f"Last update: {timestamp}"))

    def _thread_refresh_signals(self):
        """Thread-safe signal refresh."""
        signals = self.data_service.get_signals()
        self._update_queue.put(("signals", signals))

    def _thread_refresh_account(self):
        """Thread-safe account refresh."""
        data = self.data_service.get_account_data()
        self._update_queue.put(("account", data))

    def _thread_refresh_stats(self):
        """Thread-safe stats refresh."""
        stats = self.data_service.get_quick_stats()
        self._update_queue.put(("stats", stats))

    def refresh_signals(self):
        """Refresh signals display."""
        signals = self.data_service.get_signals()
        self.after(0, lambda: self.signals_frame.update_signals(signals))

    def refresh_positions(self):
        """Refresh positions display."""
        positions = self.data_service.get_positions()
        self.after(0, lambda: self.positions_frame.update_positions(positions))

    def refresh_account(self):
        """Refresh account display."""
        data = self.data_service.get_account_data()
        if data:
            self.after(0, lambda: self.account_frame.update_data(data))

    def refresh_stats(self):
        """Refresh stats display."""
        stats = self.data_service.get_quick_stats()
        if stats:
            self.after(0, lambda: self.stats_frame.update_data(stats))

    # ==================== Callback Handlers ====================

    def on_trade_executed(self):
        """Callback when manual trade is executed."""
        logging.info("Trade executed! Refreshing positions...")
        self.refresh_positions()
        self.refresh_account()

    def on_auto_trade_toggle(self, enabled: bool):
        """Callback when auto-trade is toggled."""
        logging.info(f"Auto-trade {'enabled' if enabled else 'disabled'}")
        if enabled:
            self.auto_trade_manager.start()
        else:
            self.auto_trade_manager.stop()

    def on_settings_change(self, setting_type: str, value=None):
        """Handle settings change from ConfigPanel."""
        self.settings_handler.handle_settings_change(setting_type, value)

    def _restart_websocket_service(self):
        """Restart WebSocket service with updated credentials/mode."""
        try:
            if hasattr(self, "ws_data_service") and self.ws_data_service:
                logging.info("Stopping existing WebSocket service...")
                self.ws_data_service.stop()

            self.settings_manager.load()

            logging.info(f"Creating new WebSocket service (mode={self.mode})...")
            self.ws_data_service = WebSocketDataService(mode=self.mode, settings_manager=self.settings_manager)

            if self.mode != TradingMode.DRY_RUN:
                self.websocket_handler.register_callbacks()
                self.ws_data_service.start()
                logging.info("WebSocket service restarted successfully")
            else:
                logging.info("DRY_RUN mode - WebSocket not started")

        except Exception as e:
            logging.error(f"Error restarting WebSocket service: {e}")
            import traceback

            traceback.print_exc()

    def _refresh_theme_colors(self):
        """Refresh all component colors when theme changes."""
        self.settings_handler.refresh_theme_colors()

    def on_scan_toggle(self, action):
        """Handle scanner start/stop from ScannerControl."""
        self.scanner_manager.handle_scan_toggle(action)

    def on_scanner_config_change(self, config: dict):
        """Handle scanner configuration change."""
        self.scanner_manager.handle_config_change(config)

    def on_recovery_config_change(self, event_type: str, data):
        """Handle recovery configuration change."""
        try:
            logging.info(f"Recovery {event_type}: {data}")

            if event_type == "recovery_started":
                self.settings_manager.set("recovery.enabled", True)
                self.settings_manager.set("recovery.config", data)
                self.settings_manager.save()
            elif event_type == "recovery_reset":
                self.settings_manager.set("recovery.enabled", False)
                self.settings_manager.save()
            elif event_type == "recovery_alert":
                if hasattr(self, "status_label"):
                    self.status_label.configure(text=f"Recovery: {data}")

        except Exception as e:
            logging.error(f"Error handling recovery config change: {e}")

    def on_position_action(self, action_data: dict):
        """Handle position actions from GUI."""
        return self.position_action_handler.handle_action(action_data)

    # ==================== Lifecycle ====================

    def on_closing(self):
        """Handle application shutdown."""
        try:
            if hasattr(self, "settings_manager"):
                self.settings_manager.save()
                logging.info("Settings saved on exit")
        except Exception as e:
            logging.error(f"Error saving settings: {e}")

        if hasattr(self, "ws_data_service"):
            self.ws_data_service.stop()
            logging.info("WebSocket service stopped")

        self.updater_manager.stop_all()
        self.auto_trade_manager.stop()
        self.scanner_manager._stop_scanner()

        self.destroy()
