"""Main Auto Trade Dashboard Window - orchestrates all components."""

import logging
import logging.handlers
import queue
import sys
from pathlib import Path
from typing import Any

import customtkinter as ctk

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from modules.auto_trade.gui.components.status_bar import StatusBar
from modules.auto_trade.gui.dialogs import ShortcutsHelpDialog
from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.data_service import DataService
from modules.auto_trade.gui.utils.modes import TradingMode
from modules.auto_trade.gui.utils.settings_manager import SettingsManager
from modules.auto_trade.gui.utils.shortcuts import is_editable_focus
from modules.auto_trade.gui.utils.websocket_data_service import WebSocketDataService

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

        self._update_queue: queue.Queue = queue.Queue()

        # Create log queue for GUI logging (stream to textbox)
        self.log_queue: queue.Queue = queue.Queue(maxsize=500)

        # Set up file-based logging for GUI (use absolute path)
        self.log_file_path = Path("logs/auto_trade_gui.log").absolute()
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._setup_file_logging()

        # Initialize EventSystem for position lifecycle events
        from modules.auto_trade.monitoring.event_system import EventSystem

        self.event_bus = EventSystem()

        # Initialize RecoveryManager for Gradual Recovery
        from modules.auto_trade.strategies.recovery_manager import RecoveryManager

        recovery_config = self.settings_manager.get("recovery", {})
        self.recovery_manager = RecoveryManager(
            event_bus=self.event_bus,
            config=recovery_config,
            enabled=recovery_config.get("enabled", False),
            database=True,  # Enable database persistence
        )
        self.recovery_manager.start()
        logging.info(f"RecoveryManager started (enabled={recovery_config.get('enabled', False)})")

        # Initialize managers
        self.layout_manager = LayoutManager(self)
        self.updater_manager = UpdaterManager(self)
        self.websocket_handler = WebSocketHandler(self)
        self.auto_trade_manager = AutoTradeManager(self)
        self.risk_manager = RiskManager(self)
        self.settings_handler = SettingsHandler(self)
        self.scanner_manager = ScannerManager(self)
        self.position_action_handler = PositionActionHandler(self)

        # Layout-assigned components (declared for type checker)
        self.config_panel: Any = None
        self.scanner_control: Any = None
        self.auto_trade_control: Any = None
        self.signals_frame: Any = None
        self.stats_frame: Any = None
        self.last_update_label: Any = None
        self.header_mode_label: Any = None
        self.positions_frame: Any = None
        self.account_frame: Any = None
        self.status_label: Any = None
        self.tabview: Any = None
        self.trade_form: Any = None
        self.database_panel: Any = None

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

        # Setup keyboard shortcuts
        self._setup_keyboard_shortcuts()

        # Create and add status bar (use grid to match root; layout uses row 0=header, 1=content, 2=status_frame)
        self.status_bar = StatusBar(self, mode=self.mode)
        self.grid_rowconfigure(3, weight=0)
        self.status_bar.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

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

    def _setup_keyboard_shortcuts(self):
        """Set up keyboard shortcuts (bind_all so they work regardless of focus)."""
        # Global
        self.bind_all("<F1>", lambda e: self._show_shortcuts_help())
        self.bind_all("<Control-r>", lambda e: self._handle_refresh())
        self.bind_all("<F5>", lambda e: self._handle_refresh())
        self.bind_all("<Escape>", lambda e: self._handle_escape())
        self.bind_all("<Control-s>", lambda e: self._handle_save())
        # Tab switching
        self.bind_all("<Control-Key-1>", lambda e: self._handle_tab_switch(0))
        self.bind_all("<Control-Key-2>", lambda e: self._handle_tab_switch(1))
        self.bind_all("<Control-Key-3>", lambda e: self._handle_tab_switch(2))
        self.bind_all("<Control-Key-4>", lambda e: self._handle_tab_switch(3))
        self.bind_all("<Control-Key-5>", lambda e: self._handle_tab_switch(4))
        # Context-aware (only when focus not in editable)
        self.bind_all("<Control-m>", lambda e: self._handle_manual_scan(e))
        self.bind_all("<Control-Return>", lambda e: self._handle_confirm_trade(e))
        self.bind_all("<Control-c>", lambda e: self._handle_copy_selection(e))

        logging.info(
            "Keyboard shortcuts: F1 (shortcuts help), Ctrl+R/F5, Esc, Ctrl+S, "
            "Ctrl+1..5 (tabs), Ctrl+M (scan), Ctrl+Enter (trade), Ctrl+C (copy in DB)"
        )

    def _handle_refresh(self):
        """Handle refresh keyboard shortcut."""
        logging.info("Refresh triggered by keyboard shortcut")
        self.refresh_signals()
        self.refresh_positions()
        self.refresh_account()
        self.refresh_stats()
        if hasattr(self, "status_bar"):
            self.status_bar.set_last_update()
        return "break"  # Prevent default behavior

    def _handle_escape(self):
        """Handle escape key - close any open dialogs."""
        # Close any open toplevel windows (dialogs)
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkToplevel):
                widget.destroy()
                logging.debug("Closed dialog via Escape key")
                break
        return "break"

    def _handle_save(self):
        """Handle Ctrl+S - apply and save settings."""
        self.on_apply_settings()
        return "break"

    def _show_shortcuts_help(self):
        """Open keyboard shortcuts help dialog. Called by F1 or header button."""
        ShortcutsHelpDialog(self)
        return "break"

    def _handle_tab_switch(self, index: int):
        """Switch to tab by index (0=Dashboard, 1=Scanner, 2=Trading, 3=Settings, 4=Database)."""
        tabs = ["Dashboard", "Scanner", "Trading", "Settings", "Database"]
        if 0 <= index < len(tabs) and hasattr(self, "tabview"):
            self.tabview.set(tabs[index])
        return "break"

    def _handle_manual_scan(self, event):
        """Trigger manual scan when Scanner tab is active and focus not in editable."""
        if is_editable_focus(self.focus_get()):
            return
        if hasattr(self, "tabview") and self.tabview.get() == "Scanner":
            self.on_scan_toggle("manual")
        return "break"

    def _handle_confirm_trade(self, event):
        """Open confirm trade dialog when Trading tab is active and focus not in editable."""
        if is_editable_focus(self.focus_get()):
            return
        if hasattr(self, "tabview") and self.tabview.get() == "Trading" and hasattr(self, "trade_form"):
            try:
                self.trade_form._confirm_trade()
            except Exception:
                pass
        return "break"

    def _handle_copy_selection(self, event):
        """Copy Data Viewer selection when Database tab is active and focus not in editable."""
        if is_editable_focus(self.focus_get()):
            return
        if hasattr(self, "tabview") and self.tabview.get() == "Database" and hasattr(self, "database_panel"):
            try:
                self.database_panel.copy_selection_to_clipboard()
            except Exception:
                pass
        return "break"

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
            from modules.auto_trade.gui.components.stats_frame import ModeIndicator

            self.stats_frame.mode_indicator = ModeIndicator(self.stats_frame, self.mode)
            self.stats_frame.mode_indicator.pack(pady=(0, 10))

        if hasattr(self, "header_mode_label"):
            self.header_mode_label.configure(text=f"[{mode_text}]", text_color=mode_color)

    def _update_timestamp(self):
        """Update last update timestamp."""
        from datetime import datetime

        timestamp = datetime.now()
        time_str = timestamp.strftime("%H:%M:%S")
        self.after(0, lambda: self.last_update_label.configure(text=f"Last update: {time_str}"))
        # Also update status bar
        if hasattr(self, "status_bar"):
            self.after(0, lambda: self.status_bar.set_last_update(timestamp))

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

    def _update_connection_status(self):
        """Update status bar connection status based on WebSocket state."""
        if hasattr(self, "status_bar") and hasattr(self, "ws_data_service"):
            is_connected = self.ws_data_service.is_connected
            self.status_bar.set_connection_status(is_connected)

    def refresh_signals(self):
        """Refresh signals display."""
        signals = self.data_service.get_signals()
        self.after(0, lambda: self.signals_frame.update_signals(signals))
        self._update_connection_status()

    def refresh_positions(self):
        """Refresh positions display."""
        positions = self.data_service.get_positions()
        self.after(0, lambda: self.positions_frame.update_positions(positions))
        self._update_connection_status()

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
                # Reload DataService credentials and refresh account so Dashboard shows real balance
                if hasattr(self.data_service, "_reload_credentials"):
                    self.data_service._reload_credentials()
                self.after(500, self.refresh_account)
            else:
                logging.info("DRY_RUN mode - WebSocket not started")

        except Exception as e:
            logging.error(f"Error restarting WebSocket service: {e}")
            import traceback

            traceback.print_exc()

    def _refresh_theme_colors(self):
        """Refresh all component colors when theme changes."""
        self.settings_handler.refresh_theme_colors()

    def _get_current_status(self):
        """Build status dict for Current Settings (database, api_mode, api_connection)."""
        status = {"api_mode": getattr(self, "mode", "DRY_RUN")}
        try:
            from modules.auto_trade.database import get_database_stats

            get_database_stats()
            status["database"] = "OK"
        except Exception:
            status["database"] = "Error"
        ws = getattr(self, "ws_data_service", None)
        if ws is None or status["api_mode"] == "DRY_RUN":
            status["api_connection"] = "N/A" if status["api_mode"] == "DRY_RUN" else "—"
        elif getattr(ws, "is_connected", False):
            status["api_connection"] = "Connected"
        else:
            status["api_connection"] = "Disconnected"
        return status

    def on_scan_toggle(self, action):
        """Handle scanner start/stop from ScannerControl."""
        self.scanner_manager.handle_scan_toggle(action)

    def on_scanner_config_change(self, config: dict):
        """Handle scanner configuration change."""
        self.scanner_manager.handle_config_change(config)

    def on_apply_settings(self):
        """Overwrite settings_manager from form (risk, filters, tp_sl, api, recovery) then save and apply."""
        try:
            if not hasattr(self, "config_panel"):
                return
            current = self.config_panel.get_settings()
            # Ensure Default Leverage always comes from form (avoid 10x when get_settings() returned defaults)
            if hasattr(self.config_panel, "default_leverage_var"):
                current.setdefault("risk", {})["default_leverage"] = self.config_panel.default_leverage_var.get()
            # Overwrite settings_manager with form values (ghi đè từ Apply Settings)
            for key in ("risk", "tp_sl", "api"):
                if key in current:
                    self.settings_manager.settings[key] = current[key]
            # Merge filters so we preserve keys not in form (e.g. timeframe)
            if "filters" in current:
                existing = self.settings_manager.settings.get("filters", {})
                self.settings_manager.settings["filters"] = {**existing, **current["filters"]}
            # Gradual Recovery: set current Settings panel config as default for Trading tab
            if hasattr(self.config_panel, "recovery_panel"):
                raw = self.config_panel.recovery_panel.get_config()
                try:
                    eb = raw.get("enable_streak_bonus", False)
                    enabled = raw.get("enabled", False)
                    self.settings_manager.settings["recovery"] = {
                        "enabled": enabled
                        if isinstance(enabled, bool)
                        else str(enabled).lower() in ("true", "1", "yes"),
                        "initial_loss": float(raw.get("initial_loss", 500)),
                        "target_profit_per_trade": float(raw.get("target_profit_per_trade", 5)),
                        "max_recovery_trades": int(raw.get("max_recovery_trades", 20)),
                        "margin_scaling_mode": str(raw.get("margin_scaling_mode", "fixed")),
                        "leverage_scaling_mode": str(raw.get("leverage_scaling_mode", "fixed")),
                        "min_leverage": int(raw.get("min_leverage", 2)),
                        "max_leverage": int(raw.get("max_leverage", 10)),
                        "enable_streak_bonus": (
                            eb if isinstance(eb, bool) else str(eb).lower() in ("true", "1", "yes")
                        ),
                    }
                except (TypeError, ValueError):
                    pass
            self.settings_manager.save()

            # Refresh Trading tab Current Settings so they reflect applied values
            if hasattr(self, "auto_trade_control") and hasattr(self.auto_trade_control, "update_from_settings"):
                try:
                    self.auto_trade_control.update_from_settings(
                        self.settings_manager.settings, status=self._get_current_status()
                    )
                    self.auto_trade_control.update_idletasks()
                    self.update_idletasks()
                except Exception as refresh_err:
                    logging.warning(f"Trading tab Current Settings refresh: {refresh_err}")

            # Scanner: reset pipeline so next scan uses new filters (atc_threshold, etc.)
            if hasattr(self, "scanner_manager"):
                self.scanner_manager._pipeline_initialized = False
                self.scanner_manager.pipeline = None

            if hasattr(self, "status_label"):
                self.status_label.configure(text="Settings applied (Scanner, Trading, Gradual Recovery default).")
            logging.info("Settings applied: Scanner, Trading, Gradual Recovery default (settings_manager overwritten)")
        except Exception as e:
            logging.error(f"Error applying settings: {e}")
            if hasattr(self, "status_label"):
                self.status_label.configure(text=f"Apply failed: {e}")

    def reload_current_settings(self):
        """Force reload Trading tab Current Settings: prefer Settings tab form, else in-memory settings."""
        try:
            settings_to_show = None
            # 1) Prefer current values from Settings tab form (so "settings from Settings tab" are passed)
            if hasattr(self, "config_panel"):
                current = self.config_panel.get_settings()
                if hasattr(self.config_panel, "default_leverage_var"):
                    current.setdefault("risk", {})["default_leverage"] = self.config_panel.default_leverage_var.get()
                existing_filters = self.settings_manager.settings.get("filters", {})
                settings_to_show = {
                    "risk": current.get("risk", {}),
                    "filters": {**existing_filters, **current.get("filters", {})},
                    "tp_sl": current.get("tp_sl", {}),
                    "api": current.get("api", {}),
                    "recovery": self.settings_manager.settings.get("recovery", {}),
                }
                if hasattr(self.config_panel, "recovery_panel"):
                    raw = self.config_panel.recovery_panel.get_config()
                    try:
                        eb = raw.get("enable_streak_bonus", False)
                        enabled = raw.get("enabled", False)
                        settings_to_show["recovery"] = {
                            "enabled": (
                                enabled if isinstance(enabled, bool) else str(enabled).lower() in ("true", "1", "yes")
                            ),
                            "initial_loss": float(raw.get("initial_loss", 500)),
                            "target_profit_per_trade": float(raw.get("target_profit_per_trade", 5)),
                            "max_recovery_trades": int(raw.get("max_recovery_trades", 20)),
                            "margin_scaling_mode": str(raw.get("margin_scaling_mode", "fixed")),
                            "leverage_scaling_mode": str(raw.get("leverage_scaling_mode", "fixed")),
                            "min_leverage": int(raw.get("min_leverage", 2)),
                            "max_leverage": int(raw.get("max_leverage", 10)),
                            "enable_streak_bonus": (
                                eb if isinstance(eb, bool) else str(eb).lower() in ("true", "1", "yes")
                            ),
                        }
                    except (TypeError, ValueError):
                        pass
            # 2) Fallback: use in-memory settings (no load() so we don't overwrite with file)
            if settings_to_show is None:
                self.settings_manager.load()
                settings_to_show = self.settings_manager.settings

            if hasattr(self, "auto_trade_control") and hasattr(self.auto_trade_control, "update_from_settings"):
                self.auto_trade_control.update_from_settings(settings_to_show, status=self._get_current_status())
                self.auto_trade_control.update_idletasks()
                self.update_idletasks()
            if hasattr(self, "status_label"):
                self.status_label.configure(text="Current Settings reloaded (from Settings tab form).")
            logging.info("Current Settings force-reloaded (Trading tab)")
        except Exception as e:
            logging.warning(f"Force reload Current Settings: {e}")
            if hasattr(self, "status_label"):
                self.status_label.configure(text=f"Reload failed: {e}")

    def on_recovery_config_change(self, event_type: str, data):
        """Handle recovery configuration change."""
        try:
            logging.info(f"Recovery {event_type}: {data}")

            if event_type == "recovery_started":
                self.settings_manager.set("recovery.enabled", True)
                self.settings_manager.set("recovery.config", data)
                self.settings_manager.save()

                # Update RecoveryManager with new config
                if hasattr(self, "recovery_manager"):
                    self.recovery_manager.set_enabled(True)
                    self.recovery_manager.update_config(data)

            elif event_type == "recovery_reset":
                self.settings_manager.set("recovery.enabled", False)
                self.settings_manager.save()

                # Reset RecoveryManager
                if hasattr(self, "recovery_manager"):
                    self.recovery_manager.reset()

            elif event_type == "recovery_alert":
                if hasattr(self, "status_label"):
                    self.status_label.configure(text=f"Recovery: {data}")

            elif event_type == "recovery_enabled_changed":
                # Handle enabled toggle from GUI
                enabled = data.get("enabled", False)
                self.settings_manager.set("recovery.enabled", enabled)
                self.settings_manager.save()

                if hasattr(self, "recovery_manager"):
                    self.recovery_manager.set_enabled(enabled)
                    logging.info(f"RecoveryManager enabled={enabled}")

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

        # Stop RecoveryManager
        if hasattr(self, "recovery_manager"):
            self.recovery_manager.stop()
            logging.info("RecoveryManager stopped")

        self.updater_manager.stop_all()
        self.auto_trade_manager.stop()
        self.scanner_manager._stop_scanner()

        self.destroy()
