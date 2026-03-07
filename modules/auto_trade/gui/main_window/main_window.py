"""Main Auto Trade Dashboard Window - orchestrates all components."""

import queue
import sys
from pathlib import Path
from typing import Any

import customtkinter as ctk

from modules.common.ui.logging import log_info

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from modules.auto_trade.gui.components.status_bar import StatusBar
from modules.auto_trade.gui.services.data_service import DataService
from modules.auto_trade.gui.services.settings_manager import SettingsManager
from modules.auto_trade.gui.services.websocket_data_service import WebSocketDataService
from modules.auto_trade.gui.utils.modes import TradingMode

from .auto_trade import AutoTradeManager
from .layout import LayoutManager
from .lifecycle_actions_mixin import LifecycleActionsMixin
from .position_actions import PositionActionHandler
from .risk_manager import RiskManager
from .scanner import ScannerManager
from .settings_handler import SettingsHandler
from .settings_recovery_mixin import SettingsRecoveryMixin
from .shortcuts_mixin import KeyboardShortcutsMixin
from .ui_updates_mixin import UIUpdatesMixin
from .updaters import UpdaterManager
from .websocket_handler import WebSocketHandler


class AutoTradeDashboard(
    KeyboardShortcutsMixin,
    UIUpdatesMixin,
    SettingsRecoveryMixin,
    LifecycleActionsMixin,
    ctk.CTk,
):
    """Main Auto Trade Dashboard application window."""

    def __init__(self) -> None:
        super().__init__()

        self.settings_manager = SettingsManager()
        self.settings_manager.load()

        # str() ensures Pylance infers self.mode as plain str, not TradingMode,
        # avoiding any base-class annotation conflict across the mixin MRO.
        self.mode = str(self.settings_manager.get("api.mode", TradingMode.DRY_RUN))

        self.data_service = DataService(mode=self.mode, settings_manager=self.settings_manager)
        self.ws_data_service = WebSocketDataService(
            mode=self.mode,
            settings_manager=self.settings_manager,
            event_bus=self.event_bus if hasattr(self, "event_bus") else None,
            tk_root=self,
        )

        self.title(f"Auto Trade Dashboard - [{self.mode}]")
        self.geometry("1200x800")
        self.minsize(800, 600)

        ctk.set_appearance_mode("dark")
        matrix_theme_path = Path(__file__).resolve().parent.parent / "config" / "matrix_theme.json"
        ctk.set_default_color_theme(str(matrix_theme_path))

        self._update_queue: queue.Queue = queue.Queue()
        self.log_queue: queue.Queue = queue.Queue(maxsize=500)

        # ── Redirect stdout → log_queue so Live Stream Logs textbox works ──
        # All project logging uses print() via modules.common.ui.logging,
        # so intercepting sys.stdout is sufficient to feed the GUI textbox.
        # NOTE: do NOT redirect sys.stderr — Python's C-level exception handler
        #       writes raw bytes to stderr and our text-mode proxy causes
        #       "lost sys.stderr" / TypeError in edge cases.
        from modules.auto_trade.gui.utils.stdout_redirector import StdoutRedirector

        self._original_stdout = sys.stdout
        sys.stdout = StdoutRedirector(self.log_queue, original_stdout=self._original_stdout)

        self.log_file_path = Path("logs/auto_trade_gui.log").absolute()
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._setup_file_logging()

        from modules.auto_trade.monitoring.event_system import EventSystem

        self.event_bus = EventSystem()
        self.settings_manager.set_event_bus(self.event_bus)
        if hasattr(self, "data_service") and self.data_service:
            self.data_service.set_event_bus(self.event_bus)
        if hasattr(self, "ws_data_service") and self.ws_data_service:
            self.ws_data_service.event_bus = self.event_bus

        from modules.auto_trade.strategies.recovery_manager import RecoveryManager

        recovery_config = self.settings_manager.get("recovery", {})
        self.recovery_manager = RecoveryManager(
            event_bus=self.event_bus,
            config=recovery_config,
            enabled=recovery_config.get("enabled", False),
            database=True,
        )
        self.recovery_manager.start()
        log_info(f"RecoveryManager started (enabled={recovery_config.get('enabled', False)})")

        self.layout_manager = LayoutManager(self)
        self.updater_manager = UpdaterManager(self)
        self.websocket_handler = WebSocketHandler(self)
        self.auto_trade_manager = AutoTradeManager(self)
        self.risk_manager = RiskManager(self)
        self.settings_handler = SettingsHandler(self)
        self.scanner_manager = ScannerManager(self)
        self.position_action_handler = PositionActionHandler(self)

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
        self.scheduled_exits_panel: Any = None
        self.status_bar: Any = None

        self.layout_manager.create_layout()

        log_info("GUI layout created successfully")
        log_info("LogsViewer should now be active and reading from log file")

        self._setup_keyboard_shortcuts()

        # ── Status bar — must be created BEFORE setup_updaters() so that
        # _update_timestamp() can find self.status_bar on first updater tick. ──
        self.status_bar = StatusBar(self, mode=self.mode)
        self.grid_rowconfigure(3, weight=0)
        self.status_bar.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.updater_manager.setup_updaters()
        self.websocket_handler.register_callbacks()
        self.settings_handler.apply_settings()

        if self.mode != TradingMode.DRY_RUN:
            self.ws_data_service.start()
            log_info(f"WebSocket service started (mode={self.mode})")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _setup_file_logging(self) -> None:
        """Set up file-based logging that captures all logs from all modules."""
        log_info("=" * 60)
        log_info("AUTO TRADE DASHBOARD")
        log_info(f"Mode: {self.mode}")
        log_info(f"Log file: {self.log_file_path}")
        log_info("=" * 60)
