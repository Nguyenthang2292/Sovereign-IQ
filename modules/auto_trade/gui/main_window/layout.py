"""UI Layout creation for Auto Trade Dashboard."""

import customtkinter as ctk

from gui.components.account_frame import AccountFrame
from gui.components.auto_trade_control import AutoTradeControl
from gui.components.config_panel import ConfigPanel
from gui.components.positions_frame import PositionsFrame
from gui.components.scanner_control import ScannerControl
from gui.components.signals_frame import SignalsFrame
from gui.components.stats_frame import StatsFrame
from gui.components.trade_form import TradeFormFrame
from gui.utils.colors import Colors


class LayoutManager:
    """Manages UI layout creation and component placement."""

    def __init__(self, parent):
        self.parent = parent
        self.components = {}

    def create_layout(self):
        """Create main application layout."""
        self.parent.grid_rowconfigure(1, weight=1)
        self.parent.grid_columnconfigure(0, weight=1)

        self._create_header()

        content_frame = ctk.CTkFrame(self.parent)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)

        # Create tabview
        self.parent.tabview = ctk.CTkTabview(content_frame)
        self.parent.tabview.pack(fill="both", expand=True)

        # Dashboard tab
        dashboard_tab = self.parent.tabview.add("Dashboard")
        self._populate_dashboard_tab(dashboard_tab)

        # Trading tab
        trading_tab = self.parent.tabview.add("Trading")
        self._populate_trading_tab(trading_tab)

        # Settings tab
        settings_tab = self.parent.tabview.add("Settings")
        self._populate_settings_tab(settings_tab)

        # Database tab
        database_tab = self.parent.tabview.add("Database")
        self._populate_database_tab(database_tab)

        self._create_status_bar()
        self.parent._update_mode_display()

    def _create_header(self):
        """Create header frame with title and mode indicator."""
        header_frame = ctk.CTkFrame(self.parent, height=60)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))

        title_label = ctk.CTkLabel(header_frame, text="Auto Trade Dashboard", font=("Arial", 20, "bold"))
        title_label.pack(side="left", padx=20)

        from gui.utils.modes import TradingMode

        mode_colors = {
            TradingMode.PRODUCTION: Colors.PRODUCTION,
            TradingMode.DEMO: Colors.DEMO,
            TradingMode.DRY_RUN: Colors.DRY_RUN,
        }

        mode_color = mode_colors.get(self.parent.mode, Colors.DRY_RUN)
        mode_text = self.parent.mode.replace("_", " ")

        self.parent.header_mode_label = ctk.CTkLabel(
            header_frame, text=f"[{mode_text}]", font=("Arial", 12), text_color=mode_color
        )
        self.parent.header_mode_label.pack(side="right", padx=20)

    def _create_status_bar(self):
        """Create status bar at bottom."""
        status_frame = ctk.CTkFrame(self.parent, height=30)
        status_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.parent.status_label = ctk.CTkLabel(status_frame, text="Ready", font=("Arial", 10), text_color="gray")
        self.parent.status_label.pack(side="left", padx=10)

        self.parent.last_update_label = ctk.CTkLabel(
            status_frame, text="Last update: --", font=("Arial", 10), text_color="gray"
        )
        self.parent.last_update_label.pack(side="right", padx=10)

    def _populate_dashboard_tab(self, parent):
        """Create dashboard interface."""
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)

        left_panel = ctk.CTkFrame(parent)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.parent.account_frame = AccountFrame(left_panel)
        self.parent.account_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.parent.stats_frame = StatsFrame(left_panel)
        self.parent.stats_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        right_panel = ctk.CTkFrame(parent)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=1)

        self.parent.signals_frame = SignalsFrame(right_panel)
        self.parent.signals_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.parent.positions_frame = PositionsFrame(right_panel, on_action_callback=self.parent.on_position_action)
        self.parent.positions_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def _populate_trading_tab(self, parent):
        """Create trading interface."""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        self.parent.trade_form = TradeFormFrame(parent, on_trade_callback=self.parent.on_trade_executed)
        self.parent.trade_form.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.parent.auto_trade_control = AutoTradeControl(parent, on_toggle_callback=self.parent.on_auto_trade_toggle)
        self.parent.auto_trade_control.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

    def _populate_settings_tab(self, parent):
        """Create settings interface with scroll support."""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        # Create scrollable frame for entire settings tab
        scroll_frame = ctk.CTkScrollableFrame(parent)
        scroll_frame.grid(row=0, column=0, sticky="nsew")
        scroll_frame.grid_columnconfigure(0, weight=3)
        scroll_frame.grid_columnconfigure(1, weight=2)

        self.parent.config_panel = ConfigPanel(
            scroll_frame,
            on_settings_change=self.parent.on_settings_change,
            mode=self.parent.mode,
            on_recovery_config_change=self.parent.on_recovery_config_change,
        )
        self.parent.config_panel.grid(row=0, column=0, sticky="new", padx=(0, 5))

        right_panel = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_panel.grid_rowconfigure(0, weight=0)
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        # Scanner Control (top)
        self.parent.scanner_control = ScannerControl(
            right_panel,
            on_scan_toggle=self.parent.on_scan_toggle,
            on_config_change=self.parent.on_scanner_config_change,
        )
        self.parent.scanner_control.grid(row=0, column=0, sticky="new", pady=(0, 10))

        # Logs Viewer (bottom)
        from gui.components.logs_viewer import LogsViewer

        self.parent.logs_viewer = LogsViewer(right_panel, str(self.parent.log_file_path))
        self.parent.logs_viewer.grid(row=1, column=0, sticky="nsew")

    def _populate_database_tab(self, parent):
        """Create database testing interface."""
        from gui.components.database_panel import DatabasePanel

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        self.parent.database_panel = DatabasePanel(parent, self.parent.settings_manager)
        self.parent.database_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
