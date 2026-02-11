"""UI Layout creation for Auto Trade Dashboard."""

import customtkinter as ctk

from modules.auto_trade.gui.components.account_frame import AccountFrame
from modules.auto_trade.gui.components.auto_trade_control import AutoTradeControl
from modules.auto_trade.gui.components.config_panel import ConfigPanel
from modules.auto_trade.gui.components.positions_frame import PositionsFrame
from modules.auto_trade.gui.components.signals_frame import SignalsFrame
from modules.auto_trade.gui.components.stats_frame import StatsFrame
from modules.auto_trade.gui.components.trade_form import TradeFormFrame
from modules.auto_trade.gui.utils.colors import Colors


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

        # Scanner tab (Scanner Control + Logs)
        scanner_tab = self.parent.tabview.add("Scanner")
        self._populate_scanner_tab(scanner_tab)

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

        from modules.auto_trade.gui.utils.modes import TradingMode

        mode_colors = {
            TradingMode.PRODUCTION: Colors.PRODUCTION,
            TradingMode.DEMO: Colors.DEMO,
            TradingMode.DRY_RUN: Colors.DRY_RUN,
        }

        mode_color = mode_colors.get(self.parent.mode, Colors.DRY_RUN)
        mode_text = self.parent.mode.replace("_", " ")

        shortcuts_btn = ctk.CTkButton(
            header_frame,
            text="⌨ Shortcuts",
            width=90,
            font=("Arial", 11),
            command=lambda: self.parent._show_shortcuts_help()
            if hasattr(self.parent, "_show_shortcuts_help")
            else None,
        )
        shortcuts_btn.pack(side="right", padx=(10, 10))

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

        def on_run_scanner():
            self.parent.tabview.set("Scanner")
            if hasattr(self.parent, "on_scan_toggle"):
                self.parent.on_scan_toggle("manual")

        self.parent.signals_frame = SignalsFrame(right_panel, on_run_scanner_callback=on_run_scanner)
        self.parent.signals_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.parent.positions_frame = PositionsFrame(
            right_panel,
            on_action_callback=self.parent.on_position_action,
            on_open_trade_callback=lambda: self.parent.tabview.set("Trading"),
            on_refresh_callback=self.parent.refresh_positions,
            on_sync_callback=self.parent.on_sync_positions,
        )
        self.parent.positions_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def _populate_scanner_tab(self, parent):
        """Create Scanner tab with all 4 boxes in a single horizontal row."""
        # Configure grid: 1 row, 4 columns
        parent.grid_rowconfigure(0, weight=0)  # Top row (Scanner buttons)
        parent.grid_rowconfigure(1, weight=1)  # Main row (4 boxes)
        parent.grid_columnconfigure(0, weight=2)  # Scanner Config (wider)
        parent.grid_columnconfigure(1, weight=1)  # Current Settings
        parent.grid_columnconfigure(2, weight=1)  # System Logs
        parent.grid_columnconfigure(3, weight=2)  # Live Stream Logs (wider)

        # Row 0: Scanner Control Buttons (spans all 4 columns)
        control_frame = ctk.CTkFrame(parent, fg_color="transparent")
        control_frame.grid(row=0, column=0, columnspan=4, sticky="ew", padx=10, pady=10)
        control_frame.grid_columnconfigure(0, weight=1)

        # Title
        title = ctk.CTkLabel(control_frame, text="🔍 Scanner Control", font=("Arial", 16, "bold"))
        title.pack(pady=(0, 10))

        # Status and buttons container
        status_btn_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        status_btn_frame.pack(fill="x", padx=20)

        # Status indicator (top)
        status_container = ctk.CTkFrame(status_btn_frame, fg_color="transparent")
        status_container.pack(fill="x", pady=(0, 10))

        self.parent.scanner_status_label = ctk.CTkLabel(
            status_container, text="🔴 Scanner: STOPPED", font=("Arial", 14, "bold"), text_color="gray"
        )
        self.parent.scanner_status_label.pack(anchor="w")

        self.parent.scanner_last_scan_label = ctk.CTkLabel(
            status_container, text="Last scan: Never", font=("Arial", 10), text_color="gray"
        )
        self.parent.scanner_last_scan_label.pack(anchor="w", pady=(2, 0))

        self.parent.scanner_progress_label = ctk.CTkLabel(
            status_container, text="", font=("Arial", 10), text_color="#00ff88"
        )
        self.parent.scanner_progress_label.pack(anchor="w", pady=(2, 0))

        # Buttons container (vertical stack)
        buttons_container = ctk.CTkFrame(status_btn_frame, fg_color="transparent")
        buttons_container.pack(fill="x")

        # Start button (full width)
        self.parent.scanner_start_button = ctk.CTkButton(
            buttons_container,
            text="▶️ Start Scanner",
            font=("Arial", 13, "bold"),
            fg_color="#00ff88",
            hover_color="#00cc66",
            height=40,
            command=lambda: self.parent.on_scan_toggle(True) if hasattr(self.parent, "on_scan_toggle") else None,
        )
        self.parent.scanner_start_button.pack(fill="x", pady=(0, 8))

        # Manual Scan button (full width)
        self.parent.scanner_manual_button = ctk.CTkButton(
            buttons_container,
            text="🔄 Manual Scan",
            font=("Arial", 13),
            fg_color="#4488ff",
            hover_color="#0066ff",
            height=40,
            command=lambda: self.parent.on_scan_toggle("manual") if hasattr(self.parent, "on_scan_toggle") else None,
        )
        self.parent.scanner_manual_button.pack(fill="x")

        # Setup Adapter for legacy compatibility
        class ScannerControlAdapter:
            def __init__(self, parent_window):
                self.parent = parent_window
                self.progress_label = parent_window.scanner_progress_label

            def update_last_scan_time(self):
                from datetime import datetime

                now = datetime.now().strftime("%H:%M:%S")
                if hasattr(self.parent, "scanner_last_scan_label"):
                    self.parent.scanner_last_scan_label.configure(text=f"Last scan: {now}")

            def load_config(self, config: dict):
                """Apply scanner config to layout UI (mirrors ScannerControl.load_config)."""
                p = self.parent
                if hasattr(p, "scan_interval_entry"):
                    p.scan_interval_entry.delete(0, "end")
                    p.scan_interval_entry.insert(0, str(config.get("scan_interval", 5)))
                if hasattr(p, "timeframe_var"):
                    p.timeframe_var.set(config.get("timeframe", "15m"))
                if hasattr(p, "sampling_strategy_var"):
                    p.sampling_strategy_var.set(config.get("sampling_strategy", "stratified"))
                if hasattr(p, "sample_percentage_entry"):
                    p.sample_percentage_entry.delete(0, "end")
                    p.sample_percentage_entry.insert(0, str(config.get("sample_percentage", 20)))
                if hasattr(p, "auto_scan_startup_var"):
                    p.auto_scan_startup_var.set(config.get("auto_start", True))
                if hasattr(p, "settings_labels") and isinstance(p.settings_labels, dict):
                    labels = p.settings_labels
                    if "interval" in labels:
                        labels["interval"].configure(text=f"{config.get('scan_interval', 5)} min")
                    if "timeframe" in labels:
                        labels["timeframe"].configure(text=config.get("timeframe", "15m"))
                    if "strategy" in labels:
                        labels["strategy"].configure(text=config.get("sampling_strategy", "stratified"))
                    if "sample" in labels:
                        labels["sample"].configure(text=f"{config.get('sample_percentage', 20)}%")

        self.parent.scanner_control = ScannerControlAdapter(self.parent)

        def _push_scanner_config():
            """Build scanner config from layout widgets and push to settings."""
            try:
                p = self.parent
                interval = 5
                try:
                    interval = int(p.scan_interval_entry.get().strip() or "5")
                except (ValueError, AttributeError):
                    pass
                strat = getattr(p.sampling_strategy_var, "get", lambda: "stratified")() or "stratified"
                config = {
                    "scan_interval": max(1, min(60, interval)),
                    "timeframe": getattr(p.timeframe_var, "get", lambda: "15m")() or "15m",
                    "sampling_strategy": strat,
                    "sample_percentage": 20,
                    "auto_start": getattr(p, "auto_scan_startup_var", None) and p.auto_scan_startup_var.get(),
                }
                try:
                    sample_val = (p.sample_percentage_entry.get() or "20").strip()
                    config["sample_percentage"] = max(1, min(100, int(float(sample_val))))
                except (ValueError, TypeError, AttributeError):
                    pass
                if hasattr(p, "on_scanner_config_change"):
                    p.on_scanner_config_change(config)
                # Update Current Settings panel to match
                if getattr(p, "settings_labels", None):
                    labels = p.settings_labels
                    if "interval" in labels:
                        labels["interval"].configure(text=f"{config['scan_interval']} min")
                    if "timeframe" in labels:
                        labels["timeframe"].configure(text=config["timeframe"])
                    if "strategy" in labels:
                        labels["strategy"].configure(text=config["sampling_strategy"])
                    if "sample" in labels:
                        labels["sample"].configure(text=f"{config['sample_percentage']}%")
            except Exception:
                pass

        self._push_scanner_config = _push_scanner_config

        # Column 0: Scanner Configuration
        from modules.auto_trade.gui.utils.colors import Colors

        config_frame = ctk.CTkFrame(parent, fg_color=Colors.get_card_bg(), corner_radius=10)
        config_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=(0, 10))

        config_title = ctk.CTkLabel(config_frame, text="⚙️ Scanner Configuration", font=("Arial", 12, "bold"))
        config_title.pack(pady=(10, 5))

        # Configuration inputs
        inputs_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        inputs_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        # Scan interval
        interval_row = ctk.CTkFrame(inputs_frame, fg_color="transparent")
        interval_row.pack(fill="x", pady=3)
        ctk.CTkLabel(interval_row, text="Interval (min):", font=("Arial", 10), text_color="gray").pack(
            side="left", padx=(0, 5)
        )
        self.parent.scan_interval_entry = ctk.CTkEntry(interval_row, placeholder_text="5", width=160)
        self.parent.scan_interval_entry.pack(side="right")
        self.parent.scan_interval_entry.insert(0, "5")
        self.parent.scan_interval_entry.bind("<FocusOut>", lambda e: self._push_scanner_config())
        self.parent.scan_interval_entry.bind("<Return>", lambda e: self._push_scanner_config())

        # Timeframe
        timeframe_row = ctk.CTkFrame(inputs_frame, fg_color="transparent")
        timeframe_row.pack(fill="x", pady=3)
        ctk.CTkLabel(timeframe_row, text="Timeframe:", font=("Arial", 10), text_color="gray").pack(
            side="left", padx=(0, 5)
        )
        self.parent.timeframe_var = ctk.StringVar(value="15m")
        timeframe_dropdown = ctk.CTkComboBox(
            timeframe_row,
            values=["5m", "15m", "30m", "1h", "4h", "1d"],
            variable=self.parent.timeframe_var,
            width=160,
            command=lambda _: self._push_scanner_config(),
        )
        timeframe_dropdown.pack(side="right")

        # Sampling strategy
        strategy_row = ctk.CTkFrame(inputs_frame, fg_color="transparent")
        strategy_row.pack(fill="x", pady=3)
        ctk.CTkLabel(strategy_row, text="Strategy:", font=("Arial", 10), text_color="gray").pack(
            side="left", padx=(0, 5)
        )
        self.parent.sampling_strategy_var = ctk.StringVar(value="stratified")
        strategy_dropdown = ctk.CTkComboBox(
            strategy_row,
            values=["random", "stratified", "volume_weighted"],
            variable=self.parent.sampling_strategy_var,
            width=160,
            command=lambda _: self._push_scanner_config(),
        )
        strategy_dropdown.pack(side="right")

        # Sample percentage
        sample_row = ctk.CTkFrame(inputs_frame, fg_color="transparent")
        sample_row.pack(fill="x", pady=3)
        ctk.CTkLabel(sample_row, text="Sample (%):", font=("Arial", 10), text_color="gray").pack(
            side="left", padx=(0, 5)
        )
        self.parent.sample_percentage_entry = ctk.CTkEntry(sample_row, placeholder_text="20", width=160)
        self.parent.sample_percentage_entry.pack(side="right")
        self.parent.sample_percentage_entry.insert(0, "20.0")
        self.parent.sample_percentage_entry.bind("<FocusOut>", lambda e: self._push_scanner_config())
        self.parent.sample_percentage_entry.bind("<Return>", lambda e: self._push_scanner_config())

        # Auto-scan on startup
        self.parent.auto_scan_startup_var = ctk.BooleanVar(value=True)
        auto_scan_cb = ctk.CTkCheckBox(
            inputs_frame,
            text="Auto-start on startup",
            variable=self.parent.auto_scan_startup_var,
            command=self._push_scanner_config,
        )
        auto_scan_cb.pack(fill="x", pady=(8, 2))

        # Column 1: Current Settings
        settings_frame = ctk.CTkFrame(parent, fg_color=Colors.get_card_bg(), corner_radius=10)
        settings_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=(0, 10))

        settings_title = ctk.CTkLabel(settings_frame, text="📊 Current Settings", font=("Arial", 12, "bold"))
        settings_title.pack(pady=(10, 5))

        # We need a reference to update these settings
        self.parent.settings_labels = {}

        settings_list = ctk.CTkFrame(settings_frame, fg_color="transparent")
        settings_list.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        settings_data = [
            ("Interval:", "5 min", "interval"),
            ("Timeframe:", "15m", "timeframe"),
            ("Strategy:", "stratified", "strategy"),
            ("Sample:", "20.0%", "sample"),
            ("Status:", "Stopped", "status"),
        ]

        for label_text, value_text, key in settings_data:
            row = ctk.CTkFrame(settings_list, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=label_text, font=("Arial", 10), text_color="gray").pack(side="left")
            val_label = ctk.CTkLabel(row, text=value_text, font=("Arial", 10, "bold"))
            val_label.pack(side="right")
            self.parent.settings_labels[key] = val_label

        # Column 2: System Logs
        system_logs_frame = ctk.CTkFrame(parent, fg_color=Colors.get_card_bg(), corner_radius=10)
        system_logs_frame.grid(row=1, column=2, sticky="nsew", padx=5, pady=(0, 10))

        inner = ctk.CTkFrame(system_logs_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=10, pady=10)

        system_title = ctk.CTkLabel(inner, text="System Logs", font=("Arial", 12, "bold"))
        system_title.pack(pady=(0, 8))

        info = ctk.CTkLabel(
            inner,
            text=f"Logs saved to:\n{self.parent.log_file_path.name}",
            font=("Arial", 9),
            text_color="gray",
            justify="center",
        )
        info.pack(pady=5)

        # Stack buttons vertically
        btn_frame = ctk.CTkFrame(inner, fg_color="transparent")
        btn_frame.pack(pady=10)

        ctk.CTkButton(btn_frame, text="Open Log File", width=120, command=self._open_log_file).pack(pady=(0, 6))

        ctk.CTkButton(
            btn_frame,
            text="Open Folder",
            width=120,
            fg_color="#555555",
            hover_color="#666666",
            command=self._open_log_folder,
        ).pack(pady=6)

        ctk.CTkButton(
            btn_frame,
            text="🗑️ Clear Logs",
            width=120,
            fg_color="#ff6644",
            hover_color="#cc4422",
            command=self._clear_logs,
        ).pack(pady=(6, 0))

        # Column 3: Live Stream Logs
        live_logs_frame = ctk.CTkFrame(parent, fg_color=Colors.get_card_bg(), corner_radius=10)
        live_logs_frame.grid(row=1, column=3, sticky="nsew", padx=(5, 10), pady=(0, 10))
        live_logs_frame.grid_columnconfigure(0, weight=1)
        live_logs_frame.grid_rowconfigure(1, weight=1)

        logs_label = ctk.CTkLabel(live_logs_frame, text="📡 Live Stream Logs", font=("Arial", 12, "bold"))
        logs_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        self.parent.logs_textbox = ctk.CTkTextbox(live_logs_frame, font=("Consolas", 9), wrap="word")
        self.parent.logs_textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.parent.logs_textbox.insert("1.0", "🟢 Log stream ready. Waiting for logs...\n")
        self.parent.logs_textbox.configure(state="disabled")

    def _open_log_file(self):
        """Open log file in default editor."""
        import os
        import subprocess
        import sys
        from pathlib import Path

        try:
            log_path = Path(self.parent.log_file_path)
            if not log_path.exists():
                return

            if sys.platform == "win32":
                os.startfile(str(log_path))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(log_path)])
            else:
                subprocess.run(["xdg-open", str(log_path)])
        except Exception as e:
            print(f"Error opening log file: {e}")

    def _open_log_folder(self):
        """Open folder containing log file."""
        import os
        import subprocess
        import sys
        from pathlib import Path

        try:
            folder = Path(self.parent.log_file_path).parent
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)

            if sys.platform == "win32":
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(folder)])
            else:
                subprocess.run(["xdg-open", str(folder)])
        except Exception as e:
            print(f"Error opening folder: {e}")

    def _clear_logs(self):
        """Clear all logs from textbox."""
        try:
            if hasattr(self.parent, "logs_textbox"):
                self.parent.logs_textbox.configure(state="normal")
                self.parent.logs_textbox.delete("1.0", "end")
                self.parent.logs_textbox.insert("1.0", "🟢 Logs cleared. Waiting for new logs...\n")
                self.parent.logs_textbox.configure(state="disabled")
        except Exception as e:
            print(f"Error clearing logs: {e}")

    def _populate_trading_tab(self, parent):
        """Create trading interface."""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        self.parent.trade_form = TradeFormFrame(parent, on_trade_callback=self.parent.on_trade_executed)
        self.parent.trade_form.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.parent.auto_trade_control = AutoTradeControl(
            parent,
            on_toggle_callback=self.parent.on_auto_trade_toggle,
            on_reload_settings=self.parent.reload_current_settings,
        )
        self.parent.auto_trade_control.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

    def _populate_settings_tab(self, parent):
        """Create settings interface: Config Panel + Apply Settings button."""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        scroll_frame = ctk.CTkScrollableFrame(parent)
        scroll_frame.grid(row=0, column=0, sticky="nsew")
        scroll_frame.grid_columnconfigure(0, weight=1)
        scroll_frame.grid_rowconfigure(0, weight=0)
        scroll_frame.grid_rowconfigure(1, weight=0)

        self.parent.config_panel = ConfigPanel(
            scroll_frame,
            on_settings_change=self.parent.on_settings_change,
            mode=self.parent.mode,
            on_recovery_config_change=self.parent.on_recovery_config_change,
        )
        self.parent.config_panel.grid(row=0, column=0, sticky="new", padx=10, pady=10)

        apply_btn_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        apply_btn_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 15))
        apply_btn_frame.grid_columnconfigure(0, weight=1)

        self.parent.apply_settings_btn = ctk.CTkButton(
            apply_btn_frame,
            text="Apply Settings",
            font=("Arial", 14, "bold"),
            fg_color="#1f538d",
            hover_color="#2a6bb5",
            height=40,
            command=self.parent.on_apply_settings,
        )
        self.parent.apply_settings_btn.pack(pady=5)

    def _populate_database_tab(self, parent):
        """Create database testing interface."""
        from modules.auto_trade.gui.components.database_panel import DatabasePanel

        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        self.parent.database_panel = DatabasePanel(parent, self.parent.settings_manager)
        self.parent.database_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
