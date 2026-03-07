"""UI Layout creation for Auto Trade Dashboard."""

import customtkinter as ctk

from modules.auto_trade.gui.components.account_frame import AccountFrame
from modules.auto_trade.gui.components.auto_trade_control import AutoTradeControl
from modules.auto_trade.gui.components.config_panel import ConfigPanel
from modules.auto_trade.gui.components.positions_frame import PositionsFrame
from modules.auto_trade.gui.components.scheduled_exits_panel import ScheduledExitsPanel
from modules.auto_trade.gui.components.signals_frame import SignalsFrame
from modules.auto_trade.gui.components.stats_frame import StatsFrame
from modules.auto_trade.gui.components.trade_form import TradeFormFrame
from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.svg_icons import get_icon
from modules.common.ui.logging import log_error


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

        # Settings tab
        settings_tab = self.parent.tabview.add("Settings")
        self._populate_settings_tab(settings_tab)

        # Scanner tab (Scanner Control + Logs)
        scanner_tab = self.parent.tabview.add("Scanner")
        self._populate_scanner_tab(scanner_tab)

        # Trading tab
        trading_tab = self.parent.tabview.add("Trading")
        self._populate_trading_tab(trading_tab)

        # Scheduled Exits tab
        scheduled_exits_tab = self.parent.tabview.add("Scheduled Exits")
        self._populate_scheduled_exits_tab(scheduled_exits_tab)

        # Database tab
        database_tab = self.parent.tabview.add("Database")
        self._populate_database_tab(database_tab)

        self.parent._update_mode_display()

    def _create_header(self):
        """Create header frame with title and mode indicator."""
        header_frame = ctk.CTkFrame(self.parent, height=60)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))

        title_label = ctk.CTkLabel(header_frame, text="Auto Trade Dashboard", font=("Arial", 20, "bold"))
        title_label.pack(side="left", padx=20)

        icon_kb = get_icon("keyboard", size=(16, 16), light_color="white", dark_color="white")
        shortcuts_btn = ctk.CTkButton(
            header_frame,
            text=" Shortcuts",
            image=icon_kb,
            compound="left",
            width=90,
            font=("Arial", 11),
            command=lambda: (
                self.parent._show_shortcuts_help() if hasattr(self.parent, "_show_shortcuts_help") else None
            ),
        )
        shortcuts_btn.pack(side="right", padx=(10, 10))

    def _populate_dashboard_tab(self, parent):
        """Create dashboard interface."""
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1, uniform="equal")
        parent.grid_columnconfigure(1, weight=1, uniform="equal")

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
        right_panel.grid_columnconfigure(0, weight=1)

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

    def _populate_scheduled_exits_tab(self, parent):
        """Create Scheduled Exits tab UI."""
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        self.parent.scheduled_exits_panel = ScheduledExitsPanel(
            parent,
            settings_manager=self.parent.settings_manager,
            on_open_settings=lambda: self.parent.tabview.set("Settings"),
        )
        self.parent.scheduled_exits_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

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
        title = ctk.CTkLabel(control_frame, text="Scanner Control", font=("Arial", 16, "bold"))
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

        self.parent.scanner_countdown_label = ctk.CTkLabel(
            status_container, text="", font=("Arial", 14, "bold"), text_color="#aaaaaa"
        )
        self.parent.scanner_countdown_label.pack(anchor="w", pady=(2, 0))

        # Buttons container (vertical stack)
        buttons_container = ctk.CTkFrame(status_btn_frame, fg_color="transparent")
        buttons_container.pack(fill="x")

        from modules.auto_trade.gui.utils.svg_icons import get_icon

        play_icon = get_icon("play", size=(18, 18), light_color="black", dark_color="black")
        stop_icon = get_icon("square", size=(18, 18), light_color="white", dark_color="white")
        next_scan_icon = get_icon("refresh", size=(16, 16), light_color="#aaaaaa", dark_color="#aaaaaa")
        self.parent.scanner_countdown_label.configure(image=next_scan_icon, compound="left")
        self.parent._next_scan_icon = next_scan_icon

        # Start/Stop button (full width)
        self.parent.scanner_start_button = ctk.CTkButton(
            buttons_container,
            text=" Start Scanner",
            image=play_icon,
            compound="left",
            font=("Arial", 13, "bold"),
            text_color="black",
            fg_color=Colors.BTN_SUCCESS,
            hover_color=Colors.BTN_SUCCESS_HOVER,
            height=40,
        )
        self.parent.scanner_start_button.pack(fill="x", pady=(0, 8))

        # --- Dynamic Background Button Handlers ---
        def update_scanner_buttons():
            sm = getattr(self.parent, "scanner_manager", None)
            if not sm:
                return

            # Update Scanner Button appearance
            if sm.updater is not None:
                self.parent.scanner_start_button.configure(
                    text=" Stop Scanner",
                    image=stop_icon,
                    text_color="white",
                    fg_color=Colors.BTN_DANGER,
                    hover_color=Colors.BTN_DANGER_HOVER,
                )
            else:
                self.parent.scanner_start_button.configure(
                    text=" Start Scanner",
                    image=play_icon,
                    text_color="black",
                    fg_color=Colors.BTN_SUCCESS,
                    hover_color=Colors.BTN_SUCCESS_HOVER,
                )
                # Clear countdown when scanner is stopped
                if hasattr(self.parent, "scanner_countdown_label"):
                    self.parent.scanner_countdown_label.configure(text="", text_color="#aaaaaa")

            # Delegate status label refresh to updater_manager (knows current position count)
            um = getattr(self.parent, "updater_manager", None)
            if um and hasattr(um, "_refresh_scanner_status_label"):
                um._refresh_scanner_status_label()

        self.parent.update_scanner_buttons = update_scanner_buttons

        def _on_start_click():
            if not hasattr(self.parent, "on_scan_toggle"):
                return
            sm = getattr(self.parent, "scanner_manager", None)
            if sm:
                is_running = sm.updater is not None
                self.parent.on_scan_toggle(False if is_running else True)
                update_scanner_buttons()

        self.parent.scanner_start_button.configure(command=_on_start_click)

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
                if hasattr(p, "atc_backend_var"):
                    p.atc_backend_var.set(config.get("atc_backend", "local"))
                if hasattr(p, "xgboost_backend_var"):
                    p.xgboost_backend_var.set(config.get("xgboost_backend", "local"))
                if hasattr(p, "sampling_strategy_var"):
                    p.sampling_strategy_var.set(config.get("sampling_strategy", "stratified"))
                if hasattr(p, "sample_percentage_entry"):
                    p.sample_percentage_entry.delete(0, "end")
                    p.sample_percentage_entry.insert(0, str(config.get("sample_percentage", 20)))
                if hasattr(p, "auto_scan_startup_var"):
                    p.auto_scan_startup_var.set(config.get("auto_start", True))
                # Load new filter fields
                if hasattr(p, "min_signal_score_var"):
                    p.min_signal_score_var.set(config.get("min_signal_score", 0.7))
                    if hasattr(p, "min_signal_score_label"):
                        p.min_signal_score_label.configure(text=f"{config.get('min_signal_score', 0.7):.2f}")
                if hasattr(p, "min_volume_entry"):
                    p.min_volume_entry.delete(0, "end")
                    p.min_volume_entry.insert(0, str(config.get("min_volume", 50)))
                if hasattr(p, "enable_xgboost_var"):
                    p.enable_xgboost_var.set(config.get("enable_xgboost", True))
                if hasattr(p, "atc_threshold_var"):
                    p.atc_threshold_var.set(config.get("atc_threshold", 0.6))
                    if hasattr(p, "atc_threshold_label"):
                        p.atc_threshold_label.configure(text=f"{config.get('atc_threshold', 0.6):.2f}")
                if hasattr(p, "enable_gann_square_var"):
                    p.enable_gann_square_var.set(config.get("enable_gann_square", False))
                if hasattr(p, "gann_timeframe_var"):
                    p.gann_timeframe_var.set(config.get("gann_timeframe", "1h"))
                if hasattr(p, "gann_candle_limit_entry"):
                    p.gann_candle_limit_entry.delete(0, "end")
                    p.gann_candle_limit_entry.insert(0, str(config.get("gann_candle_limit", 200)))
                if hasattr(p, "gann_lookback_entry"):
                    p.gann_lookback_entry.delete(0, "end")
                    p.gann_lookback_entry.insert(0, str(config.get("gann_lookback", 5)))
                # Order Book Gate
                if hasattr(p, "enable_order_book_var"):
                    p.enable_order_book_var.set(config.get("enable_order_book", False))
                if hasattr(p, "ob_depth_entry"):
                    p.ob_depth_entry.delete(0, "end")
                    p.ob_depth_entry.insert(0, str(config.get("ob_depth", 20)))
                if hasattr(p, "ob_imbalance_threshold_var"):
                    p.ob_imbalance_threshold_var.set(config.get("ob_imbalance_threshold", 0.2))
                    if hasattr(p, "ob_threshold_label"):
                        p.ob_threshold_label.configure(text=f"{config.get('ob_imbalance_threshold', 0.2):.2f}")

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
                    if "backend" in labels:
                        labels["backend"].configure(text=str(config.get("atc_backend", "local")).upper())
                    if "xgboost_backend" in labels:
                        labels["xgboost_backend"].configure(text=str(config.get("xgboost_backend", "local")).upper())
                    if "min_signal_score" in labels:
                        labels["min_signal_score"].configure(text=f"{config.get('min_signal_score', 0.2):.2f}")
                    if "min_volume" in labels:
                        labels["min_volume"].configure(text=f"{config.get('min_volume', 5.0):.1f}")
                    if "atc_threshold" in labels:
                        labels["atc_threshold"].configure(text=f"{config.get('atc_threshold', 0.0):.2f}")
                    if "enable_xgboost" in labels:
                        labels["enable_xgboost"].configure(
                            text="Enabled" if config.get("enable_xgboost", True) else "Disabled"
                        )
                    if "enable_gann_square" in labels:
                        labels["enable_gann_square"].configure(
                            text="Enabled" if config.get("enable_gann_square", False) else "Disabled"
                        )
                    if "gann_timeframe" in labels:
                        labels["gann_timeframe"].configure(text=config.get("gann_timeframe", "1h"))
                    if "gann_candle_limit" in labels:
                        labels["gann_candle_limit"].configure(text=str(config.get("gann_candle_limit", 200)))
                    if "gann_lookback" in labels:
                        labels["gann_lookback"].configure(text=str(config.get("gann_lookback", 5)))

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
                    "atc_backend": getattr(p.atc_backend_var, "get", lambda: "local")() or "local",
                    "xgboost_backend": getattr(p.xgboost_backend_var, "get", lambda: "local")() or "local",
                    "sampling_strategy": strat,
                    "sample_percentage": 20,
                    "auto_start": getattr(p, "auto_scan_startup_var", None) and p.auto_scan_startup_var.get(),
                    "min_signal_score": 0.7,
                    "min_volume": 50.0,
                    "enable_xgboost": True,
                    "atc_threshold": 0.6,
                }
                try:
                    sample_val = (p.sample_percentage_entry.get() or "20").strip()
                    config["sample_percentage"] = max(1, min(100, int(float(sample_val))))
                except (ValueError, TypeError, AttributeError):
                    pass
                # Collect new filter fields
                try:
                    config["min_signal_score"] = round(float(getattr(p, "min_signal_score_var", None).get()), 2)
                except (AttributeError, TypeError, ValueError):
                    pass
                try:
                    vol_str = (p.min_volume_entry.get() or "50").strip()
                    config["min_volume"] = max(0.0, float(vol_str))
                except (AttributeError, ValueError):
                    pass
                try:
                    config["enable_xgboost"] = bool(p.enable_xgboost_var.get())
                except (AttributeError, Exception):
                    pass
                try:
                    config["atc_threshold"] = round(float(p.atc_threshold_var.get()), 2)
                except (AttributeError, TypeError, ValueError):
                    pass
                try:
                    config["enable_gann_square"] = bool(p.enable_gann_square_var.get())
                except (AttributeError, Exception):
                    pass
                try:
                    config["gann_timeframe"] = p.gann_timeframe_var.get()
                except (AttributeError, Exception):
                    pass
                try:
                    config["gann_candle_limit"] = int(p.gann_candle_limit_entry.get())
                except (AttributeError, ValueError, TypeError):
                    pass
                try:
                    config["gann_lookback"] = int(p.gann_lookback_entry.get())
                except (AttributeError, ValueError, TypeError):
                    pass
                # Order Book Gate
                try:
                    config["enable_order_book"] = bool(p.enable_order_book_var.get())
                except (AttributeError, Exception):
                    pass
                try:
                    config["ob_depth"] = max(1, int(p.ob_depth_entry.get() or "20"))
                except (AttributeError, ValueError, TypeError):
                    pass
                try:
                    config["ob_imbalance_threshold"] = round(float(p.ob_imbalance_threshold_var.get()), 2)
                except (AttributeError, TypeError, ValueError):
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
                    if "backend" in labels:
                        labels["backend"].configure(text=str(config["atc_backend"]).upper())
                    if "xgboost_backend" in labels:
                        labels["xgboost_backend"].configure(text=str(config["xgboost_backend"]).upper())
                    if "min_signal_score" in labels:
                        labels["min_signal_score"].configure(text=f"{config.get('min_signal_score', 0.2):.2f}")
                    if "min_volume" in labels:
                        labels["min_volume"].configure(text=f"{config.get('min_volume', 5.0):.1f}")
                    if "atc_threshold" in labels:
                        labels["atc_threshold"].configure(text=f"{config.get('atc_threshold', 0.0):.2f}")
                    if "enable_xgboost" in labels:
                        labels["enable_xgboost"].configure(
                            text="Enabled" if config.get("enable_xgboost", True) else "Disabled"
                        )
                    if "enable_gann_square" in labels:
                        labels["enable_gann_square"].configure(
                            text="Enabled" if config.get("enable_gann_square", False) else "Disabled"
                        )
                    if "gann_timeframe" in labels:
                        labels["gann_timeframe"].configure(text=config.get("gann_timeframe", "1h"))
                    if "gann_candle_limit" in labels:
                        labels["gann_candle_limit"].configure(text=str(config.get("gann_candle_limit", 200)))
                    if "gann_lookback" in labels:
                        labels["gann_lookback"].configure(text=str(config.get("gann_lookback", 5)))
                    if "enable_order_book" in labels:
                        labels["enable_order_book"].configure(
                            text="Enabled" if config.get("enable_order_book", False) else "Disabled"
                        )
                    if "ob_depth" in labels:
                        labels["ob_depth"].configure(text=str(config.get("ob_depth", 20)))
                    if "ob_imbalance_threshold" in labels:
                        labels["ob_imbalance_threshold"].configure(
                            text=f"{config.get('ob_imbalance_threshold', 0.2):.2f}"
                        )
            except Exception:
                pass

        self._push_scanner_config = _push_scanner_config

        # Column 0: Scanner Configuration

        config_frame = ctk.CTkFrame(parent, fg_color=Colors.get_card_bg(), corner_radius=10)
        config_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=(0, 10))

        config_title = ctk.CTkLabel(config_frame, text="Scanner Configuration", font=("Arial", 14, "bold"))
        config_title.pack(pady=(10, 5))

        # Configuration inputs (scrollable to fit all controls)
        inputs_frame = ctk.CTkScrollableFrame(config_frame, fg_color="transparent")
        inputs_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        # ═══════════════════════════════════════════════
        # GROUP 1: Scan Settings
        # ═══════════════════════════════════════════════
        scan_group = ctk.CTkFrame(inputs_frame, fg_color=("gray85", "gray20"), corner_radius=8)
        scan_group.pack(fill="x", pady=(0, 8))
        scan_group.grid_columnconfigure(0, weight=0, minsize=130)
        scan_group.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(scan_group, text="Scan Settings", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4)
        )

        # Auto-scan on startup
        self.parent.auto_scan_startup_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            scan_group,
            text="Auto-start on startup",
            variable=self.parent.auto_scan_startup_var,
            command=self._push_scanner_config,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(4, 8))

        # Scan interval
        ctk.CTkLabel(scan_group, text="Interval (min):", font=("Arial", 10), text_color="gray", anchor="w").grid(
            row=2, column=0, sticky="w", padx=(10, 5), pady=4
        )
        self.parent.scan_interval_entry = ctk.CTkEntry(scan_group, placeholder_text="5")
        self.parent.scan_interval_entry.grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=4)
        self.parent.scan_interval_entry.insert(0, "5")
        self.parent.scan_interval_entry.bind("<FocusOut>", lambda e: self._push_scanner_config())
        self.parent.scan_interval_entry.bind("<Return>", lambda e: self._push_scanner_config())

        # Timeframe
        ctk.CTkLabel(scan_group, text="Timeframe:", font=("Arial", 10), text_color="gray", anchor="w").grid(
            row=3, column=0, sticky="w", padx=(10, 5), pady=4
        )
        self.parent.timeframe_var = ctk.StringVar(value="15m")
        ctk.CTkComboBox(
            scan_group,
            values=["5m", "15m", "30m", "1h", "4h", "1d"],
            variable=self.parent.timeframe_var,
            command=lambda _: self._push_scanner_config(),
        ).grid(row=3, column=1, sticky="ew", padx=(0, 10), pady=4)

        # Sampling strategy
        ctk.CTkLabel(scan_group, text="Strategy:", font=("Arial", 10), text_color="gray", anchor="w").grid(
            row=4, column=0, sticky="w", padx=(10, 5), pady=4
        )
        self.parent.sampling_strategy_var = ctk.StringVar(value="stratified")
        ctk.CTkComboBox(
            scan_group,
            values=["random", "stratified", "volume_weighted"],
            variable=self.parent.sampling_strategy_var,
            command=lambda _: self._push_scanner_config(),
        ).grid(row=4, column=1, sticky="ew", padx=(0, 10), pady=4)

        # Sample percentage
        ctk.CTkLabel(scan_group, text="Sample (%):", font=("Arial", 10), text_color="gray", anchor="w").grid(
            row=5, column=0, sticky="w", padx=(10, 5), pady=(4, 10)
        )
        self.parent.sample_percentage_entry = ctk.CTkEntry(scan_group, placeholder_text="20")
        self.parent.sample_percentage_entry.grid(row=5, column=1, sticky="ew", padx=(0, 10), pady=(4, 10))
        self.parent.sample_percentage_entry.insert(0, "20.0")
        self.parent.sample_percentage_entry.bind("<FocusOut>", lambda e: self._push_scanner_config())
        self.parent.sample_percentage_entry.bind("<Return>", lambda e: self._push_scanner_config())

        # ═══════════════════════════════════════════════
        # GROUP 2: Signal Filters
        # ═══════════════════════════════════════════════
        signal_group = ctk.CTkFrame(inputs_frame, fg_color=("gray85", "gray20"), corner_radius=8)
        signal_group.pack(fill="x", pady=(0, 8))
        signal_group.grid_columnconfigure(0, weight=0, minsize=130)
        signal_group.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(signal_group, text="Signal Filters", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4)
        )

        # Min Signal Score
        ctk.CTkLabel(signal_group, text="Min Signal Score:", font=("Arial", 10), text_color="gray", anchor="w").grid(
            row=1, column=0, sticky="w", padx=(10, 5), pady=4
        )

        slider_frame1 = ctk.CTkFrame(signal_group, fg_color="transparent")
        slider_frame1.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=4)
        slider_frame1.grid_columnconfigure(0, weight=1)
        slider_frame1.grid_columnconfigure(1, weight=0)

        self.parent.min_signal_score_var = ctk.DoubleVar(value=0.7)
        ctk.CTkSlider(
            slider_frame1,
            from_=0,
            to=1,
            number_of_steps=100,
            variable=self.parent.min_signal_score_var,
            command=lambda _: self._push_scanner_config(),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.parent.min_signal_score_label = ctk.CTkLabel(
            slider_frame1, text="0.70", font=("Arial", 9), text_color="gray", width=30
        )
        self.parent.min_signal_score_label.grid(row=0, column=1, sticky="e")

        def _on_score_change(*_args):
            try:
                v = self.parent.min_signal_score_var.get()
                self.parent.min_signal_score_label.configure(text=f"{v:.2f}")
                self._push_scanner_config()
            except Exception:
                pass

        self.parent.min_signal_score_var.trace_add("write", _on_score_change)

        # Explanation for Min Signal Score
        ctk.CTkLabel(
            signal_group,
            text="(High score = stricter filtering.\nE.g. >0.4 means fewer but stronger signals)",
            font=("Arial", 9, "italic"),
            text_color="gray",
            justify="left",
            anchor="w",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 4))

        # Min 24h Volume
        ctk.CTkLabel(signal_group, text="Min 24h Vol (M):", font=("Arial", 10), text_color="gray", anchor="w").grid(
            row=3, column=0, sticky="w", padx=(10, 5), pady=(4, 8)
        )
        self.parent.min_volume_entry = ctk.CTkEntry(signal_group, placeholder_text="50")
        self.parent.min_volume_entry.grid(row=3, column=1, sticky="ew", padx=(0, 10), pady=(4, 8))
        self.parent.min_volume_entry.insert(0, "50")
        self.parent.min_volume_entry.bind("<FocusOut>", lambda e: self._push_scanner_config())
        self.parent.min_volume_entry.bind("<Return>", lambda e: self._push_scanner_config())

        # ═══════════════════════════════════════════════
        # GROUP 3: ATC Configuration
        # ═══════════════════════════════════════════════
        atc_group = ctk.CTkFrame(inputs_frame, fg_color=("gray85", "gray20"), corner_radius=8)
        atc_group.pack(fill="x", pady=(0, 8))
        atc_group.grid_columnconfigure(0, weight=0, minsize=130)
        atc_group.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(atc_group, text="ATC Configuration", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4)
        )

        # ATC backend switch
        ctk.CTkLabel(atc_group, text="Backend:", font=("Arial", 10), text_color="gray", anchor="w").grid(
            row=1, column=0, sticky="w", padx=(10, 5), pady=4
        )
        self.parent.atc_backend_var = ctk.StringVar(value="local")
        ctk.CTkSegmentedButton(
            atc_group,
            values=["local", "serverless"],
            variable=self.parent.atc_backend_var,
            command=lambda _: self._push_scanner_config(),
        ).grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=4)

        # ATC base threshold
        ctk.CTkLabel(atc_group, text="Base threshold:", font=("Arial", 10), text_color="gray", anchor="w").grid(
            row=2, column=0, sticky="w", padx=(10, 5), pady=4
        )

        slider_frame2 = ctk.CTkFrame(atc_group, fg_color="transparent")
        slider_frame2.grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=4)
        slider_frame2.grid_columnconfigure(0, weight=1)
        slider_frame2.grid_columnconfigure(1, weight=0)

        self.parent.atc_threshold_var = ctk.DoubleVar(value=0.6)
        ctk.CTkSlider(
            slider_frame2,
            from_=0,
            to=1,
            number_of_steps=100,
            variable=self.parent.atc_threshold_var,
            command=lambda _: self._push_scanner_config(),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.parent.atc_threshold_label = ctk.CTkLabel(
            slider_frame2, text="0.60", font=("Arial", 9), text_color="gray", width=30
        )
        self.parent.atc_threshold_label.grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(
            atc_group,
            text="Scaled down when some timeframes fail.",
            font=("Arial", 9, "italic"),
            text_color="gray",
            anchor="w",
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 8))

        def _on_atc_change(*_args):
            try:
                v = self.parent.atc_threshold_var.get()
                self.parent.atc_threshold_label.configure(text=f"{v:.2f}")
                self._push_scanner_config()
            except Exception:
                pass

        self.parent.atc_threshold_var.trace_add("write", _on_atc_change)

        # ═══════════════════════════════════════════════
        # GROUP 4: XGBoost Configuration
        # ═══════════════════════════════════════════════
        xgb_group = ctk.CTkFrame(inputs_frame, fg_color=("gray85", "gray20"), corner_radius=8)
        xgb_group.pack(fill="x", pady=(0, 8))
        xgb_group.grid_columnconfigure(0, weight=0, minsize=130)
        xgb_group.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(xgb_group, text="XGBoost Configuration", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4)
        )

        # XGBoost enable checkbox
        self.parent.enable_xgboost_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            xgb_group,
            text="Enable XGBoost Model",
            variable=self.parent.enable_xgboost_var,
            command=self._push_scanner_config,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 4))

        # XGBoost backend switch
        ctk.CTkLabel(xgb_group, text="Backend:", font=("Arial", 10), text_color="gray", anchor="w").grid(
            row=2, column=0, sticky="w", padx=(10, 5), pady=(4, 8)
        )
        self.parent.xgboost_backend_var = ctk.StringVar(value="local")
        ctk.CTkSegmentedButton(
            xgb_group,
            values=["local", "serverless"],
            variable=self.parent.xgboost_backend_var,
            command=lambda _: self._push_scanner_config(),
        ).grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=(4, 8))

        # ═══════════════════════════════════════════════
        # GROUP 5: Gann Square Configuration
        # ═══════════════════════════════════════════════
        gann_group = ctk.CTkFrame(inputs_frame, fg_color=("gray85", "gray20"), corner_radius=8)
        gann_group.pack(fill="x", pady=(0, 8))
        gann_group.grid_columnconfigure(0, weight=0, minsize=130)
        gann_group.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(gann_group, text="Gann Square Configuration", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4)
        )

        self.parent.enable_gann_square_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            gann_group,
            text="Enable Gann Filter",
            variable=self.parent.enable_gann_square_var,
            command=self._push_scanner_config,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 4))

        ctk.CTkLabel(gann_group, text="Timeframe:", font=("Arial", 10), text_color="gray", anchor="w").grid(
            row=2, column=0, sticky="w", padx=(10, 5), pady=4
        )
        self.parent.gann_timeframe_var = ctk.StringVar(value="1h")
        ctk.CTkComboBox(
            gann_group,
            values=["15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"],
            variable=self.parent.gann_timeframe_var,
            command=lambda _: self._push_scanner_config(),
        ).grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=4)

        ctk.CTkLabel(gann_group, text="Candles Limit:", font=("Arial", 10), text_color="gray", anchor="w").grid(
            row=3, column=0, sticky="w", padx=(10, 5), pady=4
        )
        self.parent.gann_candle_limit_entry = ctk.CTkEntry(gann_group, placeholder_text="200")
        self.parent.gann_candle_limit_entry.grid(row=3, column=1, sticky="ew", padx=(0, 10), pady=4)
        self.parent.gann_candle_limit_entry.insert(0, "200")
        self.parent.gann_candle_limit_entry.bind("<FocusOut>", lambda e: self._push_scanner_config())
        self.parent.gann_candle_limit_entry.bind("<Return>", lambda e: self._push_scanner_config())

        ctk.CTkLabel(gann_group, text="Lookback:", font=("Arial", 10), text_color="gray", anchor="w").grid(
            row=4, column=0, sticky="w", padx=(10, 5), pady=(4, 8)
        )
        self.parent.gann_lookback_entry = ctk.CTkEntry(gann_group, placeholder_text="5")
        self.parent.gann_lookback_entry.grid(row=4, column=1, sticky="ew", padx=(0, 10), pady=(4, 8))
        self.parent.gann_lookback_entry.insert(0, "5")
        self.parent.gann_lookback_entry.bind("<FocusOut>", lambda e: self._push_scanner_config())
        self.parent.gann_lookback_entry.bind("<Return>", lambda e: self._push_scanner_config())

        # ═══════════════════════════════════════════════
        # GROUP 6: Order Book Configuration
        # ═══════════════════════════════════════════════
        ob_group = ctk.CTkFrame(inputs_frame, fg_color=("gray85", "gray20"), corner_radius=8)
        ob_group.pack(fill="x", pady=(0, 8))
        ob_group.grid_columnconfigure(0, weight=0, minsize=130)
        ob_group.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(ob_group, text="Order Book Configuration", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4)
        )

        # Enable checkbox
        self.parent.enable_order_book_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            ob_group,
            text="Enable Order Book Gate",
            variable=self.parent.enable_order_book_var,
            command=self._push_scanner_config,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 4))

        # Depth levels
        ctk.CTkLabel(ob_group, text="Depth levels:", font=("Arial", 10), text_color="gray", anchor="w").grid(
            row=2, column=0, sticky="w", padx=(10, 5), pady=4
        )
        self.parent.ob_depth_entry = ctk.CTkEntry(ob_group, placeholder_text="20")
        self.parent.ob_depth_entry.grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=4)
        self.parent.ob_depth_entry.insert(0, "20")
        self.parent.ob_depth_entry.bind("<FocusOut>", lambda e: self._push_scanner_config())
        self.parent.ob_depth_entry.bind("<Return>", lambda e: self._push_scanner_config())

        # Imbalance threshold
        ctk.CTkLabel(ob_group, text="Imbalance threshold:", font=("Arial", 10), text_color="gray", anchor="w").grid(
            row=3, column=0, sticky="w", padx=(10, 5), pady=4
        )

        slider_ob = ctk.CTkFrame(ob_group, fg_color="transparent")
        slider_ob.grid(row=3, column=1, sticky="ew", padx=(0, 10), pady=4)
        slider_ob.grid_columnconfigure(0, weight=1)
        slider_ob.grid_columnconfigure(1, weight=0)

        self.parent.ob_imbalance_threshold_var = ctk.DoubleVar(value=0.2)
        ctk.CTkSlider(
            slider_ob,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            variable=self.parent.ob_imbalance_threshold_var,
            command=lambda _: self._push_scanner_config(),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.parent.ob_threshold_label = ctk.CTkLabel(
            slider_ob, text="0.20", font=("Arial", 9), text_color="gray", width=30
        )
        self.parent.ob_threshold_label.grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(
            ob_group,
            text="Min bid/ask imbalance ratio to confirm signal.",
            font=("Arial", 9, "italic"),
            text_color="gray",
            anchor="w",
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 8))

        def _on_ob_threshold_change(*_args):
            try:
                v = self.parent.ob_imbalance_threshold_var.get()
                self.parent.ob_threshold_label.configure(text=f"{v:.2f}")
                self._push_scanner_config()
            except Exception:
                pass

        self.parent.ob_imbalance_threshold_var.trace_add("write", _on_ob_threshold_change)

        # Column 1: Current Settings
        settings_frame = ctk.CTkFrame(parent, fg_color=Colors.get_card_bg(), corner_radius=10)
        settings_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=(0, 10))

        settings_title = ctk.CTkLabel(settings_frame, text="Current Settings", font=("Arial", 14, "bold"))
        settings_title.pack(pady=(10, 5))

        # We need a reference to update these settings
        self.parent.settings_labels = {}

        settings_list = ctk.CTkScrollableFrame(settings_frame, fg_color="transparent")
        settings_list.pack(fill="both", expand=True, padx=5, pady=(5, 10))

        # Grouped settings structure: (group_title, [(label, default_value, key), ...])
        settings_groups = [
            (
                "Scan Schedule",
                [
                    ("Interval:", "5 min", "interval"),
                    ("Timeframe:", "15m", "timeframe"),
                    ("Strategy:", "stratified", "strategy"),
                    ("Sample:", "20%", "sample"),
                ],
            ),
            (
                "Signal Filters",
                [
                    ("Min Signal Score:", "0.20", "min_signal_score"),
                    ("Min 24h Vol (M):", "5.0", "min_volume"),
                ],
            ),
            (
                "ATC Config",
                [
                    ("Backend:", "LOCAL", "backend"),
                    ("ATC Threshold:", "0.00", "atc_threshold"),
                ],
            ),
            (
                "XGBoost Config",
                [
                    ("Backend:", "LOCAL", "xgboost_backend"),
                    ("XGBoost:", "Enabled", "enable_xgboost"),
                ],
            ),
            (
                "Gann Square Config",
                [
                    ("Gann Filter:", "Disabled", "enable_gann_square"),
                    ("Timeframe:", "1h", "gann_timeframe"),
                    ("Candles Limit:", "200", "gann_candle_limit"),
                    ("Lookback:", "5", "gann_lookback"),
                ],
            ),
            (
                "Order Book Config",
                [
                    ("OB Gate:", "Disabled", "enable_order_book"),
                    ("Depth levels:", "20", "ob_depth"),
                    ("Imbalance:", "0.20", "ob_imbalance_threshold"),
                ],
            ),
        ]

        for group_title, items in settings_groups:
            # Group header
            group_header = ctk.CTkLabel(
                settings_list,
                text=group_title,
                font=("Arial", 12, "bold"),
                text_color="#888888",
                anchor="w",
            )
            group_header.pack(fill="x", pady=(8, 2))
            # Separator line
            sep = ctk.CTkFrame(settings_list, height=1, fg_color=("#cccccc", "#444444"))
            sep.pack(fill="x", pady=(0, 4))
            # Items
            for label_text, value_text, key in items:
                row = ctk.CTkFrame(settings_list, fg_color="transparent")
                row.pack(fill="x", pady=1)
                ctk.CTkLabel(row, text=label_text, font=("Arial", 10), text_color="gray").pack(side="left")
                val_label = ctk.CTkLabel(row, text=value_text, font=("Arial", 10, "bold"))
                val_label.pack(side="right")
                self.parent.settings_labels[key] = val_label

        # Column 2: System Logs
        system_logs_frame = ctk.CTkFrame(parent, fg_color=Colors.get_card_bg(), corner_radius=10)
        system_logs_frame.grid(row=1, column=2, sticky="nsew", padx=5, pady=(0, 10))

        inner = ctk.CTkFrame(system_logs_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=10, pady=10)

        system_title = ctk.CTkLabel(inner, text="System Logs", font=("Arial", 14, "bold"))
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

        from modules.auto_trade.gui.utils.svg_icons import get_icon

        folder_icon = get_icon("folder_open", size=(16, 16), light_color="white", dark_color="white")
        file_icon = get_icon("file_text", size=(16, 16), light_color="white", dark_color="white")
        trash_icon = get_icon("trash", size=(16, 16), light_color="white", dark_color="white")

        ctk.CTkButton(
            btn_frame, text=" Open Log File", image=file_icon, compound="left", width=130, command=self._open_log_file
        ).pack(pady=(0, 6))

        ctk.CTkButton(
            btn_frame,
            text=" Open Folder",
            image=folder_icon,
            compound="left",
            width=130,
            fg_color=Colors.BTN_NEUTRAL,
            hover_color="#666666",
            command=self._open_log_folder,
        ).pack(pady=6)

        ctk.CTkButton(
            btn_frame,
            text=" Clear Logs",
            image=trash_icon,
            compound="left",
            width=130,
            fg_color=Colors.BTN_DANGER,
            hover_color=Colors.BTN_DANGER_HOVER,
            command=self._clear_logs,
        ).pack(pady=(6, 0))

        # Column 3: Live Stream Logs
        live_logs_frame = ctk.CTkFrame(parent, fg_color=Colors.get_card_bg(), corner_radius=10)
        live_logs_frame.grid(row=1, column=3, sticky="nsew", padx=(5, 10), pady=(0, 10))
        live_logs_frame.grid_columnconfigure(0, weight=1)
        live_logs_frame.grid_rowconfigure(1, weight=1)

        logs_label = ctk.CTkLabel(live_logs_frame, text="Live Stream Logs", font=("Arial", 14, "bold"))
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
            log_error("Error opening log file: %s", e)

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
            log_error("Error opening folder: %s", e)

    def _clear_logs(self):
        """Clear all logs from textbox."""
        try:
            if hasattr(self.parent, "logs_textbox"):
                self.parent.logs_textbox.configure(state="normal")
                self.parent.logs_textbox.delete("1.0", "end")
                self.parent.logs_textbox.insert("1.0", "🟢 Logs cleared. Waiting for new logs...\n")
                self.parent.logs_textbox.configure(state="disabled")
        except Exception as e:
            log_error("Error clearing logs: %s", e)

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
            on_risk_limits_toggle=self.parent.on_risk_limits_toggle,
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
            fg_color=Colors.BTN_PRIMARY,
            hover_color=Colors.BTN_PRIMARY_HOVER,
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
