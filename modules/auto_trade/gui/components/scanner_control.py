from datetime import datetime
from typing import Any, Callable, Dict, Optional

import customtkinter as ctk

from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.fonts import Fonts
from modules.common.ui.logging import log_error, log_warn


class ScannerControl(ctk.CTkFrame):
    """
    Scanner Control Panel
    Control scanning operations, display status, and configure scanner settings
    """

    def __init__(
        self,
        parent: Any,
        on_scan_toggle: Optional[Callable[..., Any]] = None,
        on_config_change: Optional[Callable[..., Any]] = None,
    ):
        super().__init__(
            parent,
            fg_color=Colors.get_card_bg(),
            corner_radius=0,
            border_width=1,
            border_color=Colors.BORDER_NEON,
        )

        self.on_scan_toggle = on_scan_toggle
        self.on_config_change = on_config_change
        self.scanner_running = False

        # Title
        title = ctk.CTkLabel(self, text="🔍 Scanner Control", font=Fonts.H1)
        title.pack(pady=(10, 15))

        # Status indicator
        self._create_status_indicator()

        # Control buttons
        self._create_controls()

        # Scanner configuration (includes settings display)
        self._create_configuration()

    def _create_status_indicator(self):
        """Create scanner status indicator"""
        status_frame = ctk.CTkFrame(self, fg_color=Colors.TRANSPARENT)
        status_frame.pack(fill="x", padx=15, pady=10)

        # Status label with emoji
        self.status_label = ctk.CTkLabel(
            status_frame, text="🔴 Scanner: STOPPED", font=Fonts.H2, text_color=Colors.TEXT_MUTED
        )
        self.status_label.pack()

        # Last scan timestamp
        self.last_scan_label = ctk.CTkLabel(
            status_frame, text="Last scan: Never", font=Fonts.SMALL, text_color=Colors.TEXT_MUTED
        )
        self.last_scan_label.pack(pady=(5, 0))

        # Scan progress
        self.progress_label = ctk.CTkLabel(status_frame, text="", font=Fonts.SMALL, text_color=Colors.PROFIT)
        self.progress_label.pack(pady=(2, 0))

    def _create_controls(self):
        """Create start/stop control buttons"""
        controls_frame = ctk.CTkFrame(self, fg_color=Colors.TRANSPARENT)
        controls_frame.pack(fill="x", padx=15, pady=10)

        # Start button
        self.start_button = ctk.CTkButton(
            controls_frame,
            text="▶️ START SCANNER",
            font=Fonts.BUTTON,
            fg_color=Colors.BTN_SUCCESS,
            hover_color=Colors.BTN_SUCCESS_HOVER,
            command=self._start_scanner,
        )
        self.start_button.pack(fill="x", pady=(0, 8))

        # Stop button (hidden initially)
        self.stop_button = ctk.CTkButton(
            controls_frame,
            text="⏸️ STOP SCANNER",
            font=Fonts.BUTTON,
            fg_color=Colors.BTN_DANGER,
            hover_color=Colors.BTN_DANGER_HOVER,
            command=self._stop_scanner,
        )
        self.stop_button.pack(fill="x", pady=(0, 8))
        self.stop_button.pack_forget()  # Hide initially

        # Manual scan button
        self.manual_scan_button = ctk.CTkButton(
            controls_frame,
            text="🔄 MANUAL SCAN",
            font=Fonts.BUTTON_SM,
            fg_color=Colors.BTN_PRIMARY,
            hover_color=Colors.BTN_PRIMARY_HOVER,
            command=self._manual_scan,
        )
        self.manual_scan_button.pack(fill="x", pady=(0, 5))

    def _create_configuration(self):
        """Create scanner configuration and settings display side-by-side"""
        # Container frame for 2-column layout
        container = ctk.CTkFrame(self, fg_color=Colors.TRANSPARENT)
        container.pack(fill="both", expand=True, padx=15, pady=10)

        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)

        # ===== LEFT: Scanner Configuration =====
        config_frame = ctk.CTkFrame(
            container,
            fg_color=Colors.get_card_bg(),
            corner_radius=0,
            border_width=1,
            border_color=Colors.BORDER_NEON,
        )
        config_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Title
        config_title = ctk.CTkLabel(config_frame, text="⚙️ Scanner Configuration", font=Fonts.H3)
        config_title.pack(pady=(10, 5))

        # Configuration inputs
        inputs_frame = ctk.CTkScrollableFrame(config_frame, fg_color=Colors.TRANSPARENT)
        inputs_frame.pack(fill="both", expand=True, padx=5, pady=(5, 10))

        # Scan interval
        interval_label = ctk.CTkLabel(
            inputs_frame, text="Scan Interval (minutes):", font=Fonts.BODY, text_color=Colors.TEXT_MUTED
        )
        interval_label.grid(row=0, column=0, sticky="w", pady=5)

        self.scan_interval_entry = ctk.CTkEntry(inputs_frame, placeholder_text="5", width=150)
        self.scan_interval_entry.grid(row=0, column=1, sticky="e", pady=5, padx=(10, 0))
        self.scan_interval_entry.insert(0, "5")

        # Timeframe selector
        timeframe_label = ctk.CTkLabel(inputs_frame, text="Timeframe:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        timeframe_label.grid(row=1, column=0, sticky="w", pady=5)

        self.timeframe_var = ctk.StringVar(value="1h")
        timeframe_dropdown = ctk.CTkComboBox(
            inputs_frame,
            values=["5m", "15m", "30m", "1h", "4h", "1d"],
            variable=self.timeframe_var,
            width=150,
            command=self._on_config_change,
        )
        timeframe_dropdown.grid(row=1, column=1, sticky="e", pady=5, padx=(10, 0))

        # Sampling strategy selector
        strategy_label = ctk.CTkLabel(inputs_frame, text="Sampling Strategy:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        strategy_label.grid(row=2, column=0, sticky="w", pady=5)

        self.sampling_strategy_var = ctk.StringVar(value="stratified")
        strategy_dropdown = ctk.CTkComboBox(
            inputs_frame,
            values=["random", "volume_weighted", "stratified", "top_n_hybrid", "systematic", "liquidity_weighted"],
            variable=self.sampling_strategy_var,
            width=150,
            command=self._on_config_change,
        )
        strategy_dropdown.grid(row=2, column=1, sticky="e", pady=5, padx=(10, 0))

        # Sample percentage field
        percentage_label = ctk.CTkLabel(inputs_frame, text="Sample (%):", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        percentage_label.grid(row=3, column=0, sticky="w", pady=5)

        self.sample_percentage_entry = ctk.CTkEntry(inputs_frame, placeholder_text="20", width=150)
        self.sample_percentage_entry.grid(row=3, column=1, sticky="e", pady=5, padx=(10, 0))
        self.sample_percentage_entry.insert(0, "20")

        # Auto-scan on startup
        self.auto_scan_startup_var = ctk.BooleanVar(value=True)
        auto_scan_checkbox = ctk.CTkCheckBox(
            inputs_frame,
            text="Auto-start on startup",
            variable=self.auto_scan_startup_var,
            command=self._on_config_change,
        )
        auto_scan_checkbox.grid(row=4, column=0, columnspan=2, sticky="w", pady=5)

        # Min Signal Score
        min_score_label = ctk.CTkLabel(inputs_frame, text="Min Signal Score:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        min_score_label.grid(row=5, column=0, sticky="w", pady=5)

        self.min_score_var = ctk.DoubleVar(value=0.7)
        min_score_slider = ctk.CTkSlider(
            inputs_frame, from_=0, to=1, number_of_steps=100, variable=self.min_score_var, width=150
        )
        min_score_slider.grid(row=5, column=1, sticky="e", pady=5, padx=(10, 0))

        self.min_score_value_label = ctk.CTkLabel(
            inputs_frame, text=f"{self.min_score_var.get():.2f}", font=Fonts.SMALL, text_color=Colors.TEXT_MUTED
        )
        self.min_score_value_label.grid(row=6, column=1, sticky="e", pady=(0, 3), padx=(10, 0))

        # Explanation for Min Signal Score
        min_score_desc_label = ctk.CTkLabel(
            inputs_frame,
            text="(High score = stricter filtering.\nE.g. >0.4 means fewer but stronger signals)",
            font=Fonts.TINY,
            text_color=Colors.TEXT_MUTED,
            justify="left",
            anchor="w",
        )
        min_score_desc_label.grid(row=7, column=0, columnspan=2, sticky="w", pady=(0, 5))

        def _on_min_score_change(*args):
            try:
                v = self.min_score_var.get()
                self.min_score_value_label.configure(text=f"{v:.2f}")
                self._on_config_change()
            except Exception:
                pass

        self.min_score_var.trace_add("write", _on_min_score_change)

        # Min 24h Volume
        volume_label = ctk.CTkLabel(inputs_frame, text="Min 24h Volume (M):", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        volume_label.grid(row=8, column=0, sticky="w", pady=5)

        self.min_volume_entry = ctk.CTkEntry(inputs_frame, placeholder_text="50", width=150)
        self.min_volume_entry.grid(row=8, column=1, sticky="e", pady=5, padx=(10, 0))
        self.min_volume_entry.insert(0, "50")

        # ---- Model Filters group (XGBoost + ATC) ----
        model_group = ctk.CTkFrame(inputs_frame, fg_color=("gray85", "gray20"), corner_radius=0)
        model_group.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(10, 5))

        model_title = ctk.CTkLabel(model_group, text="Model Filters", font=Fonts.H3)
        model_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))

        # XGBoost checkbox
        self.enable_xgboost_var = ctk.BooleanVar(value=True)
        xgboost_checkbox = ctk.CTkCheckBox(
            model_group,
            text="Enable XGBoost Model",
            variable=self.enable_xgboost_var,
            command=self._on_config_change,
        )
        xgboost_checkbox.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 4))

        # ATC base threshold
        atc_label = ctk.CTkLabel(model_group, text="ATC base threshold:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        atc_label.grid(row=2, column=0, sticky="w", padx=10, pady=4)

        self.atc_threshold_var = ctk.DoubleVar(value=0.6)
        atc_slider = ctk.CTkSlider(
            model_group, from_=0, to=1, number_of_steps=100, variable=self.atc_threshold_var, width=130
        )
        atc_slider.grid(row=2, column=1, sticky="e", padx=10, pady=4)

        self.atc_value_label = ctk.CTkLabel(
            model_group, text=f"{self.atc_threshold_var.get():.2f}", font=Fonts.SMALL, text_color=Colors.TEXT_MUTED
        )
        self.atc_value_label.grid(row=3, column=1, sticky="e", padx=10, pady=(0, 4))

        atc_tooltip = ctk.CTkLabel(
            model_group,
            text="Scaled down when some timeframes fail.",
            font=Fonts.TINY,
            text_color=Colors.TEXT_MUTED,
        )
        atc_tooltip.grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 8))

        def _on_atc_change(*args):
            try:
                v = self.atc_threshold_var.get()
                self.atc_value_label.configure(text=f"{v:.2f}")
                self._on_config_change()
            except Exception:
                pass

        self.atc_threshold_var.trace_add("write", _on_atc_change)

        # ---- Gann Square Filter group ----
        gann_group = ctk.CTkFrame(inputs_frame, fg_color=("gray85", "gray20"), corner_radius=0)
        gann_group.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(10, 5))

        gann_title = ctk.CTkLabel(gann_group, text="Gann Square Filter", font=Fonts.H3)
        gann_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))

        # Enable Gann Square checkbox
        self.enable_gann_square_var = ctk.BooleanVar(value=False)
        gann_checkbox = ctk.CTkCheckBox(
            gann_group,
            text="Enable Gann Square Filter",
            variable=self.enable_gann_square_var,
            command=self._on_gann_toggle,
        )
        gann_checkbox.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 4))

        # Gann sub-frame (only shown when checkbox is ON)
        self.gann_sub_frame = ctk.CTkFrame(gann_group, fg_color=Colors.TRANSPARENT)
        self.gann_sub_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))
        self.gann_sub_frame.grid_remove()

        # Gann timeframe selector
        gann_tf_label = ctk.CTkLabel(self.gann_sub_frame, text="Gann TF:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        gann_tf_label.grid(row=0, column=0, sticky="w", pady=4)

        self.gann_tf_var = ctk.StringVar(value="1h")
        gann_tf_dropdown = ctk.CTkComboBox(
            self.gann_sub_frame,
            values=["15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"],
            variable=self.gann_tf_var,
            width=100,
            command=self._on_config_change,
        )
        gann_tf_dropdown.grid(row=0, column=1, sticky="e", pady=4, padx=(10, 0))

        # Gann candle limit
        gann_candle_label = ctk.CTkLabel(self.gann_sub_frame, text="Candles:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        gann_candle_label.grid(row=1, column=0, sticky="w", pady=4)

        self.gann_candle_limit_entry = ctk.CTkEntry(self.gann_sub_frame, placeholder_text="200", width=100)
        self.gann_candle_limit_entry.grid(row=1, column=1, sticky="e", pady=4, padx=(10, 0))
        self.gann_candle_limit_entry.insert(0, "200")

        # Gann lookback
        gann_lookback_label = ctk.CTkLabel(self.gann_sub_frame, text="Lookback:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        gann_lookback_label.grid(row=2, column=0, sticky="w", pady=4)

        self.gann_lookback_entry = ctk.CTkEntry(self.gann_sub_frame, placeholder_text="5", width=100)
        self.gann_lookback_entry.grid(row=2, column=1, sticky="e", pady=4, padx=(10, 0))
        self.gann_lookback_entry.insert(0, "5")

        self.gann_sub_frame.grid_columnconfigure(0, weight=0, minsize=80)
        self.gann_sub_frame.grid_columnconfigure(1, weight=1)

        # ---- Order Book Filter group ----
        ob_group = ctk.CTkFrame(inputs_frame, fg_color=("gray85", "gray20"), corner_radius=0)
        ob_group.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(10, 5))

        ob_title = ctk.CTkLabel(ob_group, text="Order Book Filter", font=Fonts.H3)
        ob_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))

        # Enable Order Book checkbox
        self.enable_order_book_var = ctk.BooleanVar(value=False)
        ob_checkbox = ctk.CTkCheckBox(
            ob_group,
            text="Enable Order Book Gate",
            variable=self.enable_order_book_var,
            command=self._on_order_book_toggle,
        )
        ob_checkbox.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 4))

        # Order Book sub-frame (only shown when checkbox is ON)
        self.ob_sub_frame = ctk.CTkFrame(ob_group, fg_color=Colors.TRANSPARENT)
        self.ob_sub_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))
        self.ob_sub_frame.grid_remove()

        # Depth levels
        ob_depth_label = ctk.CTkLabel(self.ob_sub_frame, text="Depth levels:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED)
        ob_depth_label.grid(row=0, column=0, sticky="w", pady=4)

        self.ob_depth_entry = ctk.CTkEntry(self.ob_sub_frame, placeholder_text="20", width=100)
        self.ob_depth_entry.grid(row=0, column=1, sticky="e", pady=4, padx=(10, 0))
        self.ob_depth_entry.insert(0, "20")

        # Imbalance threshold
        ob_thresh_label = ctk.CTkLabel(
            self.ob_sub_frame, text="Imbalance threshold:", font=Fonts.BODY, text_color=Colors.TEXT_MUTED
        )
        ob_thresh_label.grid(row=1, column=0, sticky="w", pady=4)

        self.ob_imbalance_threshold_var = ctk.DoubleVar(value=0.2)
        ob_slider = ctk.CTkSlider(
            self.ob_sub_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            variable=self.ob_imbalance_threshold_var,
            width=100,
        )
        ob_slider.grid(row=1, column=1, sticky="e", pady=4, padx=(10, 0))

        self.ob_threshold_value_label = ctk.CTkLabel(
            self.ob_sub_frame,
            text=f"{self.ob_imbalance_threshold_var.get():.2f}",
            font=Fonts.SMALL,
            text_color=Colors.TEXT_MUTED,
        )
        self.ob_threshold_value_label.grid(row=2, column=1, sticky="e", padx=10, pady=(0, 4))

        ob_tooltip = ctk.CTkLabel(
            self.ob_sub_frame,
            text="Min bid/ask imbalance ratio to confirm signal.",
            font=Fonts.TINY,
            text_color=Colors.TEXT_MUTED,
        )
        ob_tooltip.grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 4))

        def _on_ob_threshold_change(*args):
            try:
                v = self.ob_imbalance_threshold_var.get()
                self.ob_threshold_value_label.configure(text=f"{v:.2f}")
                self._on_config_change()
            except Exception:
                pass

        self.ob_imbalance_threshold_var.trace_add("write", _on_ob_threshold_change)

        self.ob_sub_frame.grid_columnconfigure(0, weight=0, minsize=110)
        self.ob_sub_frame.grid_columnconfigure(1, weight=1)

        # Configure grid columns - column 0 for labels needs minimum width
        inputs_frame.grid_columnconfigure(0, weight=0, minsize=140)
        inputs_frame.grid_columnconfigure(1, weight=1)

        # ===== RIGHT: Current Settings =====
        settings_frame = ctk.CTkFrame(
            container,
            fg_color=Colors.get_card_bg(),
            corner_radius=0,
            border_width=1,
            border_color=Colors.BORDER_NEON,
        )
        settings_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # Title
        settings_title = ctk.CTkLabel(settings_frame, text="📊 Current Settings", font=Fonts.H3)
        settings_title.pack(pady=(10, 5))

        # Settings list
        settings_list_frame = ctk.CTkFrame(settings_frame, fg_color=Colors.TRANSPARENT)
        settings_list_frame.pack(fill="x", padx=10, pady=(5, 10))

        settings = [
            ("Interval:", "5 min"),
            ("Timeframe:", "1h"),
            ("Strategy:", "stratified"),
            ("Sample:", "20%"),
            ("Status:", "Stopped"),
        ]

        for i, (label_text, value_text) in enumerate(settings):
            row_frame = ctk.CTkFrame(settings_list_frame, fg_color=Colors.TRANSPARENT)
            row_frame.pack(fill="x", pady=2)

            label = ctk.CTkLabel(row_frame, text=label_text, font=Fonts.SMALL, text_color=Colors.TEXT_MUTED)
            label.pack(side="left")

            value = ctk.CTkLabel(row_frame, text=value_text, font=Fonts.BODY)
            value.pack(side="right")

            # Store reference for updates
            if i == 0:
                self.setting_interval = value
            elif i == 1:
                self.setting_timeframe = value
            elif i == 2:
                self.setting_strategy = value
            elif i == 3:
                self.setting_percentage = value
            elif i == 4:
                self.setting_status = value

    def _start_scanner(self):
        """Start scanner"""
        if self.scanner_running:
            return

        try:
            self.scanner_running = True

            # Update UI
            self.start_button.pack_forget()
            self.stop_button.pack(fill="x", pady=(0, 8))
            self._update_status_indicator(True)

            # Call callback
            if self.on_scan_toggle:
                self.on_scan_toggle(True)

            # Update timestamp
            self.last_scan_label.configure(text="Last scan: Scanning...")

        except Exception as e:
            log_error("Error starting scanner: %s", e)

    def _stop_scanner(self):
        """Stop scanner"""
        if not self.scanner_running:
            return

        try:
            self.scanner_running = False

            # Update UI
            self.stop_button.pack_forget()
            self.start_button.pack(fill="x", pady=(0, 8))
            self._update_status_indicator(False)

            # Call callback
            if self.on_scan_toggle:
                self.on_scan_toggle(False)

            # Update timestamp
            now = datetime.now().strftime("%H:%M:%S")
            self.last_scan_label.configure(text=f"Last scan: Stopped at {now}")

        except Exception as e:
            log_error("Error stopping scanner: %s", e)

    def _manual_scan(self):
        """Trigger manual scan"""
        try:
            self.progress_label.configure(text="🔄 Scanning...")

            # Call callback for manual scan
            if self.on_scan_toggle:
                self.on_scan_toggle("manual")

            # Clear progress after 3 seconds
            self.after(3000, lambda: self.progress_label.configure(text=""))

        except Exception as e:
            log_error("Error triggering manual scan: %s", e)

    def _update_status_indicator(self, running: bool):
        """Update status display"""
        if running:
            self.status_label.configure(text="🟢 Scanner: RUNNING", text_color=Colors.PROFIT)
            self.setting_status.configure(text="Running", text_color=Colors.PROFIT)
            self._animate_status()
        else:
            self.status_label.configure(text="🔴 Scanner: STOPPED", text_color=Colors.TEXT_MUTED)
            self.setting_status.configure(text="Stopped", text_color=Colors.TEXT_MUTED)

    def _animate_status(self):
        """Pulse animation when running"""
        if not self.scanner_running:
            return

        current_text = self.status_label.cget("text")
        new_text = "🟢 Scanner: SCANNING" if "RUNNING" in current_text else "🟢 Scanner: RUNNING"
        self.status_label.configure(text=new_text)

        self.after(1000, self._animate_status)

    def _on_gann_toggle(self):
        """Handle Gann Square checkbox toggle."""
        if self.enable_gann_square_var.get():
            self.gann_sub_frame.grid()
        else:
            self.gann_sub_frame.grid_remove()
        self._on_config_change()

    def _on_order_book_toggle(self):
        """Handle Order Book Gate checkbox toggle."""
        if self.enable_order_book_var.get():
            self.ob_sub_frame.grid()
        else:
            self.ob_sub_frame.grid_remove()
        self._on_config_change()

    def _on_config_change(self, choice=None):
        """Handle configuration change"""
        try:
            # Update settings display
            interval = self.scan_interval_entry.get()
            timeframe = self.timeframe_var.get()
            strategy = self.sampling_strategy_var.get()
            percentage = self.sample_percentage_entry.get()

            self.setting_interval.configure(text=f"{interval} min")
            self.setting_timeframe.configure(text=timeframe)
            self.setting_strategy.configure(text=strategy)
            self.setting_percentage.configure(text=f"{percentage}%")

            # Call callback
            if self.on_config_change:
                config = self.get_config()
                self.on_config_change(config)

        except Exception as e:
            log_warn("Error updating config: %s", e)

    def update_last_scan_time(self):
        """Update last scan timestamp"""
        now = datetime.now().strftime("%H:%M:%S")
        self.last_scan_label.configure(text=f"Last scan: {now}")

    def get_config(self) -> Dict:
        """Get current scanner configuration"""
        try:
            min_volume = float(self.min_volume_entry.get())
            if min_volume < 0:
                min_volume = 50.0
        except (ValueError, AttributeError):
            min_volume = 50.0
        try:
            ob_depth = int(self.ob_depth_entry.get())
            if ob_depth <= 0:
                ob_depth = 20
        except (ValueError, AttributeError):
            ob_depth = 20
        return {
            "scan_interval": int(self.scan_interval_entry.get()),
            "timeframe": self.timeframe_var.get(),
            "sampling_strategy": self.sampling_strategy_var.get(),
            "sample_percentage": float(self.sample_percentage_entry.get()),
            "auto_start": self.auto_scan_startup_var.get(),
            "running": self.scanner_running,
            "min_signal_score": self.min_score_var.get(),
            "enable_xgboost": self.enable_xgboost_var.get(),
            "atc_threshold": self.atc_threshold_var.get(),
            "min_volume": min_volume,
            "enable_gann_square": self.enable_gann_square_var.get(),
            "gann_timeframe": self.gann_tf_var.get(),
            "gann_candle_limit": int(self.gann_candle_limit_entry.get()) if self.gann_candle_limit_entry.get() else 200,
            "gann_lookback": int(self.gann_lookback_entry.get())
            if getattr(self, "gann_lookback_entry", None) and self.gann_lookback_entry.get()
            else 5,
            # Order Book Gate
            "enable_order_book": self.enable_order_book_var.get(),
            "ob_depth": ob_depth,
            "ob_imbalance_threshold": self.ob_imbalance_threshold_var.get(),
        }

    def load_config(self, config: Dict):
        """Load configuration into UI"""
        self.scan_interval_entry.delete(0, "end")
        self.scan_interval_entry.insert(0, str(config.get("scan_interval", 5)))
        self.timeframe_var.set(config.get("timeframe", "1h"))
        self.sampling_strategy_var.set(config.get("sampling_strategy", "stratified"))
        self.sample_percentage_entry.delete(0, "end")
        self.sample_percentage_entry.insert(0, str(config.get("sample_percentage", 20)))
        self.auto_scan_startup_var.set(config.get("auto_start", True))

        # Load migrated filter fields
        self.min_score_var.set(config.get("min_signal_score", 0.7))
        if hasattr(self, "min_score_value_label"):
            self.min_score_value_label.configure(text=f"{self.min_score_var.get():.2f}")
        self.min_volume_entry.delete(0, "end")
        self.min_volume_entry.insert(0, str(config.get("min_volume", 50)))
        self.enable_xgboost_var.set(config.get("enable_xgboost", True))
        self.atc_threshold_var.set(config.get("atc_threshold", 0.6))
        if hasattr(self, "atc_value_label"):
            self.atc_value_label.configure(text=f"{self.atc_threshold_var.get():.2f}")

        # Load Gann Square settings
        self.enable_gann_square_var.set(config.get("enable_gann_square", False))
        self.gann_tf_var.set(config.get("gann_timeframe", "1h"))
        self.gann_candle_limit_entry.delete(0, "end")
        self.gann_candle_limit_entry.insert(0, str(config.get("gann_candle_limit", 200)))
        if hasattr(self, "gann_lookback_entry"):
            self.gann_lookback_entry.delete(0, "end")
            self.gann_lookback_entry.insert(0, str(config.get("gann_lookback", 5)))
        if self.enable_gann_square_var.get():
            self.gann_sub_frame.grid()
        else:
            self.gann_sub_frame.grid_remove()

        # Load Order Book Gate settings
        self.enable_order_book_var.set(config.get("enable_order_book", False))
        if hasattr(self, "ob_depth_entry"):
            self.ob_depth_entry.delete(0, "end")
            self.ob_depth_entry.insert(0, str(config.get("ob_depth", 20)))
        if hasattr(self, "ob_imbalance_threshold_var"):
            self.ob_imbalance_threshold_var.set(config.get("ob_imbalance_threshold", 0.2))
        if hasattr(self, "ob_threshold_value_label"):
            self.ob_threshold_value_label.configure(text=f"{self.ob_imbalance_threshold_var.get():.2f}")
        if self.enable_order_book_var.get():
            self.ob_sub_frame.grid()
        else:
            self.ob_sub_frame.grid_remove()

        # Update settings display
        self.setting_interval.configure(text=f"{config.get('scan_interval', 5)} min")
        self.setting_timeframe.configure(text=config.get("timeframe", "1h"))
        self.setting_strategy.configure(text=config.get("sampling_strategy", "stratified"))
        self.setting_percentage.configure(text=f"{config.get('sample_percentage', 20)}%")
