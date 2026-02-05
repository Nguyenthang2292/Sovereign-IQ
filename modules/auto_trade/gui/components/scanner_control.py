from datetime import datetime
from typing import Callable, Dict

import customtkinter as ctk

from gui.utils.colors import Colors


class ScannerControl(ctk.CTkFrame):
    """
    Scanner Control Panel
    Control scanning operations, display status, and configure scanner settings
    """

    def __init__(self, parent, on_scan_toggle: Callable = None, on_config_change: Callable = None):
        super().__init__(parent)

        self.on_scan_toggle = on_scan_toggle
        self.on_config_change = on_config_change
        self.scanner_running = False

        # Title
        title = ctk.CTkLabel(self, text="🔍 Scanner Control", font=("Arial", 16, "bold"))
        title.pack(pady=(10, 15))

        # Status indicator
        self._create_status_indicator()

        # Control buttons
        self._create_controls()

        # Scanner configuration (includes settings display)
        self._create_configuration()

    def _create_status_indicator(self):
        """Create scanner status indicator"""
        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.pack(fill="x", padx=15, pady=10)

        # Status label with emoji
        self.status_label = ctk.CTkLabel(
            status_frame, text="🔴 Scanner: STOPPED", font=("Arial", 14, "bold"), text_color="gray"
        )
        self.status_label.pack()

        # Last scan timestamp
        self.last_scan_label = ctk.CTkLabel(
            status_frame, text="Last scan: Never", font=("Arial", 10), text_color="gray"
        )
        self.last_scan_label.pack(pady=(5, 0))

        # Scan progress
        self.progress_label = ctk.CTkLabel(status_frame, text="", font=("Arial", 10), text_color="#00ff88")
        self.progress_label.pack(pady=(2, 0))

    def _create_controls(self):
        """Create start/stop control buttons"""
        controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        controls_frame.pack(fill="x", padx=15, pady=10)

        # Start button
        self.start_button = ctk.CTkButton(
            controls_frame,
            text="▶️ Start Scanner",
            font=("Arial", 12, "bold"),
            fg_color="#00ff88",
            hover_color="#00cc66",
            command=self._start_scanner,
        )
        self.start_button.pack(fill="x", pady=(0, 8))

        # Stop button (hidden initially)
        self.stop_button = ctk.CTkButton(
            controls_frame,
            text="⏸️ Stop Scanner",
            font=("Arial", 12, "bold"),
            fg_color="#ff4444",
            hover_color="#cc0000",
            command=self._stop_scanner,
        )
        self.stop_button.pack(fill="x", pady=(0, 8))
        self.stop_button.pack_forget()  # Hide initially

        # Manual scan button
        self.manual_scan_button = ctk.CTkButton(
            controls_frame,
            text="🔄 Manual Scan",
            font=("Arial", 12),
            fg_color="#4488ff",
            hover_color="#0066ff",
            command=self._manual_scan,
        )
        self.manual_scan_button.pack(fill="x", pady=(0, 5))

    def _create_configuration(self):
        """Create scanner configuration and settings display side-by-side"""
        # Container frame for 2-column layout
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=15, pady=10)

        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)

        # ===== LEFT: Scanner Configuration =====
        config_frame = ctk.CTkFrame(container, fg_color=Colors.get_card_bg(), corner_radius=10)
        config_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Title
        config_title = ctk.CTkLabel(config_frame, text="⚙️ Scanner Configuration", font=("Arial", 12, "bold"))
        config_title.pack(pady=(10, 5))

        # Configuration inputs
        inputs_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        inputs_frame.pack(fill="x", padx=10, pady=(5, 10))

        # Scan interval
        interval_label = ctk.CTkLabel(
            inputs_frame, text="Scan Interval (minutes):", font=("Arial", 11), text_color="gray"
        )
        interval_label.grid(row=0, column=0, sticky="w", pady=5)

        self.scan_interval_entry = ctk.CTkEntry(inputs_frame, placeholder_text="5", width=150)
        self.scan_interval_entry.grid(row=0, column=1, sticky="e", pady=5, padx=(10, 0))
        self.scan_interval_entry.insert(0, "5")

        # Timeframe selector
        timeframe_label = ctk.CTkLabel(inputs_frame, text="Timeframe:", font=("Arial", 11), text_color="gray")
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
        strategy_label = ctk.CTkLabel(inputs_frame, text="Sampling Strategy:", font=("Arial", 11), text_color="gray")
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
        percentage_label = ctk.CTkLabel(inputs_frame, text="Sample (%):", font=("Arial", 11), text_color="gray")
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

        # Retrain XGBoost before scan
        self.retrain_xgboost_var = ctk.BooleanVar(value=False)
        retrain_xgboost_checkbox = ctk.CTkCheckBox(
            inputs_frame,
            text="Retrain XGBoost before scan",
            variable=self.retrain_xgboost_var,
            command=self._on_config_change,
        )
        retrain_xgboost_checkbox.grid(row=5, column=0, columnspan=2, sticky="w", pady=5)

        # Configure grid columns - column 0 for labels needs minimum width
        inputs_frame.grid_columnconfigure(0, weight=0, minsize=140)
        inputs_frame.grid_columnconfigure(1, weight=1)

        # ===== RIGHT: Current Settings =====
        settings_frame = ctk.CTkFrame(container, fg_color=Colors.get_card_bg(), corner_radius=10)
        settings_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # Title
        settings_title = ctk.CTkLabel(settings_frame, text="📊 Current Settings", font=("Arial", 12, "bold"))
        settings_title.pack(pady=(10, 5))

        # Settings list
        settings_list_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        settings_list_frame.pack(fill="x", padx=10, pady=(5, 10))

        settings = [
            ("Interval:", "5 min"),
            ("Timeframe:", "1h"),
            ("Strategy:", "stratified"),
            ("Sample:", "20%"),
            ("Status:", "Stopped"),
        ]

        for i, (label_text, value_text) in enumerate(settings):
            row_frame = ctk.CTkFrame(settings_list_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=2)

            label = ctk.CTkLabel(row_frame, text=label_text, font=("Arial", 10), text_color="gray")
            label.pack(side="left")

            value = ctk.CTkLabel(row_frame, text=value_text, font=("Arial", 10, "bold"))
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
            print(f"Error starting scanner: {e}")

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
            print(f"Error stopping scanner: {e}")

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
            print(f"Error triggering manual scan: {e}")

    def _update_status_indicator(self, running: bool):
        """Update status display"""
        if running:
            self.status_label.configure(text="🟢 Scanner: RUNNING", text_color="#00ff88")
            self.setting_status.configure(text="Running", text_color="#00ff88")
            self._animate_status()
        else:
            self.status_label.configure(text="🔴 Scanner: STOPPED", text_color="gray")
            self.setting_status.configure(text="Stopped", text_color="gray")

    def _animate_status(self):
        """Pulse animation when running"""
        if not self.scanner_running:
            return

        current_text = self.status_label.cget("text")
        new_text = "🟢 Scanner: SCANNING" if "RUNNING" in current_text else "🟢 Scanner: RUNNING"
        self.status_label.configure(text=new_text)

        self.after(1000, self._animate_status)

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
            print(f"Error updating config: {e}")

    def update_last_scan_time(self):
        """Update last scan timestamp"""
        now = datetime.now().strftime("%H:%M:%S")
        self.last_scan_label.configure(text=f"Last scan: {now}")

    def get_config(self) -> Dict:
        """Get current scanner configuration"""
        return {
            "scan_interval": int(self.scan_interval_entry.get()),
            "timeframe": self.timeframe_var.get(),
            "sampling_strategy": self.sampling_strategy_var.get(),
            "sample_percentage": float(self.sample_percentage_entry.get()),
            "auto_start": self.auto_scan_startup_var.get(),
            "retrain_xgboost": self.retrain_xgboost_var.get(),
            "running": self.scanner_running,
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
        self.retrain_xgboost_var.set(config.get("retrain_xgboost", False))

        # Update settings display
        self.setting_interval.configure(text=f"{config.get('scan_interval', 5)} min")
        self.setting_timeframe.configure(text=config.get("timeframe", "1h"))
        self.setting_strategy.configure(text=config.get("sampling_strategy", "stratified"))
        self.setting_percentage.configure(text=f"{config.get('sample_percentage', 20)}%")
