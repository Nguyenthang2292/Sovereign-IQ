from typing import Any, Callable, Optional

import customtkinter as ctk

from modules.auto_trade.gui.utils.colors import Colors


class AutoTradeControl(ctk.CTkFrame):
    """
    Auto-trading enable/disable control
    Shows status, allows toggle, displays current settings
    """

    def __init__(
        self,
        parent: Any,
        on_toggle_callback: Optional[Callable[..., Any]] = None,
        on_reload_settings: Optional[Callable[..., Any]] = None,
        on_risk_limits_toggle: Optional[Callable[[bool], Any]] = None,
    ):
        super().__init__(parent)

        self.on_toggle_callback = on_toggle_callback
        self.on_reload_settings = on_reload_settings
        self.on_risk_limits_toggle = on_risk_limits_toggle
        self.auto_trade_enabled = False
        self._suppress_risk_toggle = False
        self.risk_limits_enabled_var = ctk.BooleanVar(value=True)

        # Title
        title = ctk.CTkLabel(self, text="🤖 Auto-Trade System", font=("Arial", 16, "bold"))
        title.pack(pady=(10, 15))

        # Status indicator
        self._create_status_indicator()

        # Control buttons
        self._create_controls()

        # Settings display
        self._create_settings_display()

    def _create_status_indicator(self):
        """Visual status indicator with animation"""
        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.pack(fill="x", padx=15, pady=10)

        # Status circle + text
        self.status_label = ctk.CTkLabel(
            status_frame, text="🔴 Auto-Trade: DISABLED", font=("Arial", 14, "bold"), text_color="gray"
        )
        self.status_label.pack()

        # Last action timestamp
        self.last_action_label = ctk.CTkLabel(
            status_frame, text="Last action: Never", font=("Arial", 10), text_color="gray"
        )
        self.last_action_label.pack(pady=(5, 0))

    def _update_status_indicator(self, enabled: bool):
        """Update status display"""
        if enabled:
            self.status_label.configure(text="🟢 Auto-Trade: ACTIVE", text_color="#00ff88")
            self._animate_status()
        else:
            self.status_label.configure(text="🔴 Auto-Trade: DISABLED", text_color="gray")

    def _animate_status(self):
        """Pulse animation when active"""
        if not self.auto_trade_enabled:
            return

        current_color = self.status_label.cget("text_color")
        new_color = "#00ff88" if current_color == "#00cc66" else "#00cc66"
        self.status_label.configure(text_color=new_color)

        self.after(1000, self._animate_status)

    def _create_controls(self):
        """Enable/Disable buttons"""
        controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        controls_frame.pack(fill="x", padx=15, pady=10)

        # Enable button
        self.enable_button = ctk.CTkButton(
            controls_frame,
            text="▶️ Enable Auto-Trade",
            font=("Arial", 12, "bold"),
            fg_color="#00ff88",
            hover_color="#00cc66",
            command=self._enable_auto_trade,
        )
        self.enable_button.pack(fill="x", pady=5)

        # Disable button (hidden initially)
        self.disable_button = ctk.CTkButton(
            controls_frame,
            text="⏸️ Disable Auto-Trade",
            font=("Arial", 12, "bold"),
            fg_color="#ff4444",
            hover_color="#cc0000",
            command=self._disable_auto_trade,
        )
        self.disable_button.pack(fill="x", pady=5)
        self.disable_button.pack_forget()  # Hide initially

        risk_limits_cb = ctk.CTkCheckBox(
            controls_frame,
            text="🛡️ Enable Risk Limits",
            variable=self.risk_limits_enabled_var,
            command=self._on_risk_limits_toggle,
        )
        risk_limits_cb.pack(anchor="w", pady=(8, 0))

    def _on_risk_limits_toggle(self):
        """Persist Risk Limits toggle change from Trading tab."""
        if self._suppress_risk_toggle:
            return

        enabled = bool(self.risk_limits_enabled_var.get())
        if self.on_risk_limits_toggle:
            self.on_risk_limits_toggle(enabled)

    def _enable_auto_trade(self):
        """Enable auto-trading system"""
        # Confirmation dialog
        from tkinter import messagebox

        confirm = messagebox.askyesno(
            "Enable Auto-Trade",
            "Enable automatic trading?\n\nThe system will execute trades based on signals.\n\nThis is REAL money!",
        )

        if not confirm:
            return

        try:
            self.auto_trade_enabled = True

            # Update UI
            self.enable_button.pack_forget()
            self.disable_button.pack(fill="x", pady=5)
            self._update_status_indicator(True)

            # Call callback
            if self.on_toggle_callback:
                self.on_toggle_callback(True)

            # Update last action
            from datetime import datetime

            self.last_action_label.configure(text=f"Last action: Enabled at {datetime.now().strftime('%H:%M:%S')}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to enable auto-trade: {e}")

    def _disable_auto_trade(self):
        """Disable auto-trading system"""
        try:
            self.auto_trade_enabled = False

            # Update UI
            self.disable_button.pack_forget()
            self.enable_button.pack(fill="x", pady=5)
            self._update_status_indicator(False)

            # Call callback
            if self.on_toggle_callback:
                self.on_toggle_callback(False)

            # Update last action
            from datetime import datetime

            self.last_action_label.configure(text=f"Last action: Disabled at {datetime.now().strftime('%H:%M:%S')}")

        except Exception as e:
            from tkinter import messagebox

            messagebox.showerror("Error", f"Failed to disable auto-trade: {e}")

    def _on_reload_settings(self):
        """Trigger reload of Current Settings from main window (settings_manager + status)."""
        if self.on_reload_settings:
            self.on_reload_settings()

    def _create_settings_display(self):
        """Display current auto-trade configuration"""
        settings_frame = ctk.CTkFrame(self, fg_color=Colors.get_card_bg(), corner_radius=10)
        settings_frame.pack(fill="x", padx=15, pady=10)

        # Title row: "Current Settings" + Force reload button
        title_row = ctk.CTkFrame(settings_frame, fg_color="transparent")
        title_row.pack(fill="x", padx=10, pady=(10, 2))
        title_row.grid_columnconfigure(0, weight=1)
        settings_title = ctk.CTkLabel(title_row, text="⚙️ Current Settings", font=("Arial", 12, "bold"))
        settings_title.grid(row=0, column=0, sticky="w")
        self.reload_settings_btn = ctk.CTkButton(
            title_row,
            text="🔄 Force reload",
            font=("Arial", 10),
            width=100,
            height=28,
            fg_color="#1f538d",
            hover_color="#2a6bb5",
            command=self._on_reload_settings,
        )
        self.reload_settings_btn.grid(row=0, column=1, padx=(8, 0), sticky="e")

        # Settings list (scrollable so all settings fit)
        settings_list_frame = ctk.CTkScrollableFrame(settings_frame, fg_color="transparent", height=420)
        settings_list_frame.pack(fill="x", padx=10, pady=(5, 10))

        # Organized settings by sections
        settings_sections = [
            {
                "title": "💰 Risk Management",
                "settings": [
                    ("Risk Limits:", "On", "risk_limits_enabled"),
                    ("Min Score:", "0.7", "min_score"),
                    ("Max Position Size:", "$10 USDT", "max_position_size"),
                    ("Max Open Positions:", "3", "max_open_positions"),
                    ("Max Daily Loss:", "$50 USDT", "max_daily_loss"),
                    ("Default Leverage:", "10x", "default_leverage"),
                ],
            },
            {
                "title": "🎯 TP/SL Settings",
                "settings": [
                    ("Default TP:", "5%", "default_tp"),
                    ("Default SL:", "2.5%", "default_sl"),
                    ("TP/SL Mode:", "Percentage", "tp_sl_mode"),
                    ("Trailing Stop:", "Off", "trailing_stop"),
                ],
            },
            {
                "title": "🔍 Filters",
                "settings": [
                    ("ATC threshold (base):", "0.60", "atc_threshold"),
                    ("XGBoost:", "On", "enable_xgboost"),
                    ("Min Volume (M):", "50", "min_volume"),
                    ("Timeframe:", "1h", "timeframe"),
                ],
            },
            {
                "title": "🔄 Gradual Recovery",
                "settings": [
                    ("Recovery:", "Off", "recovery_enabled"),
                    ("Initial Loss:", "$500", "recovery_initial_loss"),
                    ("Target/Trade:", "$5", "recovery_target_profit"),
                    ("Max Recovery Trades:", "20", "recovery_max_trades"),
                    ("Margin Scaling:", "fixed", "recovery_margin_mode"),
                    ("Leverage Scaling:", "fixed", "recovery_leverage_mode"),
                    ("Streak Bonus:", "Off", "recovery_streak_bonus"),
                ],
            },
            {
                "title": "📡 Status",
                "settings": [
                    ("Database:", "—", "database_status"),
                    ("API Mode:", "DRY_RUN", "api_mode"),
                    ("API Connection:", "—", "api_connection"),
                ],
            },
        ]

        self.settings_labels = {}

        for section in settings_sections:
            # Section container with subtle background
            section_frame = ctk.CTkFrame(settings_list_frame, fg_color=Colors.get_hover_bg(), corner_radius=8)
            section_frame.pack(fill="x", pady=(0, 8), padx=2)

            # Section header
            header = ctk.CTkLabel(
                section_frame, text=section["title"], font=("Arial", 11, "bold"), text_color=Colors.get_accent()
            )
            header.pack(anchor="w", padx=10, pady=(8, 5))

            # Section settings
            for label_text, value_text, key in section["settings"]:
                row_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
                row_frame.pack(fill="x", padx=10, pady=2)

                ctk.CTkLabel(row_frame, text=label_text, font=("Arial", 10), text_color="gray").pack(side="left")

                value = ctk.CTkLabel(row_frame, text=value_text, font=("Arial", 10, "bold"))
                value.pack(side="right")
                self.settings_labels[key] = value

            # Bottom padding for section
            ctk.CTkLabel(section_frame, text="", height=5).pack()

    def update_from_settings(self, settings: dict, status: dict | None = None):
        """Refresh Current Settings from settings (risk, filters, tp_sl, recovery, api)
        and optional status (database, api_connection)."""
        labels = getattr(self, "settings_labels", None)
        if labels is None:
            return

        # Validate that labels contains CTkLabel widgets, not other types
        if not isinstance(labels, dict):
            return

        risk = settings.get("risk", {})
        filters = settings.get("filters", {})
        tp_sl = settings.get("tp_sl", {})
        api = settings.get("api", {})
        recovery = settings.get("recovery", {})

        def _safe_configure(key: str, text_value: str):
            """Safely configure a label, checking it's a valid widget first."""
            if key not in labels:
                return
            label = labels[key]
            # Ensure label is a CTkLabel widget with configure method
            if hasattr(label, "configure") and callable(label.configure):
                label.configure(text=text_value)

        # Risk & filters & TP/SL
        _safe_configure("min_score", f"{float(filters.get('min_signal_score', 0.7)):.2f}")
        risk_limits_enabled = bool(risk.get("limits_enabled", True))
        _safe_configure("risk_limits_enabled", "On" if risk_limits_enabled else "Off")
        _safe_configure("max_position_size", f"${float(risk.get('max_position_size', 100.0)):.0f} USDT")
        _safe_configure("max_open_positions", str(int(risk.get("max_open_positions", 3))))
        _safe_configure("max_daily_loss", f"${float(risk.get('max_daily_loss', 50.0)):.0f} USDT")
        _safe_configure("default_leverage", str(risk.get("default_leverage", "10x")))
        _safe_configure("default_tp", f"{float(tp_sl.get('default_tp', 5.0)):.1f}%")
        _safe_configure("default_sl", f"{float(tp_sl.get('default_sl', 2.5)):.1f}%")
        _safe_configure("tp_sl_mode", str(tp_sl.get("mode", "Percentage")))
        _safe_configure("trailing_stop", "On" if tp_sl.get("trailing_stop", False) else "Off")
        _safe_configure("atc_threshold", f"{float(filters.get('atc_threshold', 0.6)):.2f}")
        _safe_configure("enable_xgboost", "On" if filters.get("enable_xgboost", True) else "Off")
        _safe_configure("min_volume", str(int(float(filters.get("min_volume", 50.0)))))
        _safe_configure("timeframe", str(filters.get("timeframe", "1h")))

        # Gradual Recovery
        _safe_configure("recovery_enabled", "On" if recovery.get("enabled", False) else "Off")
        _safe_configure("recovery_initial_loss", f"${float(recovery.get('initial_loss', 500.0)):.0f}")
        _safe_configure("recovery_target_profit", f"${float(recovery.get('target_profit_per_trade', 5.0)):.1f}")
        _safe_configure("recovery_max_trades", str(int(recovery.get("max_recovery_trades", 20))))
        _safe_configure("recovery_margin_mode", str(recovery.get("margin_scaling_mode", "fixed")))
        _safe_configure("recovery_leverage_mode", str(recovery.get("leverage_scaling_mode", "fixed")))
        _safe_configure("recovery_streak_bonus", "On" if recovery.get("enable_streak_bonus", False) else "Off")

        # Status (database, API) — from optional status dict
        st = status or {}
        _safe_configure("database_status", str(st.get("database", "—")))
        _safe_configure("api_mode", str(st.get("api_mode", api.get("mode", "DRY_RUN"))))
        _safe_configure("api_connection", str(st.get("api_connection", "—")))

        self._suppress_risk_toggle = True
        self.risk_limits_enabled_var.set(risk_limits_enabled)
        self._suppress_risk_toggle = False
