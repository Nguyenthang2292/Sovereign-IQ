from typing import Callable

import customtkinter as ctk

from gui.utils.colors import Colors


class AutoTradeControl(ctk.CTkFrame):
    """
    Auto-trading enable/disable control
    Shows status, allows toggle, displays current settings
    """

    def __init__(self, parent, on_toggle_callback: Callable = None):
        super().__init__(parent)

        self.on_toggle_callback = on_toggle_callback
        self.auto_trade_enabled = False

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

    def _create_settings_display(self):
        """Display current auto-trade configuration"""
        settings_frame = ctk.CTkFrame(self, fg_color=Colors.get_card_bg(), corner_radius=10)
        settings_frame.pack(fill="x", padx=15, pady=10)

        # Title
        settings_title = ctk.CTkLabel(settings_frame, text="⚙️ Current Settings", font=("Arial", 12, "bold"))
        settings_title.pack(pady=(10, 5))

        # Settings list (scrollable so all settings fit)
        settings_list_frame = ctk.CTkScrollableFrame(settings_frame, fg_color="transparent", height=420)
        settings_list_frame.pack(fill="x", padx=10, pady=(5, 10))

        # Organized settings by sections
        settings_sections = [
            {
                "title": "💰 Risk Management",
                "settings": [
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
                    ("ATC Threshold:", "0.60", "atc_threshold"),
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
        if not getattr(self, "settings_labels", None):
            return
        risk = settings.get("risk", {})
        filters = settings.get("filters", {})
        tp_sl = settings.get("tp_sl", {})
        api = settings.get("api", {})
        recovery = settings.get("recovery", {})
        labels = self.settings_labels
        # Risk & filters & TP/SL
        if "min_score" in labels:
            v = filters.get("min_signal_score", 0.7)
            labels["min_score"].configure(text=f"{float(v):.2f}")
        if "max_position_size" in labels:
            v = risk.get("max_position_size", 100.0)
            labels["max_position_size"].configure(text=f"${float(v):.0f} USDT")
        if "max_open_positions" in labels:
            v = risk.get("max_open_positions", 3)
            labels["max_open_positions"].configure(text=str(int(v)))
        if "max_daily_loss" in labels:
            v = risk.get("max_daily_loss", 50.0)
            labels["max_daily_loss"].configure(text=f"${float(v):.0f} USDT")
        if "default_leverage" in labels:
            labels["default_leverage"].configure(text=str(risk.get("default_leverage", "10x")))
        if "default_tp" in labels:
            v = tp_sl.get("default_tp", 5.0)
            labels["default_tp"].configure(text=f"{float(v):.1f}%")
        if "default_sl" in labels:
            v = tp_sl.get("default_sl", 2.5)
            labels["default_sl"].configure(text=f"{float(v):.1f}%")
        if "tp_sl_mode" in labels:
            labels["tp_sl_mode"].configure(text=str(tp_sl.get("mode", "Percentage")))
        if "trailing_stop" in labels:
            labels["trailing_stop"].configure(text="On" if tp_sl.get("trailing_stop", False) else "Off")
        if "atc_threshold" in labels:
            v = filters.get("atc_threshold", 0.6)
            labels["atc_threshold"].configure(text=f"{float(v):.2f}")
        if "enable_xgboost" in labels:
            labels["enable_xgboost"].configure(text="On" if filters.get("enable_xgboost", True) else "Off")
        if "min_volume" in labels:
            v = filters.get("min_volume", 50.0)
            labels["min_volume"].configure(text=str(int(float(v))))
        if "timeframe" in labels:
            labels["timeframe"].configure(text=str(filters.get("timeframe", "1h")))
        # Gradual Recovery
        if "recovery_enabled" in labels:
            enabled = recovery.get("enabled", False)
            labels["recovery_enabled"].configure(text="On" if enabled else "Off")
        if "recovery_initial_loss" in labels:
            v = recovery.get("initial_loss", 500.0)
            labels["recovery_initial_loss"].configure(text=f"${float(v):.0f}")
        if "recovery_target_profit" in labels:
            v = recovery.get("target_profit_per_trade", 5.0)
            labels["recovery_target_profit"].configure(text=f"${float(v):.1f}")
        if "recovery_max_trades" in labels:
            v = recovery.get("max_recovery_trades", 20)
            labels["recovery_max_trades"].configure(text=str(int(v)))
        if "recovery_margin_mode" in labels:
            labels["recovery_margin_mode"].configure(text=str(recovery.get("margin_scaling_mode", "fixed")))
        if "recovery_leverage_mode" in labels:
            labels["recovery_leverage_mode"].configure(text=str(recovery.get("leverage_scaling_mode", "fixed")))
        if "recovery_streak_bonus" in labels:
            on_off = "On" if recovery.get("enable_streak_bonus", False) else "Off"
            labels["recovery_streak_bonus"].configure(text=on_off)
        # Status (database, API) — from optional status dict
        st = status or {}
        if "database_status" in labels:
            labels["database_status"].configure(text=str(st.get("database", "—")))
        if "api_mode" in labels:
            labels["api_mode"].configure(text=str(st.get("api_mode", api.get("mode", "DRY_RUN"))))
        if "api_connection" in labels:
            labels["api_connection"].configure(text=str(st.get("api_connection", "—")))
