import customtkinter as ctk
from typing import Callable, Optional


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
        settings_frame = ctk.CTkFrame(self, fg_color="#2b2b2b", corner_radius=10)
        settings_frame.pack(fill="x", padx=15, pady=10)

        # Title
        settings_title = ctk.CTkLabel(settings_frame, text="⚙️ Current Settings", font=("Arial", 12, "bold"))
        settings_title.pack(pady=(10, 5))

        # Settings list
        settings_list_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        settings_list_frame.pack(fill="x", padx=10, pady=(5, 10))

        settings = [
            ("Min Score:", "0.7"),
            ("Max Position Size:", "$10 USDT"),
            ("Max Open Positions:", "3"),
            ("Default Leverage:", "10x"),
            ("Default TP:", "5%"),
            ("Default SL:", "2.5%"),
        ]

        for label_text, value_text in settings:
            row_frame = ctk.CTkFrame(settings_list_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=2)

            label = ctk.CTkLabel(row_frame, text=label_text, font=("Arial", 10), text_color="gray")
            label.pack(side="left")

            value = ctk.CTkLabel(row_frame, text=value_text, font=("Arial", 10, "bold"))
            value.pack(side="right")
