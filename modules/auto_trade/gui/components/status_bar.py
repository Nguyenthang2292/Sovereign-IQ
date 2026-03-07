"""Status Bar Component for Auto Trade Dashboard."""

from datetime import datetime
from typing import Optional

import customtkinter as ctk

from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.fonts import Fonts


class StatusBar(ctk.CTkFrame):
    """Bottom status bar with connection status, last update time, and mode."""

    def __init__(self, parent, mode: str = "DRY_RUN"):
        """Initialize status bar.

        Args:
            parent: Parent widget.
            mode: Current trading mode (DRY_RUN, DEMO, PRODUCTION).
        """
        super().__init__(
            parent,
            height=30,
            fg_color=Colors.BG_CARD_DARK,
            border_width=1,
            border_color=Colors.BORDER_NEON,
        )

        # Configure grid
        self.grid_columnconfigure(0, weight=1)  # Left: Connection status
        self.grid_columnconfigure(1, weight=1)  # Center: Mode
        self.grid_columnconfigure(2, weight=1)  # Right: Last update

        # Connection status indicator
        self.connection_label = ctk.CTkLabel(
            self,
            text="● Disconnected",
            text_color=Colors.LOSS,
            font=Fonts.SMALL,
        )
        self.connection_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        # Mode indicator - Matrix themed colors per mode
        mode_color = self._get_mode_color(mode)
        mode_display = mode.replace("_", " ")

        self.mode_label = ctk.CTkLabel(
            self,
            text=f"Mode: {mode_display}",
            font=Fonts.H3,
            text_color=mode_color,
        )
        self.mode_label.grid(row=0, column=1, pady=5)

        # Last update time
        self.last_update_label = ctk.CTkLabel(
            self,
            text="Last update: Never",
            font=Fonts.SMALL,
            text_color=Colors.TEXT_SECONDARY_DARK,
        )
        self.last_update_label.grid(row=0, column=2, sticky="e", padx=10, pady=5)

    def _get_mode_color(self, mode: str) -> str:
        """Return Matrix-themed color for each trading mode."""
        return {
            "DRY_RUN": Colors.DRY_RUN,  # Matrix green
            "DEMO": Colors.DEMO,  # Amber
            "PRODUCTION": Colors.PRODUCTION,  # Red
        }.get(mode, Colors.TEXT_SECONDARY_DARK)

    def set_connection_status(self, connected: bool, message: Optional[str] = None):
        """Update connection status indicator.

        Args:
            connected: True if connected, False otherwise.
            message: Optional custom message to display.
        """
        if connected:
            text = message or "● Connected"
            color = Colors.LONG  # Matrix neon green
        else:
            text = message or "● Disconnected"
            color = Colors.LOSS  # Red

        self.connection_label.configure(text=text, text_color=color)

    def set_last_update(self, timestamp: Optional[datetime] = None):
        """Update last update time.

        Args:
            timestamp: Timestamp of last update. Defaults to current time.
        """
        if timestamp is None:
            timestamp = datetime.now()

        time_str = timestamp.strftime("%H:%M:%S")
        self.last_update_label.configure(text=f"Last update: {time_str}")

    def set_mode(self, mode: str):
        """Update mode indicator.

        Args:
            mode: Current trading mode.
        """
        color = self._get_mode_color(mode)
        mode_display = mode.replace("_", " ")
        self.mode_label.configure(text=f"Mode: {mode_display}", text_color=color)
