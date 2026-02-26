"""Logs Section Component for Database Panel."""

from datetime import datetime

import customtkinter as ctk

from modules.auto_trade.gui.config.database_panel_config import DatabasePanelConfig
from modules.auto_trade.gui.utils.svg_icons import get_icon


class LogsSection:
    """Activity logs section component."""

    def __init__(self, parent: ctk.CTkFrame):
        self.parent = parent
        self.logs_viewer: ctk.CTkTextbox = None
        self._create_ui()

    def _create_ui(self):
        """Create the logs section UI."""
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            header,
            text="  Activity Logs",
            font=DatabasePanelConfig.TITLE_FONT,
            image=get_icon("file_text", size=(20, 20)),
            compound="left",
        ).pack(side="left")

        ctk.CTkButton(header, text="Clear", width=60, height=24, command=self.clear).pack(side="right")

        self.logs_viewer = ctk.CTkTextbox(frame)
        self.logs_viewer.pack(fill="both", expand=True, padx=10, pady=5)

    def log(self, message: str, level: str = "INFO"):
        """Add a log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs_viewer.insert("end", f"[{timestamp}] [{level}] {message}\n")
        self.logs_viewer.see("end")

    def clear(self):
        """Clear all logs."""
        self.logs_viewer.delete("1.0", "end")
