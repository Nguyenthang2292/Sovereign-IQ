"""
Logs Viewer Component

Simple log viewer that opens log file externally and displays live stream logs.
"""

import os
import subprocess
import sys
from pathlib import Path

import customtkinter as ctk

from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.svg_icons import get_button_icon
from modules.auto_trade.gui.utils.fonts import Fonts
from modules.common.ui.logging import log_error


class LogsViewer(ctk.CTkFrame):
    """Simple logs viewer with button to open log file and live stream display."""

    def __init__(self, parent, log_file_path: str):
        """
        Initialize LogsViewer.

        Args:
            parent: Parent widget
            log_file_path: Path to log file
        """
        super().__init__(parent, fg_color=Colors.TRANSPARENT)

        self.log_file_path = Path(log_file_path)

        self._create_ui()

    def _create_ui(self):
        """
        Create simple UI layout.

        Live Stream left = Scanner Configuration width (1/2),
        System Logs right = Current Settings (1/2).
        """
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Add padding wrapper to match ScannerControl's layout
        wrapper = ctk.CTkFrame(self, fg_color=Colors.TRANSPARENT)
        wrapper.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=15, pady=10)
        wrapper.grid_columnconfigure(0, weight=1)
        wrapper.grid_columnconfigure(1, weight=1)
        wrapper.grid_rowconfigure(0, weight=1)

        # Left: Live Stream Logs (square box area)
        left_frame = ctk.CTkFrame(wrapper, fg_color=Colors.TRANSPARENT)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(1, weight=1)

        logs_label = ctk.CTkLabel(
            left_frame,
            text="📡 Live Stream Logs:",
            font=Fonts.H3,
            anchor="w",
        )
        logs_label.grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.logs_textbox = ctk.CTkTextbox(
            left_frame,
            font=("Consolas", 10),
            wrap="word",
        )
        self.logs_textbox.grid(row=1, column=0, sticky="nsew", pady=(0, 0))

        # Right: System Logs block (~1/3, aligned with Current Settings)
        system_logs_box = ctk.CTkFrame(
            wrapper,
            fg_color=Colors.get_card_bg(),
            corner_radius=0,
            border_width=1,
            border_color=Colors.BORDER_SUBTLE,
        )
        system_logs_box.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        inner = ctk.CTkFrame(system_logs_box, fg_color=Colors.TRANSPARENT)
        inner.pack(fill="x", padx=15, pady=15)

        title = ctk.CTkLabel(inner, text="System Logs", font=Fonts.H1)
        title.pack(pady=(0, 10))

        info = ctk.CTkLabel(
            inner,
            text=f"Logs are saved to:\n{self.log_file_path}",
            font=Fonts.BODY,
            text_color=Colors.TEXT_MUTED,
            justify="center",
        )
        info.pack(pady=10)

        # Stack buttons vertically for compact layout
        btn_frame = ctk.CTkFrame(inner, fg_color=Colors.TRANSPARENT)
        btn_frame.pack(pady=20)

        open_btn = ctk.CTkButton(
            btn_frame,
            text="OPEN LOG FILE",
            font=Fonts.BUTTON_SM,
            width=160,
            command=self._open_log_file,
            image=get_button_icon("file_text", size=(16, 16), variant="primary"),
            compound="left",
        )
        open_btn.pack(pady=(0, 8))

        folder_btn = ctk.CTkButton(
            btn_frame,
            text="OPEN FOLDER",
            font=Fonts.BUTTON_SM,
            width=160,
            fg_color=Colors.BTN_NEUTRAL,
            hover_color=Colors.TEXT_DISABLED,
            command=self._open_log_folder,
            image=get_button_icon("folder_open", size=(16, 16), variant="neutral"),
            compound="left",
        )
        folder_btn.pack(pady=8)

        clear_btn = ctk.CTkButton(
            btn_frame,
            text="CLEAR LOGS",
            font=Fonts.BUTTON_SM,
            width=160,
            fg_color=Colors.BTN_DANGER,
            hover_color=Colors.BTN_DANGER_HOVER,
            command=self.clear_logs,
            image=get_button_icon("trash", size=(16, 16), variant="danger"),
            compound="left",
        )
        clear_btn.pack(pady=(8, 0))

        self.status_label = ctk.CTkLabel(
            inner,
            text="Click 'Open Log File' to view logs in your text editor",
            font=Fonts.SMALL,
            text_color=Colors.TEXT_MUTED,
        )
        self.status_label.pack(pady=10)

        # Insert initial message
        self.logs_textbox.insert("1.0", "🟢 Log stream ready. Waiting for logs...\n")
        self.logs_textbox.configure(state="disabled")  # Read-only

        # Max lines to keep (prevent memory overflow)
        self.max_log_lines = 500

    def _open_log_file(self):
        """Open log file in default text editor."""
        try:
            if not self.log_file_path.exists():
                self.status_label.configure(text="Log file not found", text_color=Colors.BTN_DANGER_HOVER)
                return

            if sys.platform == "win32":
                os.startfile(str(self.log_file_path))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(self.log_file_path)])
            else:
                subprocess.run(["xdg-open", str(self.log_file_path)])

            self.status_label.configure(text="Log file opened", text_color=Colors.SUCCESS_BRIGHT)
        except Exception as e:
            self.status_label.configure(text=f"Error: {e}", text_color=Colors.BTN_DANGER_HOVER)

    def _open_log_folder(self):
        """Open folder containing log file."""
        try:
            folder = self.log_file_path.parent
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)

            if sys.platform == "win32":
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(folder)])
            else:
                subprocess.run(["xdg-open", str(folder)])

            self.status_label.configure(text="Folder opened", text_color=Colors.SUCCESS_BRIGHT)
        except Exception as e:
            self.status_label.configure(text=f"Error: {e}", text_color=Colors.BTN_DANGER_HOVER)

    def append_log(self, log_message: str):
        """
        Append a log message to the textbox.

        Args:
            log_message: Log message to append
        """
        try:
            # Enable editing temporarily
            self.logs_textbox.configure(state="normal")

            # Append log message
            self.logs_textbox.insert("end", log_message + "\n")

            # Limit number of lines
            lines = int(self.logs_textbox.index("end-1c").split(".")[0])
            if lines > self.max_log_lines:
                # Delete oldest lines
                self.logs_textbox.delete("1.0", f"{lines - self.max_log_lines}.0")

            # Auto-scroll to bottom
            self.logs_textbox.see("end")

            # Disable editing
            self.logs_textbox.configure(state="disabled")

        except Exception as e:
            log_error("Error appending log: %s", e)

    def clear_logs(self):
        """Clear all logs from the textbox."""
        try:
            self.logs_textbox.configure(state="normal")
            self.logs_textbox.delete("1.0", "end")
            self.logs_textbox.insert("1.0", "🟢 Logs cleared. Waiting for new logs...\n")
            self.logs_textbox.configure(state="disabled")
        except Exception as e:
            log_error("Error clearing logs: %s", e)
