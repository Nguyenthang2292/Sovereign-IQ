"""
Logs Viewer Component

Simple log viewer that opens log file externally and displays live stream logs.
"""

import os
import subprocess
import sys
from pathlib import Path

import customtkinter as ctk


class LogsViewer(ctk.CTkFrame):
    """Simple logs viewer with button to open log file and live stream display."""

    def __init__(self, parent, log_file_path: str):
        """
        Initialize LogsViewer.

        Args:
            parent: Parent widget
            log_file_path: Path to log file
        """
        super().__init__(parent)

        self.log_file_path = Path(log_file_path)

        self._create_ui()

    def _create_ui(self):
        """Create simple UI."""
        # Title
        title = ctk.CTkLabel(self, text="System Logs", font=("Arial", 16, "bold"))
        title.pack(pady=(20, 10))

        # Info text
        info = ctk.CTkLabel(
            self,
            text=f"Logs are saved to:\n{self.log_file_path}",
            font=("Arial", 11),
            text_color="gray",
            justify="center",
        )
        info.pack(pady=10)

        # Button frame
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=20)

        # Open log file button
        open_btn = ctk.CTkButton(
            btn_frame,
            text="Open Log File",
            width=150,
            command=self._open_log_file,
        )
        open_btn.pack(side="left", padx=5)

        # Open folder button
        folder_btn = ctk.CTkButton(
            btn_frame,
            text="Open Folder",
            width=150,
            fg_color="#555555",
            hover_color="#666666",
            command=self._open_log_folder,
        )
        folder_btn.pack(side="left", padx=5)

        # Clear logs button
        clear_btn = ctk.CTkButton(
            btn_frame,
            text="🗑️ Clear Logs",
            width=150,
            fg_color="#ff6644",
            hover_color="#cc4422",
            command=self.clear_logs,
        )
        clear_btn.pack(side="left", padx=5)

        # Status label
        self.status_label = ctk.CTkLabel(
            self,
            text="Click 'Open Log File' to view logs in your text editor",
            font=("Arial", 10),
            text_color="gray",
        )
        self.status_label.pack(pady=10)

        # Live Logs Display
        logs_label = ctk.CTkLabel(
            self,
            text="📡 Live Stream Logs:",
            font=("Arial", 12, "bold"),
            anchor="w",
        )
        logs_label.pack(fill="x", padx=20, pady=(10, 5))

        # Scrollable text area for logs
        self.logs_textbox = ctk.CTkTextbox(
            self,
            height=200,
            font=("Consolas", 10),
            wrap="word",
        )
        self.logs_textbox.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Insert initial message
        self.logs_textbox.insert("1.0", "🟢 Log stream ready. Waiting for logs...\n")
        self.logs_textbox.configure(state="disabled")  # Read-only

        # Max lines to keep (prevent memory overflow)
        self.max_log_lines = 500

    def _open_log_file(self):
        """Open log file in default text editor."""
        try:
            if not self.log_file_path.exists():
                self.status_label.configure(text="Log file not found", text_color="#ff6666")
                return

            if sys.platform == "win32":
                os.startfile(str(self.log_file_path))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(self.log_file_path)])
            else:
                subprocess.run(["xdg-open", str(self.log_file_path)])

            self.status_label.configure(text="Log file opened", text_color="#66ff66")
        except Exception as e:
            self.status_label.configure(text=f"Error: {e}", text_color="#ff6666")

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

            self.status_label.configure(text="Folder opened", text_color="#66ff66")
        except Exception as e:
            self.status_label.configure(text=f"Error: {e}", text_color="#ff6666")

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
            print(f"Error appending log: {e}")

    def clear_logs(self):
        """Clear all logs from the textbox."""
        try:
            self.logs_textbox.configure(state="normal")
            self.logs_textbox.delete("1.0", "end")
            self.logs_textbox.insert("1.0", "🟢 Logs cleared. Waiting for new logs...\n")
            self.logs_textbox.configure(state="disabled")
        except Exception as e:
            print(f"Error clearing logs: {e}")
