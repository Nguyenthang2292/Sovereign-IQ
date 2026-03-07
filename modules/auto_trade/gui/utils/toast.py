"""
Toast Notification Module

Provides temporary popup notifications for the GUI.
"""

from typing import Literal, Union

import customtkinter as ctk
from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.fonts import Fonts

# Any CTk widget that can host a toplevel (has winfo_rootx etc.)
ToastParent = Union[ctk.CTk, ctk.CTkFrame]

ToastType = Literal["info", "success", "error", "warning"]


class ToastNotification(ctk.CTkToplevel):
    """
    Temporary notification popup that appears at the bottom of the parent window.

    Features:
    - Auto-dismiss after duration
    - Fade-out animation
    - Click to dismiss
    - Color-coded by type
    """

    def __init__(
        self,
        parent: ToastParent,
        message: str,
        duration: int = 3000,
        fg_color: str = Colors.TOAST_INFO,
        text_color: str = Colors.WHITE,
    ) -> None:
        """
        Initialize toast notification.

        Args:
            parent: Parent window
            message: Message to display
            duration: Duration in milliseconds before auto-dismiss (default: 3000)
            fg_color: Background color
            text_color: Text color
        """
        super().__init__(parent)
        self.overrideredirect(True)

        # Calculate position (center bottom of parent)
        try:
            parent_x: int = parent.winfo_rootx()
            parent_y: int = parent.winfo_rooty()
            parent_width: int = parent.winfo_width()
            parent_height: int = parent.winfo_height()

            width: int = 300
            height: int = 50
            x: int = parent_x + (parent_width - width) // 2
            y: int = parent_y + parent_height - height - 50

            self.geometry(f"{width}x{height}+{x}+{y}")
        except (AttributeError, RuntimeError):
            # Fallback if parent geometry fails
            self.geometry("300x50")

        # Style
        self.configure(fg_color=fg_color)

        # Content
        self.label: ctk.CTkLabel = ctk.CTkLabel(
            self, text=message, font=Fonts.H2, text_color=text_color, fg_color=Colors.TRANSPARENT
        )
        self.label.pack(expand=True, fill="both", padx=20, pady=10)

        # Settings
        self.attributes("-alpha", 0.9)
        self.attributes("-topmost", True)

        # Auto close
        self.after(duration, self._fade_out)

        # Click to dismiss
        self.bind("<Button-1>", lambda e: self.destroy())
        self.label.bind("<Button-1>", lambda e: self.destroy())

    def _fade_out(self) -> None:
        """Fade out animation before closing."""
        try:
            alpha: float = float(self.attributes("-alpha"))
            if alpha > 0:
                alpha -= 0.1
                self.attributes("-alpha", alpha)
                self.after(50, self._fade_out)
            else:
                self.destroy()
        except (RuntimeError, ValueError):
            # Window already destroyed or attribute error
            pass


def show_toast(
    parent: ToastParent,
    message: str,
    type: ToastType = "info",
    duration: int = 3000,
) -> None:
    """
    Show a toast notification.

    Args:
        parent: Parent window
        message: Message to display
        type: Notification type ("info", "success", "error", "warning")
        duration: Duration in milliseconds before auto-dismiss
    """
    colors = {
        "info": Colors.TOAST_INFO,
        "success": Colors.SUCCESS_DIM,
        "error": Colors.TOAST_ERROR,
        "warning": Colors.TOAST_WARNING,
    }
    color = colors.get(type, Colors.TOAST_INFO)
    ToastNotification(parent, message, duration, fg_color=color)
