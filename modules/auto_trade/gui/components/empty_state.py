from modules.common.ui.logging import log_info, log_error, log_warn, log_debug, log_success, log_system
from typing import Callable, Optional

import customtkinter as ctk


class EmptyState(ctk.CTkFrame):
    """Customizable empty-state widget for GUI.

    Displays an icon, a message, an optional hint, and an optional action button.
    Used by PositionsFrame, SignalsFrame, and other panels when there is no data.
    """

    def __init__(
        self,
        parent,
        icon: str,
        message: str,
        hint: Optional[str] = None,
        action_text: Optional[str] = None,
        action_callback: Optional[Callable] = None,
        **kwargs
    ):
        """Initialize the EmptyState component.

        Args:
            parent: Parent widget.
            icon: Icon to display (e.g. emoji or text).
            message: Main message text.
            hint: Optional secondary hint.
            action_text: Optional label for the action button.
            action_callback: Optional callable for the action button.
            **kwargs: Passed to CTkFrame.
        """
        super().__init__(parent, **kwargs)

        self.icon = icon
        self.message = message
        self.hint = hint
        self.action_text = action_text
        self.action_callback = action_callback

        self._create_widgets()

    def _create_widgets(self):
        # Icon
        self.icon_label = ctk.CTkLabel(
            self,
            text=self.icon,
            font=("Segoe UI", 48)
        )
        self.icon_label.pack(pady=(20, 10), padx=20)

        # Message
        self.message_label = ctk.CTkLabel(
            self,
            text=self.message,
            font=("Segoe UI", 16, "bold")
        )
        self.message_label.pack(pady=(0, 5), padx=20)

        # Hint (optional)
        if self.hint:
            self.hint_label = ctk.CTkLabel(
                self,
                text=self.hint,
                font=("Segoe UI", 12),
                text_color=("gray60", "gray40"),  # Adapts to light/dark mode
                wraplength=300
            )
            self.hint_label.pack(pady=(0, 10), padx=20)

        # Action Button (optional)
        if self.action_text and self.action_callback:
            self.action_button = ctk.CTkButton(
                self,
                text=self.action_text,
                command=self._on_action_button_click
            )
            self.action_button.pack(pady=(10, 20), padx=20)

    def _on_action_button_click(self):
        """Handle action button click with error handling."""
        if self.action_callback:
            try:
                self.action_callback()
            except Exception as e:
                log_error(f"Error in EmptyState action callback: {e}", exc_info=True)
