"""Loading Overlay Component for async operations."""

from typing import Optional

import customtkinter as ctk


class LoadingOverlay:
    """Simple loading overlay for async operations."""

    def __init__(self, parent: ctk.CTk):
        """Initialize loading overlay.

        Args:
            parent: Parent widget to overlay on.
        """
        self.parent = parent
        self.overlay: Optional[ctk.CTkFrame] = None
        self.label: Optional[ctk.CTkLabel] = None

    def show(self, message: str = "Loading..."):
        """Show loading overlay.

        Args:
            message: Message to display in the overlay.
        """
        if self.overlay:
            return

        # Create semi-transparent overlay frame
        self.overlay = ctk.CTkFrame(
            self.parent,
            fg_color=("gray80", "gray20"),
            corner_radius=10,
        )
        self.overlay.place(relx=0.5, rely=0.5, anchor="center")

        # Add loading icon and message
        self.label = ctk.CTkLabel(
            self.overlay,
            text=f"⏳ {message}",
            font=("Arial", 16),
        )
        self.label.pack(padx=40, pady=20)

        # Force update to show immediately
        self.parent.update_idletasks()

    def hide(self):
        """Hide loading overlay."""
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None
            self.label = None

    def is_visible(self) -> bool:
        """Check if overlay is currently visible.

        Returns:
            True if overlay is showing, False otherwise.
        """
        return self.overlay is not None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - always hide overlay."""
        self.hide()
        return False  # Don't suppress exceptions
