"""Keyboard shortcuts registry and helpers for Auto Trade GUI."""

from typing import List, Tuple

# (key_display, description, context)
# context: "Global" | "Dashboard" | "Scanner" | "Trading" | "Settings" | "Database"
SHORTCUTS_LIST: List[Tuple[str, str, str]] = [
    # Global
    ("F1", "Show keyboard shortcuts", "Global"),
    ("Ctrl+R", "Refresh data", "Global"),
    ("F5", "Refresh data", "Global"),
    ("Ctrl+S", "Apply / save settings", "Global"),
    ("Escape", "Close dialog", "Global"),
    ("Ctrl+1", "Switch to Dashboard tab", "Global"),
    ("Ctrl+2", "Switch to Scanner tab", "Global"),
    ("Ctrl+3", "Switch to Trading tab", "Global"),
    ("Ctrl+4", "Switch to Settings tab", "Global"),
    ("Ctrl+5", "Switch to Database tab", "Global"),
    # Scanner
    ("Ctrl+M", "Manual scan (when Scanner tab active)", "Scanner"),
    # Trading
    ("Ctrl+Enter", "Confirm trade / open confirm dialog (when Trading tab active)", "Trading"),
    # Database
    ("Ctrl+C", "Copy selection in Data Viewer (when Database tab active, focus not in edit)", "Database"),
]


def is_editable_focus(widget) -> bool:
    """Return True if the focused widget is an editable control (Entry, Text, etc.)."""
    if widget is None:
        return False
    w = widget.winfo_class() if hasattr(widget, "winfo_class") else ""
    # CTkEntry and CTkTextbox report as Tk entry/text or customtkinter class
    try:
        name = (getattr(widget, "__class__", None) or type(widget)).__name__
    except Exception:
        name = ""
    editable_classes = (
        "Entry",
        "CTkEntry",
        "Text",
        "CTkTextbox",
        "Spinbox",
    )
    return w in editable_classes or any(editable in name for editable in editable_classes)
