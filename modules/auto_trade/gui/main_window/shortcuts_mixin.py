"""Keyboard shortcut handlers for Auto Trade Dashboard."""

import customtkinter as ctk

from modules.auto_trade.gui.dialogs import ShortcutsHelpDialog
from modules.auto_trade.gui.utils.shortcuts import is_editable_focus
from modules.common.ui.logging import log_debug, log_info


class KeyboardShortcutsMixin:
    """Provide keyboard shortcut setup and handlers."""

    def _setup_keyboard_shortcuts(self):
        """Set up keyboard shortcuts (bind_all so they work regardless of focus)."""
        self.bind_all("<F1>", lambda e: self._show_shortcuts_help())
        self.bind_all("<Control-r>", lambda e: self._handle_refresh())
        self.bind_all("<F5>", lambda e: self._handle_refresh())
        self.bind_all("<Escape>", lambda e: self._handle_escape())
        self.bind_all("<Control-s>", lambda e: self._handle_save())
        self.bind_all("<Control-Key-1>", lambda e: self._handle_tab_switch(0))
        self.bind_all("<Control-Key-2>", lambda e: self._handle_tab_switch(1))
        self.bind_all("<Control-Key-3>", lambda e: self._handle_tab_switch(2))
        self.bind_all("<Control-Key-4>", lambda e: self._handle_tab_switch(3))
        self.bind_all("<Control-Key-5>", lambda e: self._handle_tab_switch(4))
        self.bind_all("<Control-Return>", lambda e: self._handle_confirm_trade(e))
        self.bind_all("<Control-c>", lambda e: self._handle_copy_selection(e))

        log_info(
            "Keyboard shortcuts: F1 (shortcuts help), Ctrl+R/F5, Esc, Ctrl+S, "
            "Ctrl+1..5 (tabs), Ctrl+Enter (trade), Ctrl+C (copy in DB)"
        )

    def _handle_refresh(self):
        """Handle refresh keyboard shortcut."""
        log_info("Refresh triggered by keyboard shortcut")
        self.refresh_signals()
        self.refresh_positions()
        self.refresh_account()
        self.refresh_stats()
        if hasattr(self, "status_bar"):
            self.status_bar.set_last_update()
        return "break"

    def _handle_escape(self):
        """Handle escape key - close any open dialogs."""
        for widget in self.winfo_children():
            if isinstance(widget, ctk.CTkToplevel):
                widget.destroy()
                log_debug("Closed dialog via Escape key")
                break
        return "break"

    def _handle_save(self):
        """Handle Ctrl+S - apply and save settings."""
        self.on_apply_settings()
        return "break"

    def _show_shortcuts_help(self):
        """Open keyboard shortcuts help dialog."""
        ShortcutsHelpDialog(self)
        return "break"

    def _handle_tab_switch(self, index: int):
        """Switch to tab by index."""
        tabs = ["Dashboard", "Scanner", "Trading", "Settings", "Database"]
        if 0 <= index < len(tabs) and hasattr(self, "tabview"):
            self.tabview.set(tabs[index])
        return "break"

    def _handle_confirm_trade(self, event):
        """Open confirm trade dialog when Trading tab is active and focus not in editable."""
        if is_editable_focus(self.focus_get()):
            return
        if hasattr(self, "tabview") and self.tabview.get() == "Trading" and hasattr(self, "trade_form"):
            try:
                self.trade_form._confirm_trade()
            except Exception:
                pass
        return "break"

    def _handle_copy_selection(self, event):
        """Copy Data Viewer selection when Database tab is active and focus not in editable."""
        if is_editable_focus(self.focus_get()):
            return
        if hasattr(self, "tabview") and self.tabview.get() == "Database" and hasattr(self, "database_panel"):
            try:
                self.database_panel.copy_selection_to_clipboard()
            except Exception:
                pass
        return "break"
