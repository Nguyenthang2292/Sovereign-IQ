"""Keyboard shortcuts help dialog - lists all implemented shortcuts."""

import customtkinter as ctk

from modules.auto_trade.gui.utils.shortcuts import SHORTCUTS_LIST
from modules.auto_trade.gui.utils.windows_utils import apply_dark_titlebar


class ShortcutsHelpDialog(ctk.CTkToplevel):
    """Dialog showing the list of keyboard shortcuts implemented in the app."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Keyboard Shortcuts")
        self.geometry("560x420")
        self.minsize(400, 300)
        self.transient(parent)
        
        apply_dark_titlebar(self)

        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (560 // 2)
        y = (self.winfo_screenheight() // 2) - (420 // 2)
        self.geometry(f"560x420+{x}+{y}")

        self._create_ui()
        self.bind("<Escape>", lambda e: self.destroy())
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _create_ui(self):
        header = ctk.CTkLabel(
            self,
            text="⌨ Keyboard Shortcuts",
            font=("Arial", 16, "bold"),
        )
        header.pack(pady=(16, 8))

        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        # Group by context
        by_context: dict = {}
        for key_display, description, context in SHORTCUTS_LIST:
            by_context.setdefault(context, []).append((key_display, description))

        for context in ["Global", "Dashboard", "Scanner", "Trading", "Settings", "Database"]:
            if context not in by_context:
                continue
            section_label = ctk.CTkLabel(
                scroll,
                text=context,
                font=("Arial", 12, "bold"),
                text_color=("gray50", "gray70"),
            )
            section_label.pack(anchor="w", pady=(12, 4))

            for key_display, description in by_context[context]:
                row = ctk.CTkFrame(scroll, fg_color="transparent")
                row.pack(fill="x", pady=2)

                key_lbl = ctk.CTkLabel(
                    row,
                    text=key_display,
                    font=("Consolas", 11, "bold"),
                    width=120,
                    anchor="w",
                )
                key_lbl.pack(side="left", padx=(0, 12))

                desc_lbl = ctk.CTkLabel(
                    row,
                    text=description,
                    font=("Arial", 11),
                    anchor="w",
                )
                desc_lbl.pack(side="left", fill="x", expand=True)

        close_btn = ctk.CTkButton(
            self,
            text="Close",
            width=100,
            command=self.destroy,
        )
        close_btn.pack(pady=(0, 16))
