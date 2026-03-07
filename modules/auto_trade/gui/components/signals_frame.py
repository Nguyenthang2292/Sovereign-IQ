import os
from tkinter import ttk
from typing import Callable, Dict, List, Optional

import customtkinter as ctk

from modules.auto_trade.gui.components.empty_state import EmptyState
from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.fonts import Fonts
from modules.auto_trade.gui.utils.svg_icons import get_icon


class SignalsFrame(ctk.CTkFrame):
    def __init__(self, parent, on_run_scanner_callback: Optional[Callable] = None):
        super().__init__(
            parent,
            fg_color=Colors.get_card_bg(),
            corner_radius=0,
            border_width=1,
            border_color=Colors.BORDER_NEON,
        )

        self._on_run_scanner_callback = on_run_scanner_callback

        # Store original signals for filtering
        self._all_signals: List[Dict] = []

        self._create_header()
        self._create_table()
        self._create_empty_state()
        self.bind("<Button-1>", lambda _e: self.configure(border_color=Colors.BORDER_ACTIVE))

        self.refresh_label = ctk.CTkLabel(
            self,
            text="Auto-refresh: 30s",
            font=Fonts.SMALL,
            text_color=Colors.TEXT_SECONDARY_DARK,
        )
        self.refresh_label.pack(pady=5)

    def _create_empty_state(self):
        # In pytest/headless runs, avoid Tk image handles that can outlive widget lifecycle.
        is_test_mode = bool(os.environ.get("PYTEST_CURRENT_TEST"))
        icon_img = None if is_test_mode else get_icon("satellite_dish", size=(64, 64))

        self.empty_state = EmptyState(
            self,
            icon=icon_img if icon_img else "📡",
            message="No signals yet",
            hint="Run the scanner to get live signals.",
            action_text="Run scanner",
            action_callback=self._on_run_scanner_callback,
        )
        self.empty_state.pack(fill="both", expand=True)
        self.empty_state.pack_forget()  # Initially hide empty state

    def _create_header(self):
        header = ctk.CTkFrame(self, fg_color=Colors.TRANSPARENT)
        header.pack(fill="x", padx=10, pady=(10, 0))

        title = ctk.CTkLabel(header, text="Live Signals", font=Fonts.H1, text_color=Colors.get_accent())
        title.pack(side="left")

        filters_frame = ctk.CTkFrame(header, fg_color=Colors.TRANSPARENT)
        filters_frame.pack(side="right")

        self.filter_long = ctk.CTkCheckBox(
            filters_frame,
            text="LONG",
            command=self.apply_filters,
            font=Fonts.SMALL,
            text_color=Colors.LONG,
            checkmark_color=Colors.BLACK,  # Black tick visible on green bg
            border_color=Colors.BTN_SUCCESS,
            fg_color=Colors.BTN_SUCCESS,
            hover_color=Colors.BTN_SUCCESS_HOVER,
        )
        self.filter_long.pack(side="left", padx=5)
        self.filter_long.select()

        self.filter_short = ctk.CTkCheckBox(
            filters_frame,
            text="SHORT",
            command=self.apply_filters,
            font=Fonts.SMALL,
            text_color=Colors.SHORT,
            checkmark_color=Colors.ACCENT,  # Matrix neon green tick on red bg
            border_color=Colors.BTN_DANGER,
            fg_color=Colors.BTN_DANGER,
            hover_color=Colors.BTN_DANGER_HOVER,
        )
        self.filter_short.pack(side="left", padx=5)
        self.filter_short.select()

        score_label = ctk.CTkLabel(
            filters_frame,
            text="Min Score:",
            font=Fonts.SMALL,
            text_color=Colors.get_text_secondary(),
        )
        score_label.pack(side="left", padx=(10, 5))

        self.min_score = ctk.CTkEntry(
            filters_frame,
            width=60,
            font=Fonts.INPUT,
            text_color=Colors.get_text_primary(),
            fg_color=Colors.BG_INPUT,
            border_color=Colors.BORDER_NEON,
        )
        self.min_score.insert(0, "0.7")
        self.min_score.pack(side="left")
        # Bind Enter key to apply filters when user changes min score
        self.min_score.bind("<Return>", lambda e: self.apply_filters())

    def _create_table(self):
        self.table_frame = ctk.CTkFrame(
            self,
            fg_color=Colors.BG_HIGHLIGHT,
            border_width=1,
            border_color=Colors.BORDER_NEON,
            corner_radius=0,
        )
        self.table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar = ctk.CTkScrollbar(self.table_frame)
        scrollbar.pack(side="right", fill="y")

        columns = ("Symbol", "Signal", "Score", "Time")
        self.table = ttk.Treeview(
            self.table_frame, columns=columns, show="headings", yscrollcommand=scrollbar.set, height=10
        )

        self.table.heading("Symbol", text="Symbol")
        self.table.heading("Signal", text="Signal")
        self.table.heading("Score", text="Score")
        self.table.heading("Time", text="Time")

        self.table.column("Symbol", width=100)
        self.table.column("Signal", width=80)
        self.table.column("Score", width=80)
        self.table.column("Time", width=100)

        self.table.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=self.table.yview)

        self._configure_table_tags()

    def _configure_table_tags(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Theme-aware colors
        bg_color = Colors.get_card_bg()
        header_bg = Colors.get_header_bg()
        text_color = Colors.get_text_primary()
        # Matrix neon green hover for headings (made brighter for visibility)
        heading_hover_bg = Colors.BTN_NEUTRAL_HOVER  # Brighter hover background
        selected_bg = Colors.ACCENT_DIM

        style.configure(
            "Treeview",
            background=bg_color,
            foreground=text_color,
            fieldbackground=bg_color,
            font=(Fonts.FAMILY, 11),
            rowheight=24,
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background=header_bg,
            foreground=text_color,
            font=(Fonts.FAMILY, 11, "bold"),
            relief="flat",
            borderwidth=0,
        )
        # Override default gray hover on headings with Matrix green
        style.map(
            "Treeview.Heading",
            background=[
                ("active", heading_hover_bg),
                ("pressed", Colors.ACCENT_DIM),
            ],
            foreground=[
                ("active", Colors.ACCENT),
                ("pressed", Colors.ACCENT),
            ],
            relief=[("active", "flat"), ("pressed", "flat")],
        )
        # Matrix row selection colors
        style.map(
            "Treeview",
            background=[("selected", selected_bg)],
            foreground=[("selected", Colors.ACCENT)],
        )

    def update_signals(self, signals: List[Dict]):
        """Update signals display with new data."""
        # Store all signals for filtering
        self._all_signals = signals

        # Apply current filters and display
        self.apply_filters()

    def apply_filters(self):
        """Apply filters to signals and update display."""
        # Get filter values
        show_long = self.filter_long.get()
        show_short = self.filter_short.get()

        try:
            min_score = float(self.min_score.get())
        except ValueError:
            min_score = 0.0

        # Filter signals
        filtered_signals = []
        for signal in self._all_signals:
            signal_type = signal["signal"].upper()
            signal_score = signal.get("score", 0.0)

            # Apply signal type filter
            if signal_type == "LONG" and not show_long:
                continue
            if signal_type == "SHORT" and not show_short:
                continue

            # Apply score filter
            if signal_score < min_score:
                continue

            filtered_signals.append(signal)

        # Update table display
        self._display_signals(filtered_signals)

    def _display_signals(self, signals: List[Dict]):
        """Display signals in the table."""
        if not signals:
            self.table_frame.pack_forget()
            self.empty_state.pack(fill="both", expand=True)
            return

        self.empty_state.pack_forget()
        self.table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Clear existing items
        for item in self.table.get_children():
            self.table.delete(item)

        # Insert filtered signals
        for signal in signals:
            tag = signal["signal"].lower()
            self.table.insert(
                "",
                "end",
                values=(signal["symbol"], signal["signal"], f"{signal['score']:.2f}", signal["time"]),
                tags=(tag,),
            )

        # Configure tag colors
        self.table.tag_configure("long", foreground=Colors.PROFIT)
        self.table.tag_configure("short", foreground=Colors.LOSS)
        self.table.tag_configure("neutral", foreground=Colors.NEUTRAL)
