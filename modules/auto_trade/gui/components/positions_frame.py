import tkinter as tk
from typing import Callable, Dict, List, Optional

import customtkinter as ctk

from modules.auto_trade.gui.components.empty_state import EmptyState
from modules.auto_trade.gui.components.position_details import PositionDetails
from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.formatters import format_pnl, format_price


class PositionCard(ctk.CTkFrame):
    def __init__(self, parent, position: Dict, on_action_callback: Optional[Callable] = None):
        super().__init__(parent, fg_color=Colors.get_card_bg(), corner_radius=10)
        self.position = position
        self.on_action_callback = on_action_callback

        # Make the card clickable
        self.bind("<Button-1>", self._on_click)
        self.bind("<Button-3>", self._show_context_menu)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))
        header.bind("<Button-1>", self._on_click)
        header.bind("<Button-3>", self._show_context_menu)

        symbol_label = ctk.CTkLabel(header, text=position["symbol"], font=("Arial", 14, "bold"))
        symbol_label.pack(side="left")
        symbol_label.bind("<Button-1>", self._on_click)
        symbol_label.bind("<Button-3>", self._show_context_menu)

        side_color = Colors.LONG if position["side"] == "LONG" else Colors.SHORT
        side_label = ctk.CTkLabel(header, text=position["side"], font=("Arial", 12, "bold"), text_color=side_color)
        side_label.pack(side="right")
        side_label.bind("<Button-1>", self._on_click)
        side_label.bind("<Button-3>", self._show_context_menu)

        self._create_details(position)
        self._create_context_menu()

    def _create_details(self, position: Dict):
        details_frame = ctk.CTkFrame(self, fg_color="transparent")
        details_frame.pack(fill="x", padx=10, pady=5)
        details_frame.bind("<Button-1>", self._on_click)
        details_frame.bind("<Button-3>", self._show_context_menu)

        # Format TP/SL/BE with None handling
        tp_value = position.get("take_profit")
        sl_value = position.get("stop_loss")
        be_value = position.get("break_even")

        rows = [
            ("Size:", f"{position['size']:.4f}"),
            ("Entry:", format_price(position["entry_price"])),
            ("Current:", format_price(position["current_price"])),
            ("TP:", format_price(float(tp_value)) if tp_value is not None else "N/A"),
            ("SL:", format_price(float(sl_value)) if sl_value is not None else "N/A"),
            ("BE:", format_price(float(be_value)) if be_value is not None else "N/A"),
            ("P&L:", format_pnl(position["pnl"])),
        ]

        for i, (label, value) in enumerate(rows):
            label_widget = ctk.CTkLabel(details_frame, text=label, font=("Arial", 11), text_color="gray")
            label_widget.grid(row=i, column=0, sticky="w", pady=2)
            label_widget.bind("<Button-1>", self._on_click)
            label_widget.bind("<Button-3>", self._show_context_menu)

            # Color coding for different rows
            if i == 6:  # P&L row
                text_color = Colors.PROFIT if position["pnl"] >= 0 else Colors.LOSS
            elif i == 3:  # TP row
                text_color = Colors.PROFIT if value != "N/A" else "gray"
            elif i == 4:  # SL row
                text_color = Colors.LOSS if value != "N/A" else "gray"
            elif i == 5:  # BE row
                text_color = "#FFA500" if value != "N/A" else "gray"  # Orange for BE
            else:
                text_color = "white"

            value_widget = ctk.CTkLabel(details_frame, text=value, font=("Arial", 11, "bold"), text_color=text_color)
            value_widget.grid(row=i, column=1, sticky="e", pady=2)
            value_widget.bind("<Button-1>", self._on_click)
            value_widget.bind("<Button-3>", self._show_context_menu)

        details_frame.grid_columnconfigure(1, weight=1)

    def _create_context_menu(self):
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(label="Close Position", command=self._quick_close)
        self.context_menu.add_command(label="Modify TP/SL", command=self._modify_tp_sl)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Copy ID", command=self._copy_id)

    def _show_context_menu(self, event):
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def _quick_close(self):
        # Open details for now, acting as quick access
        PositionDetails(self.winfo_toplevel(), self.position, on_action_callback=self.on_action_callback)

    def _modify_tp_sl(self):
        # Open details for now
        PositionDetails(self.winfo_toplevel(), self.position, on_action_callback=self.on_action_callback)

    def _copy_id(self):
        self.clipboard_clear()
        self.clipboard_append(self.position.get("id", "Unknown"))
        self.update()

    def _on_click(self, event):
        """Handle click event to show position details"""
        try:
            PositionDetails(self.winfo_toplevel(), self.position, on_action_callback=self.on_action_callback)
        except Exception as e:
            print(f"Error opening position details: {e}")


class PositionsFrame(ctk.CTkFrame):
    def __init__(self, parent, on_action_callback: Optional[Callable] = None, on_open_trade_callback: Optional[Callable] = None, on_refresh_callback: Optional[Callable] = None):
        super().__init__(parent)
        self.on_action_callback = on_action_callback
        self.on_open_trade_callback = on_open_trade_callback
        self.on_refresh_callback = on_refresh_callback
        self._empty_state: Optional[EmptyState] = None

        # Header with title and refresh button
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", pady=(10, 5), padx=10)

        title = ctk.CTkLabel(header_frame, text="Open Positions", font=("Arial", 16, "bold"))
        title.pack(side="left")

        if on_refresh_callback:
            refresh_btn = ctk.CTkButton(
                header_frame,
                text="🔄 Refresh",
                width=80,
                height=24,
                command=on_refresh_callback,
                fg_color="#2B6CB0",
                hover_color="#3182CE"
            )
            refresh_btn.pack(side="right")

        self.scroll_frame = ctk.CTkScrollableFrame(self, height=300)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def update_positions(self, positions: List[Dict]):
        print(f"[PositionsFrame] update_positions called with {len(positions) if positions else 0} positions")
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        if not positions:
            self._empty_state = EmptyState(
                self.scroll_frame,
                icon="📭",
                message="No open positions",
                hint="Open a trade or wait for a signal.",
                action_text="Open Trade" if self.on_open_trade_callback else "",
                action_callback=self.on_open_trade_callback
            )
            self._empty_state.pack(pady=50, fill="x")
            return

        for position in positions:
            card = PositionCard(self.scroll_frame, position, on_action_callback=self.on_action_callback)
            card.pack(fill="x", padx=5, pady=5)
