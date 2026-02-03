import customtkinter as ctk
from typing import List, Dict
from gui.utils.colors import Colors
from gui.utils.formatters import format_pnl, format_price


class PositionCard(ctk.CTkFrame):
    def __init__(self, parent, position: Dict):
        super().__init__(parent, fg_color="gray20", corner_radius=10)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))

        symbol_label = ctk.CTkLabel(header, text=position["symbol"], font=("Arial", 14, "bold"))
        symbol_label.pack(side="left")

        side_color = Colors.LONG if position["side"] == "LONG" else Colors.SHORT
        side_label = ctk.CTkLabel(header, text=position["side"], font=("Arial", 12, "bold"), text_color=side_color)
        side_label.pack(side="right")

        self._create_details(position)

    def _create_details(self, position: Dict):
        details_frame = ctk.CTkFrame(self, fg_color="transparent")
        details_frame.pack(fill="x", padx=10, pady=5)

        rows = [
            ("Size:", f"{position['size']:.4f}"),
            ("Entry:", format_price(position["entry_price"])),
            ("Current:", format_price(position["current_price"])),
            ("P&L:", format_pnl(position["pnl"])),
        ]

        for i, (label, value) in enumerate(rows):
            label_widget = ctk.CTkLabel(details_frame, text=label, font=("Arial", 11), text_color="gray")
            label_widget.grid(row=i, column=0, sticky="w", pady=2)

            pnl_color = Colors.PROFIT if i == 3 and position["pnl"] >= 0 else (Colors.LOSS if i == 3 else "white")
            value_widget = ctk.CTkLabel(details_frame, text=value, font=("Arial", 11, "bold"), text_color=pnl_color)
            value_widget.grid(row=i, column=1, sticky="e", pady=2)

        details_frame.grid_columnconfigure(1, weight=1)


class PositionsFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        title = ctk.CTkLabel(self, text="Open Positions", font=("Arial", 16, "bold"))
        title.pack(pady=(10, 15))

        self.scroll_frame = ctk.CTkScrollableFrame(self, height=300)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.empty_label = ctk.CTkLabel(
            self.scroll_frame, text="No open positions", font=("Arial", 14), text_color="gray"
        )
        self.empty_label.pack(pady=50)

    def update_positions(self, positions: List[Dict]):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        if not positions:
            self.empty_label = ctk.CTkLabel(
                self.scroll_frame, text="No open positions", font=("Arial", 14), text_color="gray"
            )
            self.empty_label.pack(pady=50)
            return

        for position in positions:
            card = PositionCard(self.scroll_frame, position)
            card.pack(fill="x", padx=5, pady=5)
