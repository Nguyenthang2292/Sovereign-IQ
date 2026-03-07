from typing import Dict

import customtkinter as ctk

from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.fonts import Fonts


class StatCard(ctk.CTkFrame):
    def __init__(self, parent, label: str, value: str = "0.00", unit: str = "USDT", color: str = Colors.WHITE):
        super().__init__(
            parent,
            fg_color=Colors.get_card_bg(),
            corner_radius=0,
            border_width=1,
            border_color=Colors.BORDER_NEON,
        )

        self.label = ctk.CTkLabel(self, text=label, font=Fonts.H3, text_color=Colors.get_text_secondary())
        self.label.pack(pady=(10, 5))

        self.value_label = ctk.CTkLabel(self, text=f"{value} {unit}", font=Fonts.DATA, text_color=color)
        self.value_label.pack(pady=(0, 10))

    def update(self, value: str, color: str = Colors.WHITE):
        self.value_label.configure(text=value, text_color=color)


class AccountFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(
            parent,
            fg_color=Colors.get_card_bg(),
            corner_radius=0,
            border_width=1,
            border_color=Colors.BORDER_NEON,
        )

        title = ctk.CTkLabel(self, text="Account Overview", font=Fonts.H1)
        title.pack(pady=(10, 15))

        self._create_stats_grid()
        self._apply_highlights()
        self.bind("<Button-1>", self._set_active_border)

    def _create_stats_grid(self):
        stats_frame = ctk.CTkFrame(self, fg_color=Colors.TRANSPARENT)
        stats_frame.pack(fill="both", expand=True, padx=10, pady=10)

        for i in range(3):
            stats_frame.grid_columnconfigure(i, weight=1)

        self.balance_card = StatCard(stats_frame, "Balance")
        self.balance_card.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.available_card = StatCard(stats_frame, "Available")
        self.available_card.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.margin_card = StatCard(stats_frame, "Margin Used")
        self.margin_card.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.pnl_card = StatCard(stats_frame, "Unrealized P&L")
        self.pnl_card.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.daily_pnl_card = StatCard(stats_frame, "Daily P&L")
        self.daily_pnl_card.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.daily_pnl_percent_card = StatCard(stats_frame, "Daily P&L %")
        self.daily_pnl_percent_card.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

    def _apply_highlights(self):
        for card in (self.balance_card, self.pnl_card, self.daily_pnl_card):
            card.configure(fg_color=Colors.BG_HIGHLIGHT, border_color=Colors.BORDER_NEON)
            card.bind("<Button-1>", self._set_active_border)

    def _set_active_border(self, _event=None):
        self.configure(border_color=Colors.BORDER_ACTIVE)

    def update_data(self, account_data: Dict):
        from modules.auto_trade.gui.utils.formatters import format_percent, format_pnl

        self.balance_card.update(f"{account_data.get('balance', 0):,.2f} USDT")
        self.available_card.update(f"{account_data.get('available', 0):,.2f} USDT")
        self.margin_card.update(f"{account_data.get('margin_used', 0):,.2f} USDT")

        unrealized_pnl = account_data.get("unrealized_pnl", 0)
        pnl_color = Colors.PROFIT if unrealized_pnl >= 0 else Colors.LOSS
        self.pnl_card.update(format_pnl(unrealized_pnl), pnl_color)

        daily_pnl = account_data.get("daily_pnl", 0)
        daily_pnl_color = Colors.PROFIT if daily_pnl >= 0 else Colors.LOSS
        self.daily_pnl_card.update(format_pnl(daily_pnl), daily_pnl_color)

        daily_pnl_percent = account_data.get("daily_pnl_percent", 0)
        percent_color = Colors.PROFIT if daily_pnl_percent >= 0 else Colors.LOSS
        self.daily_pnl_percent_card.update(format_percent(daily_pnl_percent), percent_color)
