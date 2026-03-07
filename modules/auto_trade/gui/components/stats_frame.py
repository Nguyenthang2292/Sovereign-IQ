from typing import Dict

import customtkinter as ctk

from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.fonts import Fonts
from modules.auto_trade.gui.utils.modes import TradingMode


class ModeIndicator(ctk.CTkFrame):
    def __init__(self, parent, mode: str):
        super().__init__(parent, fg_color=Colors.TRANSPARENT)

        if mode == TradingMode.PRODUCTION:
            mode_text = "PRODUCTION"
            mode_color = Colors.PRODUCTION
        elif mode == TradingMode.DRY_RUN:
            mode_text = "DRY RUN"
            mode_color = Colors.DRY_RUN
        else:
            mode_text = "DEMO"
            mode_color = Colors.DEMO

        self.indicator = ctk.CTkLabel(self, text=f"{mode_text}", font=Fonts.H2, text_color=mode_color)
        self.indicator.pack()

        self.animate()

    def animate(self):
        if self.winfo_exists():
            current_color = self.indicator.cget("text_color")
            mode_colors = {
                Colors.PRODUCTION: [Colors.LOSS, Colors.BTN_DANGER_HOVER],
                Colors.DEMO: [Colors.BTN_WARNING, Colors.WARNING_BRIGHT],
                Colors.DRY_RUN: [Colors.ACCENT, Colors.TEXT_BRIGHT],
            }

            for key in mode_colors:
                if current_color in mode_colors[key]:
                    colors = mode_colors[key]
                    new_color = colors[1] if current_color == colors[0] else colors[0]
                    self.indicator.configure(text_color=new_color)
                    break

            self.after(1000, self.animate)


class StatsFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(
            parent,
            fg_color=Colors.get_card_bg(),
            border_width=1,
            border_color=Colors.BORDER_NEON,
        )

        title = ctk.CTkLabel(self, text="Quick Stats", font=Fonts.H1, text_color=Colors.get_accent())
        title.pack(pady=(10, 15))

        self._create_stats()

    def _create_stats(self):
        stats_frame = ctk.CTkFrame(self, fg_color=Colors.TRANSPARENT)
        stats_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.positions_label = ctk.CTkLabel(
            stats_frame,
            text="Open Positions: 0",
            font=Fonts.BODY,
            text_color=Colors.get_text_primary(),
        )
        self.positions_label.pack(anchor="w", pady=5)

        self.trades_label = ctk.CTkLabel(
            stats_frame,
            text="Today's Trades: 0",
            font=Fonts.BODY,
            text_color=Colors.get_text_primary(),
        )
        self.trades_label.pack(anchor="w", pady=5)

        self.winrate_label = ctk.CTkLabel(
            stats_frame,
            text="Win Rate: 0%",
            font=Fonts.BODY,
            text_color=Colors.get_text_primary(),
        )
        self.winrate_label.pack(anchor="w", pady=5)

        mode_frame = ctk.CTkFrame(
            stats_frame,
            fg_color=Colors.BG_HIGHLIGHT,
            border_width=1,
            border_color=Colors.BORDER_NEON,
            corner_radius=0,
        )
        mode_frame.pack(fill="x", pady=(20, 10))

        ctk.CTkLabel(
            mode_frame,
            text="CURRENT MODE",
            font=Fonts.SMALL,
            text_color=Colors.get_text_secondary(),
        ).pack(pady=(10, 5))
        self.mode_indicator = ModeIndicator(mode_frame, "DEMO")
        self.mode_indicator.pack(pady=(0, 10))

    def update_data(self, stats_data: Dict):
        self.positions_label.configure(text=f"Open Positions: {stats_data.get('open_positions', 0)}")
        self.trades_label.configure(text=f"Today's Trades: {stats_data.get('today_trades', 0)}")
        winrate = stats_data.get("win_rate", 0)
        self.winrate_label.configure(text=f"Win Rate: {winrate:.1f}%")
