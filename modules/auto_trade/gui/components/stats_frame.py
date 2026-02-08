from typing import Dict

import customtkinter as ctk

from modules.auto_trade.gui.utils.colors import Colors
from modules.auto_trade.gui.utils.modes import TradingMode


class ModeIndicator(ctk.CTkFrame):
    def __init__(self, parent, mode: str):
        super().__init__(parent, fg_color="transparent")

        if mode == TradingMode.PRODUCTION:
            mode_text = "PRODUCTION"
            mode_color = Colors.PRODUCTION
        elif mode == TradingMode.DRY_RUN:
            mode_text = "DRY RUN"
            mode_color = Colors.DRY_RUN
        else:
            mode_text = "DEMO"
            mode_color = Colors.DEMO

        self.indicator = ctk.CTkLabel(self, text=f"{mode_text}", font=("Arial", 14, "bold"), text_color=mode_color)
        self.indicator.pack()

        self.animate()

    def animate(self):
        if self.winfo_exists():
            current_color = self.indicator.cget("text_color")
            mode_colors = {
                Colors.PRODUCTION: ["#ff4444", "#ff6666"],
                Colors.DEMO: ["#ffaa00", "#ffcc00"],
                Colors.DRY_RUN: ["#4488ff", "#66aaff"],
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
        super().__init__(parent)

        title = ctk.CTkLabel(self, text="Quick Stats", font=("Arial", 16, "bold"))
        title.pack(pady=(10, 15))

        self._create_stats()

    def _create_stats(self):
        stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        stats_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.positions_label = ctk.CTkLabel(stats_frame, text="Open Positions: 0", font=("Arial", 14))
        self.positions_label.pack(anchor="w", pady=5)

        self.trades_label = ctk.CTkLabel(stats_frame, text="Today's Trades: 0", font=("Arial", 14))
        self.trades_label.pack(anchor="w", pady=5)

        self.winrate_label = ctk.CTkLabel(stats_frame, text="Win Rate: 0%", font=("Arial", 14))
        self.winrate_label.pack(anchor="w", pady=5)

        mode_frame = ctk.CTkFrame(stats_frame, fg_color=Colors.get_card_bg(), corner_radius=10)
        mode_frame.pack(fill="x", pady=(20, 10))

        ctk.CTkLabel(mode_frame, text="Current Mode", font=("Arial", 11), text_color=Colors.get_text_secondary()).pack(
            pady=(10, 5)
        )
        self.mode_indicator = ModeIndicator(mode_frame, "DEMO")
        self.mode_indicator.pack(pady=(0, 10))

    def update_data(self, stats_data: Dict):
        self.positions_label.configure(text=f"Open Positions: {stats_data.get('open_positions', 0)}")
        self.trades_label.configure(text=f"Today's Trades: {stats_data.get('today_trades', 0)}")
        winrate = stats_data.get("win_rate", 0)
        self.winrate_label.configure(text=f"Win Rate: {winrate:.1f}%")
