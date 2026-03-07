"""Martingale Section Component for Database Panel."""

from typing import Callable

import customtkinter as ctk

from modules.auto_trade.database.repository.context import RepositoryContext
from modules.auto_trade.gui.config.database_panel_config import DatabasePanelConfig
from modules.auto_trade.gui.utils.svg_icons import get_icon


class MartingaleSection:
    """Martingale testing section component."""

    def __init__(self, parent: ctk.CTkFrame | ctk.CTkScrollableFrame, log_callback: Callable):
        self.parent = parent
        self.log_callback = log_callback
        self._create_ui()

    def _create_ui(self):
        """Create the martingale section UI."""
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(
            frame,
            text="  Martingale Testing",
            font=DatabasePanelConfig.TITLE_FONT,
            image=get_icon("repeat", size=(20, 20)),
            compound="left",
        ).pack(
            anchor="w",
            padx=DatabasePanelConfig.PADX_MEDIUM,
            pady=(DatabasePanelConfig.PADX_MEDIUM, DatabasePanelConfig.PADY_SMALL),
        )

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(
            btn_frame,
            text="  Get Active Chains",
            command=self._get_active_chains,
            image=get_icon("link", size=(16, 16)),
            compound="left",
        ).pack(side="left", padx=(0, 5), fill="x", expand=True)

        ctk.CTkButton(
            btn_frame,
            text="  Chain Statistics",
            command=self._get_chain_stats,
            image=get_icon("bar_chart_2", size=(16, 16)),
            compound="left",
        ).pack(side="left", padx=(5, 0), fill="x", expand=True)

    def _get_active_chains(self):
        """Get active martingale chains via RepositoryContext."""
        try:
            ctx = RepositoryContext.from_env()
            chains = ctx.martingale.get_active_martingale_chains()

            output = "Active Martingale Chains:\n"
            output += "-" * 60 + "\n"
            for chain in chains:
                if isinstance(chain, dict):
                    chain_id = chain.get("chain_id", "")
                    symbol = chain.get("symbol", "")
                    step = chain.get("current_step", 0)
                    max_steps = chain.get("max_allowed_steps", 0)
                    total_loss = float(chain.get("total_loss", 0) or 0)
                    total_recovery = float(chain.get("total_recovery", 0) or 0)
                else:
                    chain_id = chain.chain_id
                    symbol = chain.symbol
                    step = chain.current_step
                    max_steps = chain.max_allowed_steps
                    total_loss = float(chain.total_loss or 0)
                    total_recovery = float(chain.total_recovery or 0)
                output += (
                    f"ID: {chain_id} | {symbol} | Step: {step}/{max_steps} | PnL: {total_loss + total_recovery:.2f}\n"
                )

            self._show_in_data_viewer(output)
            self.log_callback(f"Retrieved {len(chains)} active chains", "INFO")

        except Exception as e:
            self.log_callback(f"Failed to get active chains: {e}", "ERROR")

    def _get_chain_stats(self):
        """Get martingale chain statistics via RepositoryContext."""
        try:
            ctx = RepositoryContext.from_env()
            active_chains = ctx.martingale.get_active_martingale_chains()
            all_chains = ctx.martingale.get_martingale_chains_cursor(limit=99999)

            total = len(all_chains)
            active = len(active_chains)
            completed = sum(
                1
                for c in all_chains
                if (c.get("status") if isinstance(c, dict) else getattr(c, "status", "")) == "RECOVERED"
            )
            success_rate = (completed / total * 100) if total > 0 else 0

            output = "Martingale Chain Statistics:\n"
            output += "=" * 30 + "\n"
            output += f"Total Chains: {total}\n"
            output += f"Active Chains: {active}\n"
            output += f"Completed (Recovered): {completed}\n"
            output += f"Success Rate: {success_rate:.2f}%\n"

            self._show_in_data_viewer(output)
            self.log_callback("Retrieved chain stats", "INFO")

        except Exception as e:
            self.log_callback(f"Failed to get chain stats: {e}", "ERROR")

    def _show_in_data_viewer(self, content: str) -> None:
        """Show content in data viewer."""
        callback = getattr(self.parent, "data_viewer_callback", None)
        if callable(callback):
            callback(content)
