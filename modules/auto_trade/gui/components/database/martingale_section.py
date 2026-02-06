"""Martingale Section Component for Database Panel."""

import customtkinter as ctk
import logging
from typing import Callable

from modules.auto_trade.database import (
    session_scope,
    get_active_martingale_chains,
)
from modules.auto_trade.database.models import MartingaleChain

logger = logging.getLogger(__name__)


class MartingaleSection:
    """Martingale testing section component."""

    def __init__(self, parent: ctk.CTkFrame, log_callback: Callable):
        self.parent = parent
        self.log_callback = log_callback
        self._create_ui()

    def _create_ui(self):
        """Create the martingale section UI."""
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(frame, text="🔄 Martingale Testing", font=("Roboto", 14, "bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkButton(btn_frame, text="🔗 Get Active Chains", command=self._get_active_chains).pack(
            side="left", padx=(0, 5), fill="x", expand=True
        )
        ctk.CTkButton(btn_frame, text="📊 Chain Statistics", command=self._get_chain_stats).pack(
            side="left", padx=(5, 0), fill="x", expand=True
        )

    def _get_active_chains(self):
        """Get active martingale chains."""
        try:
            with session_scope() as session:
                chains = get_active_martingale_chains(session)

                output = "Active Martingale Chains:\n"
                output += "-" * 60 + "\n"
                for chain in chains:
                    output += f"ID: {chain.chain_id} | {chain.symbol} | Step: {chain.current_step}/{chain.max_allowed_steps} | PnL: {chain.total_loss + chain.total_recovery:.2f}\n"

                self._show_in_data_viewer(output)
                self.log_callback(f"Retrieved {len(chains)} active chains", "INFO")

        except Exception as e:
            self.log_callback(f"Failed to get active chains: {e}", "ERROR")

    def _get_chain_stats(self):
        """Get martingale chain statistics."""
        try:
            with session_scope() as session:
                total = session.query(MartingaleChain).count()
                active = session.query(MartingaleChain).filter(MartingaleChain.status == "ACTIVE").count()
                completed = session.query(MartingaleChain).filter(MartingaleChain.status == "RECOVERED").count()

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

    def _show_in_data_viewer(self, content: str):
        """Show content in data viewer."""
        if hasattr(self.parent, "data_viewer_callback"):
            self.parent.data_viewer_callback(content)
