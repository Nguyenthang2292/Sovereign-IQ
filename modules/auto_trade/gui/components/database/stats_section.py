"""Stats Section Component for Database Panel."""

import customtkinter as ctk
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Callable

from modules.auto_trade.database import session_scope
from modules.auto_trade.database.models import Order, Signal, MartingaleChain, AuditLog
from modules.auto_trade.database.config import DEFAULT_BACKUP_DIR

logger = logging.getLogger(__name__)


class StatsSection:
    """Database statistics section component."""

    def __init__(self, parent: ctk.CTkFrame):
        self.parent = parent
        self.stats_labels: Dict[str, ctk.CTkLabel] = {}
        self._create_ui()

    def _create_ui(self):
        """Create the stats section UI."""
        frame = ctk.CTkFrame(self.parent)
        frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(frame, text="📊 Database Stats", font=("Roboto", 14, "bold")).pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        stats_items = [
            ("total_orders", "Total Orders"),
            ("open_positions", "Open Positions"),
            ("total_signals", "Total Signals"),
            ("active_chains", "Active Chains"),
            ("audit_logs", "Audit Logs"),
            ("last_backup", "Last Backup"),
        ]

        for key, label in stats_items:
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=2)

            ctk.CTkLabel(row, text=f"{label}:").pack(side="left")
            value_label = ctk.CTkLabel(row, text="...")
            value_label.pack(side="right")

            self.stats_labels[key] = value_label

    def refresh(self):
        """Refresh statistics from database."""
        try:
            with session_scope() as session:
                # Count records
                total_orders = session.query(Order).count()
                open_positions = session.query(Order).filter(Order.status == "OPEN").count()
                total_signals = session.query(Signal).count()
                active_chains = session.query(MartingaleChain).filter(MartingaleChain.status == "ACTIVE").count()
                audit_logs = session.query(AuditLog).count()

                # Update labels
                if "total_orders" in self.stats_labels:
                    self.stats_labels["total_orders"].configure(text=str(total_orders))
                    self.stats_labels["open_positions"].configure(text=str(open_positions))
                    self.stats_labels["total_signals"].configure(text=str(total_signals))
                    self.stats_labels["active_chains"].configure(text=str(active_chains))
                    self.stats_labels["audit_logs"].configure(text=str(audit_logs))

                # Check last backup
                try:
                    backup_dir = Path(DEFAULT_BACKUP_DIR)
                    if backup_dir.exists():
                        backups = sorted(list(backup_dir.glob("*.db")), key=lambda f: f.stat().st_mtime, reverse=True)
                        if backups and "last_backup" in self.stats_labels:
                            last_backup_time = datetime.fromtimestamp(backups[0].stat().st_mtime).strftime(
                                "%Y-%m-%d %H:%M"
                            )
                            self.stats_labels["last_backup"].configure(text=last_backup_time)
                        elif "last_backup" in self.stats_labels:
                            self.stats_labels["last_backup"].configure(text="None")
                    elif "last_backup" in self.stats_labels:
                        self.stats_labels["last_backup"].configure(text="None")
                except Exception:
                    if "last_backup" in self.stats_labels:
                        self.stats_labels["last_backup"].configure(text="Error")

        except Exception as e:
            logger.error(f"Failed to refresh stats: {e}")
