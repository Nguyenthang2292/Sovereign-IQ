"""Stats Section Component for Database Panel."""

from modules.common.ui.logging import log_info, log_error, log_warn, log_debug, log_success, log_system
from typing import Dict

import customtkinter as ctk

from modules.auto_trade.gui.config.database_panel_config import DatabasePanelConfig
from modules.auto_trade.gui.services.database_service import DatabaseService



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

        ctk.CTkLabel(frame, text="📊 Database Stats", font=DatabasePanelConfig.TITLE_FONT).pack(
            anchor="w",
            padx=DatabasePanelConfig.PADX_MEDIUM,
            pady=(DatabasePanelConfig.PADX_MEDIUM, DatabasePanelConfig.PADY_SMALL),
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
        """Refresh statistics from database using DatabaseService."""
        try:
            # Get stats from service layer
            stats = DatabaseService.get_stats()

            # Update labels
            if stats and "total_orders" in self.stats_labels:
                self.stats_labels["total_orders"].configure(text=str(stats.get("total_orders", 0)))
                self.stats_labels["open_positions"].configure(text=str(stats.get("open_positions", 0)))
                self.stats_labels["total_signals"].configure(text=str(stats.get("total_signals", 0)))
                self.stats_labels["active_chains"].configure(text=str(stats.get("active_chains", 0)))
                self.stats_labels["audit_logs"].configure(text=str(stats.get("audit_logs", 0)))

            # Check last backup via service
            last_backup = DatabaseService.get_last_backup_time()
            if "last_backup" in self.stats_labels:
                if last_backup:
                    self.stats_labels["last_backup"].configure(text=last_backup)
                else:
                    self.stats_labels["last_backup"].configure(text="None")

        except Exception as e:
            log_error(f"Failed to refresh stats: {e}")
