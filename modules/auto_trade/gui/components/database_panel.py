"""Database Panel Component - Refactored as container."""

import customtkinter as ctk
from typing import Optional, Dict, Any
import logging

from modules.auto_trade.database import DatabaseManager
from .database import (
    OrdersSection,
    SignalsSection,
    MartingaleSection,
    RecoverySection,
    DataViewerSection,
    StatsSection,
    LogsSection,
    ActionsSection,
)

logger = logging.getLogger(__name__)


class DatabasePanel(ctk.CTkFrame):
    """Database panel container composing multiple section components."""

    def __init__(self, parent, settings_manager):
        super().__init__(parent)
        self.settings_manager = settings_manager

        # Initialize database
        self.db_manager = self._init_database()

        # Create layout
        self._create_layout()

        # Load initial stats
        self._load_initial_stats()

    def _init_database(self):
        """Initialize database connection."""
        try:
            db_path = "crypto_trading.db"
            if hasattr(self.settings_manager, "get_setting"):
                path_setting = self.settings_manager.get_setting("database.path")
                if path_setting:
                    db_path = path_setting
            return DatabaseManager
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return None

    def _create_layout(self):
        """Create the main layout structure."""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=3)  # Left panel (60%)
        self.grid_columnconfigure(1, weight=2)  # Right panel (40%)

        # Left panel (scrollable)
        self.left_panel = ctk.CTkScrollableFrame(self)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Right panel (fixed)
        self.right_panel = ctk.CTkFrame(self)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # Create sections
        self._create_sections()

    def _create_sections(self):
        """Create and compose all section components."""
        # Data viewer first (needed by other sections)
        self.data_viewer_section = DataViewerSection(self.left_panel, self._log)

        # Left panel sections
        self.orders_section = OrdersSection(self.left_panel, self._log, self._refresh_stats)
        self.signals_section = SignalsSection(self.left_panel, self._log, self._refresh_stats)
        self.martingale_section = MartingaleSection(self.left_panel, self._log)
        self.recovery_section = RecoverySection(self.left_panel, self._log, self.data_viewer_section.data_viewer)

        # Right panel sections
        self.stats_section = StatsSection(self.right_panel)
        self.actions_section = ActionsSection(
            self.right_panel,
            self._log,
            self._refresh_stats,
            self.data_viewer_section.get_current_table,
            self.settings_manager,
        )
        self.logs_section = LogsSection(self.right_panel)

    def _log(self, message: str, level: str = "INFO"):
        """Log message to activity logs."""
        self.logs_section.log(message, level)

    def _refresh_stats(self):
        """Refresh statistics display."""
        self.stats_section.refresh()
        self.data_viewer_section.refresh()

    def _load_initial_stats(self):
        """Load initial statistics on startup."""
        self._refresh_stats()

    def copy_selection_to_clipboard(self):
        """Copy selected text from the Data Viewer to clipboard. No-op if no selection."""
        try:
            tv = self.data_viewer_section.data_viewer
            try:
                sel = tv.get("sel.first", "sel.last")
            except Exception:
                return
            if not sel or not sel.strip():
                return
            root = self.winfo_toplevel()
            root.clipboard_clear()
            root.clipboard_append(sel.strip())
            if logger:
                logger.debug("Copied selection to clipboard")
        except Exception as e:
            if logger:
                logger.debug("Copy selection failed: %s", e)
