"""Database Panel Component - Refactored as container."""

import customtkinter as ctk

from modules.auto_trade.gui.config.database_panel_config import DatabasePanelConfig
from modules.common.ui.logging import log_debug

from .database import (
    ActionsSection,
    DataViewerSection,
    LogsSection,
    MartingaleSection,
    OrdersSection,
    RecoverySection,
    SignalsSection,
    StatsSection,
)


class DatabasePanel(ctk.CTkFrame):
    """Database panel container composing multiple section components.

    Composes OrdersSection, SignalsSection, MartingaleSection, RecoverySection,
    DataViewerSection, StatsSection, LogsSection, and ActionsSection. Layout
    and constants are driven by DatabasePanelConfig.
    """

    def __init__(self, parent, settings_manager):
        """Initialize the database panel.

        Args:
            parent: Parent widget (e.g. tab or frame).
            settings_manager: Object with optional get_setting(key) for database.path.
        """
        super().__init__(parent)
        self.settings_manager = settings_manager

        # Create layout
        self._create_layout()

        # Load initial stats asynchronously after startup settles.
        # Running synchronous DynamoDB queries too early can block initial window display.
        self.after(3000, self._load_initial_stats)

    def _create_layout(self):
        """Create the main layout structure (left/right panels and sections)."""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=DatabasePanelConfig.LEFT_PANEL_WEIGHT)  # Left panel (60%)
        self.grid_columnconfigure(1, weight=DatabasePanelConfig.RIGHT_PANEL_WEIGHT)  # Right panel (40%)

        # Left panel (scrollable)
        self.left_panel = ctk.CTkScrollableFrame(self)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, DatabasePanelConfig.PADX_SMALL))

        # Right panel (fixed)
        self.right_panel = ctk.CTkFrame(self)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(DatabasePanelConfig.PADX_SMALL, 0))

        # Create sections
        self._create_sections()

    def _create_sections(self):
        """Create and compose all section components (orders, signals, martingale, recovery, stats, actions, logs)."""
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
        """Log message to activity logs.

        Args:
            message: Text to log.
            level: Log level (e.g. INFO, ERROR). Defaults to INFO.
        """
        self.logs_section.log(message, level)

    def _refresh_stats(self):
        """Refresh statistics display and data viewer."""
        self.stats_section.refresh()
        self.data_viewer_section.refresh()

    def _load_initial_stats(self):
        """Load initial statistics on startup (delegates to _refresh_stats)."""
        self._refresh_stats()

    def copy_selection_to_clipboard(self):
        """Copy selected text from the Data Viewer to clipboard.

        No-op if no selection or on error. Uses toplevel clipboard.
        """
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
            log_debug("Copied selection to clipboard")
        except Exception as e:
            log_debug("Copy selection failed: %s", e)
