"""Database Panel Configuration Constants.

This module contains all configuration constants used by the Database Panel
and its associated sections to avoid magic numbers and enable easy customization.
"""

from typing import Tuple

from modules.auto_trade.gui.utils.fonts import Fonts


class DatabasePanelConfig:
    """Configuration constants for Database Panel components.

    Centralizes pagination, reconciliation, cleanup, fonts, layout weights,
    and data viewer settings to avoid magic numbers. Used by DatabasePanel
    and all database sections (OrdersSection, DataViewerSection, ActionsSection, etc.).
    """

    # Database defaults
    DEFAULT_DB_NAME: str = "crypto_trading.db"

    # Pagination settings
    DEFAULT_PAGE_SIZE: int = 20
    INITIAL_PAGE: int = 1

    # Reconciliation settings
    DEFAULT_RECONCILE_HOURS: int = 24
    MAX_RECONCILE_ERRORS_SHOWN: int = 5

    # Cleanup settings
    DEFAULT_DAYS_TO_KEEP: int = 90

    # Stats refresh interval (milliseconds)
    STATS_REFRESH_INTERVAL_MS: int = 30000  # 30 seconds

    # Font configurations
    TEXTBOX_FONT: Tuple[str, int] = Fonts.INPUT
    TITLE_FONT: Tuple[str, int, str] = Fonts.H2
    HEADER_FONT: Tuple[str, int, str] = Fonts.H1

    # Layout weights for grid configuration
    LEFT_PANEL_WEIGHT: int = 3  # 60%
    RIGHT_PANEL_WEIGHT: int = 2  # 40%

    # Padding constants
    PADX_SMALL: int = 5
    PADX_MEDIUM: int = 10
    PADX_LARGE: int = 20
    PADY_SMALL: int = 5
    PADY_MEDIUM: int = 10

    # Data viewer settings
    DATA_VIEWER_HEIGHT: int = 200
    MAX_DISPLAY_COLUMNS: int = 5
    COLUMN_WIDTH: int = 15

    # Table names
    TABLE_ORDERS: str = "Orders"
    TABLE_SIGNALS: str = "Signals"
    TABLE_MARTINGALE_CHAINS: str = "Martingale Chains"
    TABLE_AUDIT_LOG: str = "Audit Log"
    AVAILABLE_TABLES: Tuple[str, ...] = (TABLE_ORDERS, TABLE_SIGNALS, TABLE_MARTINGALE_CHAINS, TABLE_AUDIT_LOG)
