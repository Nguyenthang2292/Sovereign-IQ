"""
Auto Trading Database Module
=============================

Database layer for the auto trading system.

Provides:
- Repository interaction layer (DynamoDB backends only)
- RepositoryContext for backend-agnostic operations
- Query functions for convenient access

Usage:
    # New code - use RepositoryContext
    from modules.auto_trade.database import RepositoryContext

    ctx = RepositoryContext.from_env()
    orders = ctx.orders.get_open_positions()

    # Legacy compatible - query functions
    from modules.auto_trade.database import get_open_positions

    orders = get_open_positions()  # No session needed

Created: 2026-02-03
Modified: 2026-02-20
"""

# Import configuration
from .config import DB_BACKEND

# Import query functions for backward compatibility
from .queries import (
    cancel_gradual_recovery,
    create_audit_log,
    create_gradual_recovery,
    create_order,
    find_or_create_martingale_chain,
    get_active_gradual_recovery,
    get_active_martingale_chains,
    get_all_gradual_recoveries,
    get_all_programmatic_orders,
    get_audit_log_cursor,
    get_daily_stats,
    get_gradual_recovery_by_id,
    get_last_closed_order,
    get_martingale_chains_cursor,
    get_martingale_state,
    get_open_positions,
    get_order_by_client_id,
    get_order_by_id,
    get_orders_by_symbol,
    get_orders_cursor,
    get_overall_stats,
    get_recent_audit_logs,
    get_recent_signals,
    get_signal_performance_stats,
    get_signals_cursor,
    get_system_state,
    is_programmatic_order,
    mark_be_moved,
    mark_signal_executed,
    save_signal,
    set_system_state,
    update_gradual_recovery,
    update_martingale_chain,
    update_order_status,
    update_order_status_by_client_id,
    update_signal_outcome,
)

# Import repository abstracts and context
from .repository import (
    RepositoryContext,
    get_audit_log_repository,
    get_gradual_recovery_repository,
    get_martingale_repository,
    get_order_repository,
    get_signal_repository,
    get_system_state_repository,
)

# ---------------------------------------------------------------------------
# Legacy SQLite compatibility helpers
# ---------------------------------------------------------------------------


def initialize_database(db_path: str = ":memory:", schema_path: str | None = None) -> bool:
    """Initialize legacy SQLite DB for backward-compatible test tooling."""
    from .config import DEFAULT_SCHEMA_PATH
    from .migrations import MigrationManager

    manager = MigrationManager(db_path, schema_path or DEFAULT_SCHEMA_PATH)
    manager.initialize_database()
    manager.auto_migrate()
    return True


def session_scope(db_path: str = ":memory:"):
    """Provide a legacy SQLAlchemy session context manager for old tests."""
    from .utils import DatabaseManager

    manager = DatabaseManager(db_path, echo=False)
    return manager.session_scope()

# ---------------------------------------------------------------------------
# Binance → DynamoDB reconciliation
# ---------------------------------------------------------------------------


def reconcile_orders_with_binance(
    api_key: str,
    api_secret: str,
    testnet: bool = False,
    symbols=None,
    since_hours: int = 24,
) -> dict:
    """
    Reconcile open Binance positions into DynamoDB.

    Fetches all positions that are currently open on Binance and inserts any
    that are missing from the local DynamoDB table.  This is called:
    - Once at auto-trade startup
    - Periodically every hour while auto-trade is running
    - Manually via the "Reconcile with Binance" button in the DB panel

    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        testnet: True to use testnet
        symbols: Optional list of symbols to filter (unused; all open positions are checked)
        since_hours: Unused kept for API compat with legacy SQLite version

    Returns:
        Dict with keys: inserted (int), skipped (int), errors (list[str])
    """
    from modules.auto_trade.execution.binance_client import BinanceClient
    from modules.auto_trade.gui.utils.position_sync_service import PositionSyncService

    client = BinanceClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        dry_run=False,
    )
    stats = PositionSyncService.sync_all_positions(client)
    return {
        "inserted": stats.get("synced", 0),
        "skipped": stats.get("existing", 0),
        "closed": stats.get("closed", 0),
        "errors": [f"failed={stats.get('failed', 0)}"] if stats.get("failed") else [],
    }


# Module version
__version__ = "1.2.1"

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Version
    "__version__",
    # Repository Layer
    "RepositoryContext",
    "DB_BACKEND",
    "get_order_repository",
    "get_signal_repository",
    "get_martingale_repository",
    "get_gradual_recovery_repository",
    "get_system_state_repository",
    "get_audit_log_repository",
    # Query functions (backward compatible)
    "get_open_positions",
    "get_last_closed_order",
    "get_all_programmatic_orders",
    "get_orders_cursor",
    "is_programmatic_order",
    "get_order_by_id",
    "get_order_by_client_id",
    "update_order_status",
    "update_order_status_by_client_id",
    "mark_be_moved",
    "create_order",
    "get_orders_by_symbol",
    "get_martingale_state",
    "find_or_create_martingale_chain",
    "update_martingale_chain",
    "get_active_martingale_chains",
    "get_martingale_chains_cursor",
    "save_signal",
    "mark_signal_executed",
    "update_signal_outcome",
    "get_recent_signals",
    "get_signal_performance_stats",
    "get_signals_cursor",
    "get_system_state",
    "set_system_state",
    "create_audit_log",
    "get_recent_audit_logs",
    "get_audit_log_cursor",
    "get_daily_stats",
    "get_overall_stats",
    "get_active_gradual_recovery",
    "create_gradual_recovery",
    "update_gradual_recovery",
    "cancel_gradual_recovery",
    "get_gradual_recovery_by_id",
    "get_all_gradual_recoveries",
    # Binance → DB reconciliation
    "reconcile_orders_with_binance",
    # Legacy SQLite compatibility
    "initialize_database",
    "session_scope",
]
