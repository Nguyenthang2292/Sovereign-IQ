"""
Auto Trading Database Module
==============================

Database layer for the auto trading system.

Provides:
- SQLAlchemy ORM models (Order, Signal, MartingaleChain, etc.)
- Query functions with programmatic order filtering
- Migration system
- Backup and recovery
- Database utilities and helpers

Usage:
    from modules.auto_trade.database import (
        get_db_manager,
        get_session,
        create_order,
        get_open_positions,
        save_signal
    )

    # Initialize database
    db_manager = get_db_manager()

    # Get session
    with db_manager.session_scope() as session:
        orders = get_open_positions(session)

Created: 2026-02-03
"""

import threading
from pathlib import Path

# Import backup
from .backup import BackupManager, BackupScheduler, create_backup, list_all_backups, restore_latest_backup

# Import configuration
from .config import DEFAULT_DB_PATH, DEFAULT_SCHEMA_PATH

# Import migrations
from .migrations import CommonMigrations, MigrationManager, get_migration_manager, initialize_database_if_needed

# Import models
from .models import AuditLog, Base, GradualRecovery, MartingaleChain, Order, Signal, SystemState

# Import query functions
from .queries import (
    # Audit log queries
    create_audit_log,
    create_order,
    find_or_create_martingale_chain,
    get_active_martingale_chains,
    get_all_programmatic_orders,
    # Statistics queries
    get_daily_stats,
    get_last_closed_order,
    # Martingale queries
    get_martingale_state,
    # Order queries
    get_open_positions,
    get_order_by_client_id,
    get_order_by_id,
    get_orders_by_symbol,
    get_overall_stats,
    get_recent_audit_logs,
    get_recent_signals,
    get_signal_performance_stats,
    # System state queries
    get_system_state,
    is_programmatic_order,
    mark_be_moved,
    mark_signal_executed,
    # Signal queries
    save_signal,
    set_system_state,
    update_martingale_chain,
    update_order_status,
    update_signal_outcome,
)

# Import reconcile
from .reconcile import reconcile_orders_with_binance

# Import utilities
from .utils import (
    DatabaseCleaner,
    DatabaseManager,
    DataExporter,
    export_all_data,
    get_database_manager,
    reset_database_for_testing,
    safe_commit,
    seed_test_data,
    transaction,
)

# Module version
__version__ = "1.0.0"

# Note: DEFAULT_DB_PATH and DEFAULT_SCHEMA_PATH are now imported from config.py
# They can be overridden via environment variables:
# - AUTO_TRADE_DB_DIR: Directory for database files (default: "data")
# - AUTO_TRADE_DB_NAME: Database filename (default: "auto_trade.db")


# ============================================================================
# GLOBAL DATABASE MANAGER (Singleton)
# ============================================================================

_db_manager_instance = None
_db_manager_lock = threading.Lock()


def get_db_manager(
    db_path: str = DEFAULT_DB_PATH, echo: bool = False, initialize: bool = True, pool_size: int = 5
) -> DatabaseManager:
    """
    Get global database manager instance (singleton with thread-safety).

    Args:
        db_path: Path to database file
        echo: Whether to echo SQL statements
        initialize: Whether to initialize database if needed
        pool_size: Connection pool size

    Returns:
        DatabaseManager instance
    """
    global _db_manager_instance

    # Fast path: instance already exists
    if _db_manager_instance is not None:
        return _db_manager_instance

    # Slow path: acquire lock and create instance
    with _db_manager_lock:
        # Double-check pattern to prevent race conditions
        if _db_manager_instance is None:
            _db_manager_instance = DatabaseManager(db_path, echo=echo, pool_size=pool_size)

            if initialize:
                # Initialize database if needed
                initialize_database_if_needed(db_path=db_path, schema_path=DEFAULT_SCHEMA_PATH, auto_migrate=True)

    return _db_manager_instance


def get_session():
    """
    Get a new database session from global manager.

    Returns:
        SQLAlchemy Session
    """
    manager = get_db_manager()
    return manager.get_session()


def session_scope():
    """
    Get session scope context manager from global manager.

    Usage:
        with session_scope() as session:
            orders = get_open_positions(session)

    Yields:
        SQLAlchemy Session
    """
    manager = get_db_manager()
    return manager.session_scope()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def initialize_database(db_path: str = DEFAULT_DB_PATH, force_recreate: bool = False):
    """
    Initialize database with schema.

    Args:
        db_path: Path to database file
        force_recreate: If True, drop and recreate all tables
    """
    if force_recreate:
        manager = get_db_manager(db_path, initialize=False)
        manager.drop_all_tables()
        manager.create_all_tables()
    else:
        initialize_database_if_needed(db_path=db_path, schema_path=DEFAULT_SCHEMA_PATH, auto_migrate=True)


def create_database_backup(compress: bool = True) -> str:
    """
    Create a backup of the database.

    Args:
        compress: Whether to compress backup

    Returns:
        Path to backup file
    """
    return create_backup(db_path=DEFAULT_DB_PATH, backup_dir="data/backups", compress=compress)


def get_database_stats() -> dict:
    """
    Get comprehensive database statistics.

    Returns:
        Dictionary with statistics
    """
    manager = get_db_manager()

    # Get basic stats
    stats = manager.get_database_stats()

    # Add backup stats
    backup_manager = BackupManager(DEFAULT_DB_PATH)
    stats["backup_stats"] = backup_manager.get_backup_stats()

    # Add performance stats
    with session_scope() as session:
        stats["signal_performance"] = get_signal_performance_stats(session, days=30)
        stats["overall_trading"] = get_overall_stats(session)

    return stats


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Version
    "__version__",
    # Models
    "Base",
    "Order",
    "Signal",
    "MartingaleChain",
    "GradualRecovery",
    "SystemState",
    "AuditLog",
    # Order Queries
    "get_open_positions",
    "get_last_closed_order",
    "get_all_programmatic_orders",
    "is_programmatic_order",
    "get_order_by_id",
    "get_order_by_client_id",
    "update_order_status",
    "mark_be_moved",
    "create_order",
    "get_orders_by_symbol",
    # Martingale Queries
    "get_martingale_state",
    "find_or_create_martingale_chain",
    "update_martingale_chain",
    "get_active_martingale_chains",
    # Signal Queries
    "save_signal",
    "mark_signal_executed",
    "update_signal_outcome",
    "get_recent_signals",
    "get_signal_performance_stats",
    # System State
    "get_system_state",
    "set_system_state",
    # Audit Log
    "create_audit_log",
    "get_recent_audit_logs",
    # Statistics
    "get_daily_stats",
    "get_overall_stats",
    # Database Management
    "DatabaseManager",
    "get_db_manager",
    "reconcile_orders_with_binance",
    "get_session",
    "session_scope",
    "initialize_database",
    # Transactions
    "transaction",
    "safe_commit",
    # Utilities
    "DataExporter",
    "DatabaseCleaner",
    "export_all_data",
    "get_database_stats",
    # Migrations
    "MigrationManager",
    "CommonMigrations",
    "initialize_database_if_needed",
    "get_migration_manager",
    # Backup
    "BackupManager",
    "BackupScheduler",
    "create_backup",
    "create_database_backup",
    "restore_latest_backup",
    "list_all_backups",
    # Testing
    "reset_database_for_testing",
    "seed_test_data",
]
