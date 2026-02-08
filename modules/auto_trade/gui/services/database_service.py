"""Database Service Layer.

Extracts database operations from UI components to provide a clean service layer.
All database operations should go through this service rather than being called
directly from UI components.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

from modules.auto_trade.database import (
    session_scope,
    get_open_positions,
    create_database_backup,
    get_migration_manager,
    get_recent_audit_logs,
    reconcile_orders_with_binance,
    get_db_manager,
)
from modules.auto_trade.database.models import Order, Signal, MartingaleChain, AuditLog
from modules.auto_trade.database.config import DEFAULT_DB_PATH, DEFAULT_SCHEMA_PATH
from modules.auto_trade.gui.config.database_panel_config import DatabasePanelConfig

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for database operations."""

    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with session_scope() as session:
                return {
                    "total_orders": session.query(Order).count(),
                    "open_positions": session.query(Order).filter(Order.status == "OPEN").count(),
                    "total_signals": session.query(Signal).count(),
                    "active_chains": session.query(MartingaleChain).filter(MartingaleChain.status == "ACTIVE").count(),
                    "audit_logs": session.query(AuditLog).count(),
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    @staticmethod
    def get_last_backup_time() -> Optional[str]:
        """Get last backup timestamp as formatted string."""
        try:
            from modules.auto_trade.database.config import DEFAULT_BACKUP_DIR

            backup_dir = Path(DEFAULT_BACKUP_DIR)
            if backup_dir.exists():
                backups = sorted(list(backup_dir.glob("*.db")), key=lambda f: f.stat().st_mtime, reverse=True)
                if backups:
                    from datetime import datetime

                    return datetime.fromtimestamp(backups[0].stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            return None
        except Exception:
            return None

    @staticmethod
    def create_backup() -> Optional[str]:
        """Create database backup. Returns backup path or None."""
        try:
            return create_database_backup()
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return None

    @staticmethod
    def run_migrations() -> Tuple[bool, str]:
        """Run database migrations. Returns (success, message)."""
        try:
            manager = get_migration_manager(DEFAULT_DB_PATH, DEFAULT_SCHEMA_PATH)
            if manager:
                return (True, "Migration manager ready")
            return (False, "Migration manager not available")
        except Exception as e:
            return (False, str(e))

    @staticmethod
    def cleanup_old_records(days_to_keep: int = None) -> Tuple[bool, str]:
        """Cleanup old records. Returns (success, message)."""
        if days_to_keep is None:
            days_to_keep = DatabasePanelConfig.DEFAULT_DAYS_TO_KEEP

        try:
            from modules.auto_trade.database.utils import DatabaseCleaner

            with session_scope() as session:
                deleted_orders = DatabaseCleaner.cleanup_old_records(session, Order, days_to_keep=days_to_keep)
                deleted_signals = DatabaseCleaner.cleanup_old_records(session, Signal, days_to_keep=days_to_keep)
                deleted_logs = DatabaseCleaner.cleanup_old_records(
                    session, AuditLog, days_to_keep=days_to_keep, date_column="timestamp"
                )

                msg = f"Deleted: {deleted_orders} orders, {deleted_signals} signals, {deleted_logs} logs"
                return (True, msg)
        except Exception as e:
            return (False, str(e))

    @staticmethod
    def check_integrity() -> Tuple[bool, str]:
        """Check database integrity. Returns (is_ok, status)."""
        try:
            from sqlalchemy import text

            manager = get_db_manager()
            with manager.engine.connect() as conn:
                result = conn.execute(text("PRAGMA integrity_check")).fetchone()
                status = result[0] if result else "Unknown"
                return (status == "ok", status)
        except Exception as e:
            return (False, str(e))


class ReconciliationService:
    """Service for Binance reconciliation operations."""

    @staticmethod
    def reconcile_with_binance(
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        symbols: Optional[List[str]] = None,
        since_hours: int = None,
    ) -> Dict[str, Any]:
        """Reconcile orders with Binance."""
        if since_hours is None:
            since_hours = DatabasePanelConfig.DEFAULT_RECONCILE_HOURS

        try:
            return reconcile_orders_with_binance(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                symbols=symbols,
                since_hours=since_hours,
            )
        except Exception as e:
            logger.error(f"Reconcile failed: {e}")
            return {"inserted": 0, "skipped": 0, "closed_stale": 0, "errors": [str(e)]}


class DataViewerService:
    """Service for data viewer operations."""

    @staticmethod
    def get_table_count(table_name: str) -> int:
        """Get total count for a table."""
        try:
            with session_scope() as session:
                if table_name == "Orders":
                    return session.query(Order).count()
                elif table_name == "Signals":
                    return session.query(Signal).count()
                elif table_name == "Martingale Chains":
                    return session.query(MartingaleChain).count()
                elif table_name == "Audit Log":
                    return session.query(AuditLog).count()
                return 0
        except Exception as e:
            logger.error(f"Failed to get count: {e}")
            return 0

    @staticmethod
    def get_table_data(table_name: str, limit: int = None, last_id: Optional[int] = None) -> List[Any]:
        """Get paginated data from a table."""
        from modules.auto_trade.database import (
            get_audit_log_cursor,
            get_martingale_chains_cursor,
            get_orders_cursor,
            get_signals_cursor,
        )

        if limit is None:
            limit = DatabasePanelConfig.DEFAULT_PAGE_SIZE

        try:
            with session_scope() as session:
                if table_name == "Orders":
                    return get_orders_cursor(session, last_id=last_id, limit=limit)
                elif table_name == "Signals":
                    return get_signals_cursor(session, last_id=last_id, limit=limit)
                elif table_name == "Martingale Chains":
                    return get_martingale_chains_cursor(session, last_id=last_id, limit=limit)
                elif table_name == "Audit Log":
                    return get_audit_log_cursor(session, last_id=last_id, limit=limit)
                return []
        except Exception as e:
            logger.error(f"Failed to get data: {e}")
            return []

    @staticmethod
    def get_audit_logs(limit: int = 100) -> List[Any]:
        """Get recent audit logs."""
        try:
            with session_scope() as session:
                return get_recent_audit_logs(session, limit=limit)
        except Exception as e:
            logger.error(f"Failed to get audit logs: {e}")
            return []
