"""
Database Utilities for Auto Trading System
===========================================

Helper functions for database operations, connection management,
transactions, and data export.

Created: 2026-02-03
"""

import csv
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Row
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Base

logger = logging.getLogger(__name__)


# ============================================================================
# DATABASE ENGINE AND SESSION MANAGEMENT
# ============================================================================


class DatabaseManager:
    """
    Manages database engine, sessions, and connections.
    """

    def __init__(self, db_path: str, echo: bool = False, pool_size: int = 5, max_overflow: int = 10):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
            echo: Whether to echo SQL statements (for debugging)
            pool_size: Connection pool size
            max_overflow: Maximum pool overflow
        """
        self.db_path = db_path
        self.echo = echo

        # Create engine
        if db_path == ":memory:":
            # In-memory database (for testing)
            self.engine = create_engine(
                "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool, echo=echo
            )
        else:
            # File-based database
            db_url = f"sqlite:///{db_path}"
            self.engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False, "timeout": 30},
                echo=echo,
                pool_size=pool_size,
                max_overflow=max_overflow,
            )

        # Configure SQLite for optimal performance
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            # Enable WAL mode for better concurrent access
            cursor.execute("PRAGMA journal_mode=WAL")
            # Synchronous mode for durability
            cursor.execute("PRAGMA synchronous=NORMAL")
            # Foreign key constraints
            cursor.execute("PRAGMA foreign_keys=ON")
            # Cache size (negative means KB, positive means pages)
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
            # Temp store in memory
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.close()

        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        logger.info(f"Database manager initialized: {db_path}")

    def create_all_tables(self):
        """Create all tables from models."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("All database tables created")

    def drop_all_tables(self):
        """Drop all tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All database tables dropped")

    def get_session(self) -> Session:
        """
        Get a new database session.

        Returns:
            SQLAlchemy Session
        """
        return self.SessionLocal()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope for database operations.

        Usage:
            with db_manager.session_scope() as session:
                # Do database operations
                session.add(order)

        Yields:
            SQLAlchemy Session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def execute_raw_sql(self, sql: str, params: tuple = ()) -> Sequence[Row[Any]]:
        """
        Execute raw SQL query.

        Args:
            sql: SQL query string
            params: Query parameters

        Returns:
            Sequence of query result rows
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params)
            return result.fetchall()

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        stats: Dict[str, Any] = {}

        with self.session_scope() as session:
            # Get table counts
            from .models import AuditLog, MartingaleChain, Order, Signal

            stats["total_orders"] = session.query(Order).filter(Order.order_source == "PROGRAMMATIC").count()
            stats["open_orders"] = (
                session.query(Order).filter(Order.order_source == "PROGRAMMATIC", Order.status == "OPEN").count()
            )
            stats["closed_orders"] = (
                session.query(Order).filter(Order.order_source == "PROGRAMMATIC", Order.status == "CLOSED").count()
            )
            stats["total_signals"] = session.query(Signal).count()
            stats["executed_signals"] = session.query(Signal).filter(Signal.executed.is_(True)).count()
            stats["active_martingale_chains"] = (
                session.query(MartingaleChain).filter(MartingaleChain.status == "ACTIVE").count()
            )
            stats["audit_log_entries"] = session.query(AuditLog).count()

        # Get database file size
        if self.db_path != ":memory:":
            import os

            if os.path.exists(self.db_path):
                size_bytes = os.path.getsize(self.db_path)
                stats["database_size_mb"] = float(round(size_bytes / (1024 * 1024), 2))

        return stats

    def check_connection(self) -> bool:
        """
        Check if database connection is working.

        Returns:
            True if connection is healthy
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False

    def optimize_database(self):
        """Run database optimization commands."""
        with self.engine.connect() as conn:
            # Analyze tables for query optimizer
            conn.execute(text("ANALYZE"))
            conn.commit()
            logger.info("Database ANALYZE completed")

            # Vacuum (reclaim space) - only if not in WAL mode
            try:
                conn.execute(text("VACUUM"))
                conn.commit()
                logger.info("Database VACUUM completed")
            except Exception as e:
                logger.warning(f"VACUUM failed (may be in WAL mode): {e}")


# ============================================================================
# TRANSACTION HELPERS
# ============================================================================


@contextmanager
def transaction(session: Session):
    """
    Context manager for database transactions.

    Usage:
        with transaction(session):
            # Do multiple operations
            session.add(order1)
            session.add(order2)

    Args:
        session: SQLAlchemy session

    Yields:
        Same session
    """
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Transaction rolled back: {e}")
        raise


def safe_commit(session: Session) -> bool:
    """
    Safely commit session with error handling.

    Args:
        session: SQLAlchemy session

    Returns:
        True if committed successfully
    """
    try:
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Commit failed: {e}")
        return False


# ============================================================================
# DATA EXPORT UTILITIES
# ============================================================================


class DataExporter:
    """
    Export database data to various formats.
    """

    @staticmethod
    def export_to_csv(session: Session, model_class: Any, output_path: str, filters: Optional[Dict] = None) -> bool:
        """
        Export table data to CSV.

        Args:
            session: Database session
            model_class: SQLAlchemy model class
            output_path: Path to output CSV file
            filters: Optional filters dictionary

        Returns:
            True if successful
        """
        try:
            # Build query
            query = session.query(model_class)

            if filters:
                for key, value in filters.items():
                    if hasattr(model_class, key):
                        query = query.filter(getattr(model_class, key) == value)

            # Get all records
            records = query.all()

            if not records:
                logger.warning("No records to export")
                return False

            # Get column names
            columns = [c.name for c in model_class.__table__.columns]

            # Write to CSV
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

                for record in records:
                    row = {col: getattr(record, col) for col in columns}
                    writer.writerow(row)

            logger.info(f"Exported {len(records)} records to {output_path}")
            return True

        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return False

    @staticmethod
    def export_to_json(
        session: Session, model_class: Any, output_path: str, filters: Optional[Dict] = None, pretty: bool = True
    ) -> bool:
        """
        Export table data to JSON.

        Args:
            session: Database session
            model_class: SQLAlchemy model class
            output_path: Path to output JSON file
            filters: Optional filters dictionary
            pretty: Whether to pretty-print JSON

        Returns:
            True if successful
        """
        try:
            # Build query
            query = session.query(model_class)

            if filters:
                for key, value in filters.items():
                    if hasattr(model_class, key):
                        query = query.filter(getattr(model_class, key) == value)

            # Get all records
            records = query.all()

            if not records:
                logger.warning("No records to export")
                return False

            # Convert to dict
            data = []
            for record in records:
                if hasattr(record, "to_dict"):
                    data.append(record.to_dict())
                else:
                    # Manual conversion
                    row_dict = {}
                    for col in model_class.__table__.columns:
                        value = getattr(record, col.name)
                        # Convert datetime to string
                        if isinstance(value, datetime):
                            value = value.isoformat()
                        row_dict[col.name] = value
                    data.append(row_dict)

            # Write to JSON
            with open(output_path, "w", encoding="utf-8") as f:
                if pretty:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(data, f, ensure_ascii=False)

            logger.info(f"Exported {len(records)} records to {output_path}")
            return True

        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return False


# ============================================================================
# DATABASE CLEANUP UTILITIES
# ============================================================================


class DatabaseCleaner:
    """
    Utilities for cleaning up old database records.
    """

    @staticmethod
    def cleanup_old_records(
        session: Session, model_class: Any, days_to_keep: int = 90, date_column: str = "created_at"
    ) -> int:
        """
        Delete records older than specified days.

        Args:
            session: Database session
            model_class: SQLAlchemy model class
            days_to_keep: Number of days to keep
            date_column: Name of date column to filter on

        Returns:
            Number of records deleted
        """
        try:
            from datetime import timedelta

            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            # Build delete query
            date_col = getattr(model_class, date_column)
            deleted = session.query(model_class).filter(date_col < cutoff_date).delete()

            session.commit()

            logger.info(f"Deleted {deleted} old records from {model_class.__tablename__}")
            return deleted

        except Exception as e:
            session.rollback()
            logger.error(f"Cleanup failed: {e}")
            return 0

    @staticmethod
    def archive_old_orders(session: Session, days_to_keep: int = 90, archive_path: str = "data/archive") -> int:
        """
        Archive old closed orders to JSON before deletion.

        Args:
            session: Database session
            days_to_keep: Number of days to keep in main DB
            archive_path: Directory for archive files

        Returns:
            Number of orders archived
        """
        try:
            from datetime import timedelta

            from .models import Order

            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            # Get old closed orders
            old_orders = session.query(Order).filter(Order.status == "CLOSED", Order.closed_at < cutoff_date).all()

            if not old_orders:
                return 0

            # Create archive directory
            archive_dir = Path(archive_path)
            archive_dir.mkdir(parents=True, exist_ok=True)

            # Archive to JSON
            archive_file = archive_dir / f"orders_archive_{datetime.now().strftime('%Y%m%d')}.json"

            archived_data = [order.to_dict() for order in old_orders]

            with open(archive_file, "w", encoding="utf-8") as f:
                json.dump(archived_data, f, indent=2, ensure_ascii=False)

            # Delete from database
            for order in old_orders:
                session.delete(order)

            session.commit()

            logger.info(f"Archived {len(old_orders)} orders to {archive_file}")
            return len(old_orders)

        except Exception as e:
            session.rollback()
            logger.error(f"Archive failed: {e}")
            return 0


# ============================================================================
# DATABASE RESET UTILITIES (For Testing)
# ============================================================================


def reset_database_for_testing(db_manager: DatabaseManager):
    """
    Reset database for testing (drops all data).

    WARNING: This deletes all data!

    Args:
        db_manager: DatabaseManager instance
    """
    logger.warning("Resetting database - all data will be deleted!")

    db_manager.drop_all_tables()
    db_manager.create_all_tables()

    logger.info("Database reset completed")


def seed_test_data(session: Session):
    """
    Seed database with test data.

    Args:
        session: Database session
    """
    from .models import Order, Signal, SystemState

    # Add test system state
    test_state = SystemState(
        key="test_mode", value="true", value_type="boolean", description="Test mode flag", category="TESTING"
    )
    session.add(test_state)

    # Add test signal
    test_signal = Signal(
        correlation_id="TEST_SIGNAL_001",
        symbol="BTCUSDT",
        signal_type="LONG",
        confidence=0.85,
        atc_score=0.7,
        xgboost_score=0.9,
        gemini_score=0.85,
    )
    session.add(test_signal)

    # Add test order
    test_order = Order(
        order_id="TEST_ORDER_001",
        client_order_id="AT_TEST_001",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000.0,
        amount=0.01,
        leverage=2,
        stop_loss=45000.0,
        take_profit=52500.0,
        status="CLOSED",
        pnl=50.0,
        order_source="PROGRAMMATIC",
        execution_mode="AUTO",
    )
    session.add(test_order)

    session.commit()
    logger.info("Test data seeded successfully")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def get_database_manager(db_path: str = "data/auto_trade.db", echo: bool = False) -> DatabaseManager:
    """
    Get a database manager instance.

    Args:
        db_path: Path to database file
        echo: Whether to echo SQL

    Returns:
        DatabaseManager instance
    """
    return DatabaseManager(db_path, echo=echo)


def export_all_data(session: Session, output_dir: str = "data/exports") -> bool:
    """
    Export all tables to JSON files.

    Args:
        session: Database session
        output_dir: Output directory

    Returns:
        True if successful
    """
    from .models import AuditLog, MartingaleChain, Order, Signal

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exporter = DataExporter()

    tables = {"orders": Order, "signals": Signal, "martingale_chains": MartingaleChain, "audit_logs": AuditLog}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for name, model_class in tables.items():
        filename = f"{name}_{timestamp}.json"
        filepath = output_path / filename
        exporter.export_to_json(session, model_class, str(filepath))

    logger.info(f"All data exported to {output_dir}")
    return True
