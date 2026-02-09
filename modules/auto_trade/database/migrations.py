"""
Database Migration System for Auto Trading System
===================================================

Handles database schema migrations and version management.
Uses Alembic-like approach for SQLite.

Created: 2026-02-03
"""

import hashlib
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MigrationManager:
    """
    Manages database schema migrations.
    """

    def __init__(self, db_path: str, schema_path: str) -> None:
        """
        Initialize migration manager.

        Args:
            db_path: Path to SQLite database file
            schema_path: Path to schema.sql file
        """
        self.db_path = db_path
        self.schema_path = schema_path
        self.migrations_dir = Path(__file__).parent / "migrations"
        self.migrations_dir.mkdir(exist_ok=True)

    def get_current_version(self) -> str:
        """
        Get current database schema version.

        Returns:
            Current version string (e.g., '1.0.0')
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM system_state WHERE key = 'schema_version'")
            result = cursor.fetchone()
            conn.close()

            return result[0] if result else "0.0.0"
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return "0.0.0"

    def set_version(self, version: str) -> None:
        """
        Set database schema version.

        Args:
            version: Version string to set
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO system_state (key, value, value_type, description, category)
            VALUES ('schema_version', ?, 'string', 'Database schema version', 'SYSTEM')
            """,
            (version,),
        )
        conn.commit()
        conn.close()

        logger.info(f"Database schema version updated to {version}")

    def initialize_database(self) -> bool:
        """
        Initialize database from schema.sql if database is empty.
        """
        if not os.path.exists(self.db_path):
            logger.info(f"Database does not exist. Creating: {self.db_path}")

            # Create parent directories if needed
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            # Read schema
            with open(self.schema_path, "r", encoding="utf-8") as f:
                schema_sql = f.read()

            # Execute schema
            conn = sqlite3.connect(self.db_path)
            conn.executescript(schema_sql)
            conn.commit()
            conn.close()

            logger.info(f"Database initialized successfully from {self.schema_path}")

            # Enable WAL mode for better concurrent access
            self.enable_wal_mode()

            return True

        return False

    def _ensure_migrations_applied_table(self) -> None:
        """
        Create migrations_applied table if it does not exist.
        Safe to call on every run; required when DB exists but was created from an older schema.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS migrations_applied (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                migration_name TEXT UNIQUE NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum TEXT
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_migrations_applied_name ON migrations_applied(migration_name)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_migrations_applied_at ON migrations_applied(applied_at)"
        )
        conn.commit()
        conn.close()

    def _column_exists(self, table: str, column: str) -> bool:
        """Return True if table has the given column (safe for migration idempotency)."""
        if table not in {"orders", "signals", "martingale_chain", "gradual_recovery", "system_state", "audit_log", "migrations_applied"}:
            return False
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table})")
            rows = cursor.fetchall()
            conn.close()
            # row: (cid, name, type, notnull, default_value, pk)
            return any(r[1] == column for r in rows)
        except Exception:
            return False

    def enable_wal_mode(self) -> bool:
        """
        Enable Write-Ahead Logging (WAL) mode for better concurrent access.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            result = cursor.fetchone()
            conn.close()

            if result and result[0] == "wal":
                logger.info("WAL mode enabled for database")
                return True
            else:
                logger.warning("Failed to enable WAL mode")
                return False
        except Exception as e:
            logger.error(f"Error enabling WAL mode: {e}")
            return False

    def check_integrity(self) -> bool:
        """
        Check database integrity.

        Returns:
            True if integrity check passes
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            conn.close()

            if result and result[0] == "ok":
                logger.info("Database integrity check passed")
                return True
            else:
                logger.error(f"Database integrity check failed: {result}")
                return False
        except Exception as e:
            logger.error(f"Error checking database integrity: {e}")
            return False

    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get table schema information.

        Args:
            table_name: Name of table

        Returns:
            List of column information dictionaries
        """
        # Whitelist of allowed tables to prevent SQL injection
        ALLOWED_TABLES = {
            "orders",
            "signals",
            "martingale_chain",
            "gradual_recovery",
            "system_state",
            "audit_log",
            "migrations_applied",
        }

        if table_name not in ALLOWED_TABLES:
            logger.error(f"Invalid table name: {table_name}")
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Safe to use f-string after validation
        cursor.execute(f"PRAGMA table_info({table_name})")

        columns = []
        for row in cursor.fetchall():
            columns.append(
                {
                    "cid": row[0],
                    "name": row[1],
                    "type": row[2],
                    "notnull": bool(row[3]),
                    "default_value": row[4],
                    "primary_key": bool(row[5]),
                }
            )

        conn.close()
        return columns

    def get_all_tables(self) -> List[str]:
        """
        Get list of all tables in database.

        Returns:
            List of table names
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        return tables

    def apply_migration(self, migration_sql: str, migration_name: str) -> bool:
        """
        Apply a migration script.

        Args:
            migration_sql: SQL script to execute
            migration_name: Name of migration

        Returns:
            True if successful
        """
        try:
            logger.info(f"Applying migration: {migration_name}")
            self._ensure_migrations_applied_table()

            # Idempotency: 003 adds orders.trailing_step_index; schema.sql may already have it
            if migration_name == "003_add_trailing_step_index.sql" and self._column_exists("orders", "trailing_step_index"):
                logger.info(f"Migration {migration_name}: column already exists, marking as applied")
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                checksum = hashlib.sha256(migration_sql.encode()).hexdigest()
                cursor.execute(
                    """
                    INSERT INTO migrations_applied (migration_name, applied_at, checksum)
                    VALUES (?, datetime('now'), ?)
                    ON CONFLICT(migration_name) DO UPDATE SET
                        applied_at = excluded.applied_at,
                        checksum = excluded.checksum
                    """,
                    (migration_name, checksum),
                )
                conn.commit()
                conn.close()
                return True

            conn = sqlite3.connect(self.db_path)
            conn.executescript(migration_sql)
            conn.commit()

            # Record migration in migrations_applied table
            cursor = conn.cursor()
            checksum = hashlib.sha256(migration_sql.encode()).hexdigest()
            cursor.execute(
                """
                INSERT INTO migrations_applied (migration_name, applied_at, checksum)
                VALUES (?, datetime('now'), ?)
                ON CONFLICT(migration_name) DO UPDATE SET
                    applied_at = excluded.applied_at,
                    checksum = excluded.checksum
                """,
                (migration_name, checksum),
            )
            conn.commit()
            conn.close()

            logger.info(f"Migration {migration_name} applied successfully")
            return True

        except Exception as e:
            logger.error(f"Error applying migration {migration_name}: {e}")
            return False

    def create_migration_template(self, migration_name: str) -> str:
        """
        Create a new migration file template.

        Args:
            migration_name: Name of migration

        Returns:
            Path to created migration file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{migration_name}.sql"
        filepath = self.migrations_dir / filename

        template = f"""-- Migration: {migration_name}
-- Created: {datetime.now().isoformat()}
-- Description: [Add description here]

-- UPGRADE
-- Add your schema changes here

-- Example: Add new column
-- ALTER TABLE orders ADD COLUMN new_column TEXT;

-- Example: Create index
-- CREATE INDEX idx_new_column ON orders(new_column);

-- DOWNGRADE (optional)
-- Add rollback script here if needed
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(template)

        logger.info(f"Created migration template: {filepath}")
        return str(filepath)

    def get_pending_migrations(self) -> List[str]:
        """
        Get list of pending migrations that haven't been applied.

        Returns:
            List of migration filenames
        """
        # Get all migration files
        migration_files = sorted(self.migrations_dir.glob("*.sql"))
        all_migrations = [f.name for f in migration_files]

        # Get already applied migrations from database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT migration_name FROM migrations_applied")
            applied = {row[0] for row in cursor.fetchall()}
            conn.close()
        except sqlite3.OperationalError:
            # Table doesn't exist yet (first run)
            applied = set()

        # Return only pending migrations
        pending = [m for m in all_migrations if m not in applied]
        return pending

    def auto_migrate(self) -> bool:
        """
        Automatically apply all pending migrations.

        Returns:
            True if all migrations applied successfully
        """
        self._ensure_migrations_applied_table()
        pending = self.get_pending_migrations()

        if not pending:
            logger.info("No pending migrations")
            return True

        logger.info(f"Found {len(pending)} pending migrations")

        for migration_file in pending:
            migration_path = self.migrations_dir / migration_file

            with open(migration_path, "r", encoding="utf-8") as f:
                migration_sql = f.read()

            if not self.apply_migration(migration_sql, migration_file):
                logger.error(f"Failed to apply migration: {migration_file}")
                return False

        logger.info("All migrations applied successfully")
        return True

    def vacuum_database(self) -> bool:
        """
        Vacuum database to reclaim space and optimize.
        """
        try:
            logger.info("Starting database VACUUM operation")
            conn = sqlite3.connect(self.db_path)
            conn.execute("VACUUM")
            conn.close()
            logger.info("Database VACUUM completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during VACUUM: {e}")
            return False

    def analyze_database(self) -> bool:
        """
        Update query planner statistics for better performance.
        """
        try:
            logger.info("Analyzing database statistics")
            conn = sqlite3.connect(self.db_path)
            conn.execute("ANALYZE")
            conn.close()
            logger.info("Database ANALYZE completed")
            return True
        except Exception as e:
            logger.error(f"Error during ANALYZE: {e}")
            return False

    def get_database_size(self) -> int:
        """
        Get database file size in bytes.

        Returns:
            Database size in bytes
        """
        if os.path.exists(self.db_path):
            return os.path.getsize(self.db_path)
        return 0

    def get_table_row_counts(self) -> Dict[str, int]:
        """
        Get row counts for all tables.

        Returns:
            Dictionary mapping table names to row counts
        """
        # Whitelist of allowed tables to prevent SQL injection
        ALLOWED_TABLES = {
            "orders",
            "signals",
            "martingale_chain",
            "gradual_recovery",
            "system_state",
            "audit_log",
            "migrations_applied",
        }

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        tables = self.get_all_tables()
        counts = {}

        for table in tables:
            # Validate table name against whitelist
            if table not in ALLOWED_TABLES:
                logger.warning(f"Skipping unknown table: {table}")
                continue

            # Safe to use f-string now after validation
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]

        conn.close()
        return counts


# ============================================================================
# PRE-DEFINED MIGRATIONS
# ============================================================================


class CommonMigrations:
    """
    Common migration scripts that can be reused.
    """

    @staticmethod
    def add_column(table: str, column: str, column_type: str, default=None) -> str:
        """
        Generate SQL to add a column.

        Args:
            table: Table name
            column: Column name
            column_type: Column type (e.g., 'TEXT', 'INTEGER')
            default: Optional default value

        Returns:
            SQL string
        """
        sql = f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
        if default is not None:
            if isinstance(default, str):
                sql += f" DEFAULT '{default}'"
            else:
                sql += f" DEFAULT {default}"
        sql += ";"
        return sql

    @staticmethod
    def create_index(table: str, column: str, index_name: Optional[str] = None) -> str:
        """
        Generate SQL to create an index.

        Args:
            table: Table name
            column: Column name
            index_name: Optional custom index name

        Returns:
            SQL string
        """
        if not index_name:
            index_name = f"idx_{table}_{column}"

        return f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({column});"

    @staticmethod
    def drop_index(index_name: str) -> str:
        """
        Generate SQL to drop an index.

        Args:
            index_name: Index name

        Returns:
            SQL string
        """
        return f"DROP INDEX IF EXISTS {index_name};"


# ============================================================================
# MIGRATION HELPERS
# ============================================================================


def initialize_database_if_needed(db_path: str, schema_path: str, auto_migrate: bool = True) -> bool:
    """
    Initialize database if it doesn't exist, apply migrations if needed.

    Args:
        db_path: Path to database file
        schema_path: Path to schema.sql
        auto_migrate: Whether to auto-apply pending migrations

    Returns:
        True if successful
    """
    manager = MigrationManager(db_path, schema_path)

    # Initialize if needed
    if manager.initialize_database():
        logger.info("Database initialized from schema")
    else:
        logger.info("Database already exists")

    # Check integrity
    if not manager.check_integrity():
        logger.error("Database integrity check failed!")
        return False

    # Apply migrations
    if auto_migrate:
        if not manager.auto_migrate():
            logger.error("Auto-migration failed!")
            return False

    # Optimize
    manager.analyze_database()

    logger.info("Database ready")
    return True


def get_migration_manager(db_path: str, schema_path: str) -> MigrationManager:
    """
    Get a migration manager instance.

    Args:
        db_path: Path to database file
        schema_path: Path to schema.sql

    Returns:
        MigrationManager instance
    """
    return MigrationManager(db_path, schema_path)
