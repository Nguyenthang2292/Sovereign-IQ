"""
Database Configuration for Auto Trading System
===============================================

Centralized configuration for database paths, settings, and constants.

Created: 2026-02-03
"""

import os
from pathlib import Path
from typing import Optional

# ============================================================================
# DATABASE PATHS
# ============================================================================

# Default database directory
DEFAULT_DB_DIR = os.getenv("AUTO_TRADE_DB_DIR", "data")

# Default database file name
DEFAULT_DB_NAME = os.getenv("AUTO_TRADE_DB_NAME", "auto_trade.db")

# Full database path
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, DEFAULT_DB_NAME)

# Schema file path (relative to this file)
DEFAULT_SCHEMA_PATH = str((Path(__file__).resolve().parent / "schema.sql"))

# Backup directory
DEFAULT_BACKUP_DIR = os.path.join(DEFAULT_DB_DIR, "backups")

# Archive directory for old data
DEFAULT_ARCHIVE_DIR = os.path.join(DEFAULT_DB_DIR, "archive")

# Export directory for data exports
DEFAULT_EXPORT_DIR = os.path.join(DEFAULT_DB_DIR, "exports")

# ============================================================================
# CONNECTION SETTINGS
# ============================================================================

# Connection pool size
DB_POOL_SIZE = int(os.getenv("AUTO_TRADE_DB_POOL_SIZE", "5"))

# Maximum pool overflow
DB_MAX_OVERFLOW = int(os.getenv("AUTO_TRADE_DB_MAX_OVERFLOW", "10"))

# Connection timeout (seconds)
DB_TIMEOUT = int(os.getenv("AUTO_TRADE_DB_TIMEOUT", "30"))

# Echo SQL statements (for debugging)
DB_ECHO = os.getenv("AUTO_TRADE_DB_ECHO", "false").lower() == "true"

# ============================================================================
# SQLITE OPTIMIZATION SETTINGS
# ============================================================================

# Journal mode (WAL for better concurrency)
SQLITE_JOURNAL_MODE = "WAL"

# Synchronous mode (NORMAL for balance of safety and speed)
SQLITE_SYNCHRONOUS = "NORMAL"

# Cache size in KB (negative means KB, positive means pages)
SQLITE_CACHE_SIZE = -64000  # 64MB cache

# Temp store location
SQLITE_TEMP_STORE = "MEMORY"

# ============================================================================
# BACKUP SETTINGS
# ============================================================================

# Maximum number of backups to keep
MAX_BACKUPS = int(os.getenv("AUTO_TRADE_MAX_BACKUPS", "30"))

# Compress backups by default
BACKUP_COMPRESS = os.getenv("AUTO_TRADE_BACKUP_COMPRESS", "true").lower() == "true"

# Backup interval in hours
BACKUP_INTERVAL_HOURS = int(os.getenv("AUTO_TRADE_BACKUP_INTERVAL", "24"))

# Backup name prefix
BACKUP_PREFIX = "auto_trade_backup"

# ============================================================================
# DATA RETENTION SETTINGS
# ============================================================================

# Days to keep closed orders before archiving
ORDERS_RETENTION_DAYS = int(os.getenv("AUTO_TRADE_ORDERS_RETENTION", "90"))

# Days to keep audit logs
AUDIT_LOG_RETENTION_DAYS = int(os.getenv("AUTO_TRADE_AUDIT_RETENTION", "90"))

# Days to keep signal records
SIGNALS_RETENTION_DAYS = int(os.getenv("AUTO_TRADE_SIGNALS_RETENTION", "180"))

# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

# Allowed table names (for SQL injection prevention)
ALLOWED_TABLES = {
    "orders",
    "signals",
    "martingale_chain",
    "system_state",
    "audit_log",
}

# Valid order statuses
VALID_ORDER_STATUSES = {"PENDING", "OPEN", "CLOSED", "CANCELLED", "FAILED"}

# Valid order sides
VALID_ORDER_SIDES = {"LONG", "SHORT"}

# Valid order sources
VALID_ORDER_SOURCES = {"PROGRAMMATIC", "MANUAL", "EXTERNAL"}

# Valid execution modes
VALID_EXECUTION_MODES = {"AUTO", "MANUAL", "EXTERNAL"}

# Valid signal types
VALID_SIGNAL_TYPES = {"LONG", "SHORT", "NEUTRAL"}

# Valid signal qualities
VALID_SIGNAL_QUALITIES = {"HIGH", "MEDIUM", "LOW"}

# Valid signal outcomes
VALID_SIGNAL_OUTCOMES = {"WIN", "LOSS", "BREAKEVEN", "PENDING"}

# Valid martingale chain statuses
VALID_MARTINGALE_STATUSES = {"ACTIVE", "RECOVERED", "FAILED", "CANCELLED"}

# Valid audit log severities
VALID_AUDIT_SEVERITIES = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# ============================================================================
# LEVERAGE LIMITS
# ============================================================================

# Minimum leverage
MIN_LEVERAGE = 1

# Maximum leverage
MAX_LEVERAGE = 125

# ============================================================================
# QUERY DEFAULTS
# ============================================================================

# Default query limit
DEFAULT_QUERY_LIMIT = 100

# Maximum query limit
MAX_QUERY_LIMIT = 1000

# Default pagination offset
DEFAULT_PAGINATION_OFFSET = 0

# ============================================================================
# RETRY SETTINGS
# ============================================================================

# Maximum retry attempts for database operations
MAX_RETRY_ATTEMPTS = int(os.getenv("AUTO_TRADE_DB_RETRY_ATTEMPTS", "3"))

# Retry delay multiplier (exponential backoff)
RETRY_DELAY_MULTIPLIER = float(os.getenv("AUTO_TRADE_DB_RETRY_MULTIPLIER", "1.0"))

# Minimum retry delay in seconds
MIN_RETRY_DELAY = float(os.getenv("AUTO_TRADE_DB_MIN_RETRY_DELAY", "2.0"))

# Maximum retry delay in seconds
MAX_RETRY_DELAY = float(os.getenv("AUTO_TRADE_DB_MAX_RETRY_DELAY", "10.0"))

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

# Log SQL queries
LOG_SQL_QUERIES = os.getenv("AUTO_TRADE_LOG_SQL", "false").lower() == "true"

# Log database operations
LOG_DB_OPERATIONS = os.getenv("AUTO_TRADE_LOG_DB_OPS", "true").lower() == "true"

# Log performance metrics
LOG_PERFORMANCE = os.getenv("AUTO_TRADE_LOG_PERFORMANCE", "false").lower() == "true"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_db_path(custom_path: Optional[str] = None) -> str:
    """
    Get database path with optional override.

    Args:
        custom_path: Optional custom database path

    Returns:
        Database file path
    """
    return custom_path or DEFAULT_DB_PATH


def get_backup_dir(custom_dir: Optional[str] = None) -> str:
    """
    Get backup directory with optional override.

    Args:
        custom_dir: Optional custom backup directory

    Returns:
        Backup directory path
    """
    return custom_dir or DEFAULT_BACKUP_DIR


def validate_leverage(leverage: int) -> bool:
    """
    Validate leverage value.

    Args:
        leverage: Leverage value to validate

    Returns:
        True if valid

    Raises:
        ValueError: If leverage is out of range
    """
    if not isinstance(leverage, int):
        raise ValueError(f"Leverage must be an integer, got {type(leverage).__name__}")

    if leverage < MIN_LEVERAGE or leverage > MAX_LEVERAGE:
        raise ValueError(f"Leverage must be between {MIN_LEVERAGE} and {MAX_LEVERAGE}, got {leverage}")

    return True


def validate_order_status(status: str) -> bool:
    """
    Validate order status.

    Args:
        status: Status to validate

    Returns:
        True if valid

    Raises:
        ValueError: If status is invalid
    """
    if status not in VALID_ORDER_STATUSES:
        raise ValueError(f"Invalid order status: {status}. Must be one of {VALID_ORDER_STATUSES}")

    return True


def validate_table_name(table_name: str) -> bool:
    """
    Validate table name against whitelist (SQL injection prevention).

    Args:
        table_name: Table name to validate

    Returns:
        True if valid

    Raises:
        ValueError: If table name is not in whitelist
    """
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table_name}. Must be one of {ALLOWED_TABLES}")

    return True
