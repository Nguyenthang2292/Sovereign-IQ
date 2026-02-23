"""
Database Configuration for Auto Trading System
===============================================

Centralized configuration for database paths, settings, and constants.
Currently using DynamoDB exclusively.

Created: 2026-02-03
Modified: 2026-02-20
"""

import os
from pathlib import Path
from typing import Optional

# ============================================================================
# DATABASE PATHS (Kept for exports/archives if needed)
# ============================================================================

# Default configuration directory
DEFAULT_DATA_DIR = os.getenv("AUTO_TRADE_DATA_DIR", "data")

# Archive directory for old data
DEFAULT_ARCHIVE_DIR = os.path.join(DEFAULT_DATA_DIR, "archive")

# Export directory for data exports
DEFAULT_EXPORT_DIR = os.path.join(DEFAULT_DATA_DIR, "exports")

# Legacy SQLite schema path retained for backwards compatibility in tests/tools
DEFAULT_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "archives",
    "sqlite_legacy",
    "database",
    "schema.sql",
)

# ============================================================================
# BACKEND SELECTION
# ============================================================================

# Database backend
DB_BACKEND = "dynamodb"

# DynamoDB Settings
DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "AutoTrade")
DYNAMODB_ENDPOINT_URL = os.getenv("DYNAMODB_ENDPOINT_URL")  # None = AWS production
DYNAMODB_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

# Valid backends
VALID_DB_BACKENDS = {"dynamodb"}

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

# Allowed table names (models logical names)
ALLOWED_TABLES = {
    "orders",
    "signals",
    "martingale_chain",
    "system_state",
    "audit_log",
    "gradual_recovery",
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

# Log database operations
LOG_DB_OPERATIONS = os.getenv("AUTO_TRADE_LOG_DB_OPS", "true").lower() == "true"

# Log performance metrics
LOG_PERFORMANCE = os.getenv("AUTO_TRADE_LOG_PERFORMANCE", "false").lower() == "true"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def validate_leverage(leverage: int) -> bool:
    """Validate leverage value."""
    if not isinstance(leverage, int):
        raise ValueError(f"Leverage must be an integer, got {type(leverage).__name__}")
    if leverage < MIN_LEVERAGE or leverage > MAX_LEVERAGE:
        raise ValueError(f"Leverage must be between {MIN_LEVERAGE} and {MAX_LEVERAGE}, got {leverage}")
    return True


def validate_order_status(status: str) -> bool:
    """Validate order status."""
    if status not in VALID_ORDER_STATUSES:
        raise ValueError(f"Invalid order status: {status}. Must be one of {VALID_ORDER_STATUSES}")
    return True


def validate_table_name(table_name: str) -> bool:
    """Validate table name (entity name) against whitelist."""
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table_name}. Must be one of {ALLOWED_TABLES}")
    return True
