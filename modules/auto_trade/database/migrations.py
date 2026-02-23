"""Compatibility shim for legacy SQLite migration imports.

This module preserves historical import paths used by older tests/tools:
    from modules.auto_trade.database.migrations import MigrationManager
"""

from modules.auto_trade.archives.sqlite_legacy.database.migrations import MigrationManager

__all__ = ["MigrationManager"]
