"""Compatibility shim for legacy SQLite backup imports."""

from modules.auto_trade.archives.sqlite_legacy.database.backup import BackupManager

__all__ = ["BackupManager"]
