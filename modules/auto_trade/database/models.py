"""Compatibility shim for legacy SQLite ORM model imports.

This module preserves historical import paths used by older tests/tools:
    from modules.auto_trade.database.models import Order
"""

from modules.auto_trade.archives.sqlite_legacy.database.models import (
    AuditLog,
    Base,
    GradualRecovery,
    MartingaleChain,
    Order,
    Signal,
    SystemState,
)

__all__ = [
    "Base",
    "Order",
    "Signal",
    "MartingaleChain",
    "GradualRecovery",
    "SystemState",
    "AuditLog",
]
