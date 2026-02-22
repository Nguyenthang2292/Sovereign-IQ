"""
Repository Module
=================

Abstract repository interfaces and implementations for database operations.

This module provides:
- Abstract base classes (OrderRepository, SignalRepository, etc.)
- DynamoDB implementations
- Factory for backend selection
- RepositoryContext for dependency injection

Usage:
    from modules.auto_trade.database.repository import (
        RepositoryContext,
        OrderRepository,
        get_order_repository,
    )

    # Using context (recommended)
    ctx = RepositoryContext.from_env()
    orders = ctx.orders.get_open_positions()

Created: 2026-02-20
"""

from .base import (
    AuditLogRepository,
    GradualRecoveryRepository,
    MartingaleRepository,
    OrderRepository,
    PaginatedResult,
    SignalRepository,
    SystemStateRepository,
)
from .context import RepositoryContext
from .factory import (
    DB_BACKEND,
    get_audit_log_repository,
    get_gradual_recovery_repository,
    get_martingale_repository,
    get_order_repository,
    get_signal_repository,
    get_system_state_repository,
)

__all__ = [
    "PaginatedResult",
    "OrderRepository",
    "SignalRepository",
    "MartingaleRepository",
    "GradualRecoveryRepository",
    "SystemStateRepository",
    "AuditLogRepository",
    "RepositoryContext",
    "DB_BACKEND",
    "get_order_repository",
    "get_signal_repository",
    "get_martingale_repository",
    "get_gradual_recovery_repository",
    "get_system_state_repository",
    "get_audit_log_repository",
]
