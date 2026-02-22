"""
Repository Factory
==================

Factory functions for creating repository instances for DynamoDB.
DynamoDB is the sole database backend currently supported.

Created: 2026-02-20
"""

from .base import (
    AuditLogRepository,
    GradualRecoveryRepository,
    MartingaleRepository,
    OrderRepository,
    SignalRepository,
    SystemStateRepository,
)
from .dynamodb import (
    DynamoDBAuditLogRepository,
    DynamoDBGradualRecoveryRepository,
    DynamoDBMartingaleRepository,
    DynamoDBOrderRepository,
    DynamoDBSignalRepository,
    DynamoDBSystemStateRepository,
)

# Keep DB_BACKEND for backward compatibility if any module expects it,
# but hardcode it to dynamodb
DB_BACKEND = "dynamodb"


def get_order_repository() -> OrderRepository:
    """Get OrderRepository instance."""
    return DynamoDBOrderRepository()


def get_signal_repository() -> SignalRepository:
    """Get SignalRepository instance."""
    return DynamoDBSignalRepository()


def get_martingale_repository() -> MartingaleRepository:
    """Get MartingaleRepository instance."""
    return DynamoDBMartingaleRepository()


def get_gradual_recovery_repository() -> GradualRecoveryRepository:
    """Get GradualRecoveryRepository instance."""
    return DynamoDBGradualRecoveryRepository()


def get_system_state_repository() -> SystemStateRepository:
    """Get SystemStateRepository instance."""
    return DynamoDBSystemStateRepository()


def get_audit_log_repository() -> AuditLogRepository:
    """Get AuditLogRepository instance."""
    return DynamoDBAuditLogRepository()


__all__ = [
    "DB_BACKEND",
    "get_order_repository",
    "get_signal_repository",
    "get_martingale_repository",
    "get_gradual_recovery_repository",
    "get_system_state_repository",
    "get_audit_log_repository",
]
