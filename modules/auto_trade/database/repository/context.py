"""
Repository Context
==================

Bundles all repositories together for dependency injection.
Provides convenient access to all repository interfaces.

Now fully uses DynamoDB in all cases.

Created: 2026-02-20
"""

from typing import Optional

from .base import (
    AuditLogRepository,
    GradualRecoveryRepository,
    MartingaleRepository,
    OrderRepository,
    SignalRepository,
    SystemStateRepository,
)


_INSTANCE: Optional["RepositoryContext"] = None


class RepositoryContext:
    """
    Bundle of all repositories for dependency injection.

    Usage:
        ctx = RepositoryContext.from_env()
        orders = ctx.orders.get_open_positions()
    """

    def __init__(
        self,
        orders: OrderRepository,
        signals: SignalRepository,
        martingale: MartingaleRepository,
        gradual_recovery: GradualRecoveryRepository,
        system_state: SystemStateRepository,
        audit_log: AuditLogRepository,
    ):
        self.orders = orders
        self.signals = signals
        self.martingale = martingale
        self.gradual_recovery = gradual_recovery
        self.system_state = system_state
        self.audit_log = audit_log

    @classmethod
    def for_dynamodb(cls) -> "RepositoryContext":
        """
        Create context with DynamoDB repositories.

        Returns:
            RepositoryContext with DynamoDB implementations
        """
        from .dynamodb import (
            DynamoDBAuditLogRepository,
            DynamoDBGradualRecoveryRepository,
            DynamoDBMartingaleRepository,
            DynamoDBOrderRepository,
            DynamoDBSignalRepository,
            DynamoDBSystemStateRepository,
        )

        return cls(
            orders=DynamoDBOrderRepository(),
            signals=DynamoDBSignalRepository(),
            martingale=DynamoDBMartingaleRepository(),
            gradual_recovery=DynamoDBGradualRecoveryRepository(),
            system_state=DynamoDBSystemStateRepository(),
            audit_log=DynamoDBAuditLogRepository(),
        )

    @classmethod
    def from_env(cls, *args, **kwargs) -> "RepositoryContext":
        """
        Create context (ignores args now as we only use DynamoDB).

        Returns:
            RepositoryContext with DynamoDB implementations
        """
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = cls.for_dynamodb()
        return _INSTANCE

    @property
    def backend_type(self) -> str:
        """Return the current backend type."""
        return "dynamodb"


__all__ = ["RepositoryContext"]
