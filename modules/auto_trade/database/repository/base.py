"""
Repository Base Interfaces
==========================

Abstract base classes defining the repository interfaces for all database operations.
These interfaces allow access to DynamoDB (and theoretically other) backends.

Created: 2026-02-20
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict


class PaginatedResult(TypedDict, total=False):
    """Paginated query result with cursor support."""

    items: List[Dict[str, Any]]
    last_key: Optional[Any]
    has_more: bool
    total_count: Optional[int]


class OrderRepository(ABC):
    """
    Abstract repository for Order operations.

    All methods return dicts (not ORM objects) for backend-agnostic access.
    """

    @abstractmethod
    def create_order(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new order.

        Args:
            data: Order data dictionary (must include order_id, symbol, side, entry_price, amount)

        Returns:
            Created order as dict

        Raises:
            ValueError: If required fields are missing or invalid
        """
        pass

    @abstractmethod
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open positions (programmatic orders only).

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders as dicts
        """
        pass

    @abstractmethod
    def get_order_by_id(self, order_id: str, verify_programmatic: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get order by order_id.

        Args:
            order_id: Order ID
            verify_programmatic: If True, only return programmatic orders

        Returns:
            Order dict or None
        """
        pass

    @abstractmethod
    def get_order_by_client_id(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order by client_order_id.

        Args:
            client_order_id: Client order ID

        Returns:
            Order dict or None
        """
        pass

    @abstractmethod
    def update_order_status(
        self, order_id: str, status: str, pnl: Optional[float] = None, verify_programmatic: bool = True
    ) -> bool:
        """
        Update order status.

        Args:
            order_id: Order ID
            status: New status
            pnl: Optional P&L value
            verify_programmatic: If True, only update programmatic orders

        Returns:
            True if updated, False otherwise
        """
        pass

    @abstractmethod
    def update_order_status_by_client_id(
        self, client_order_id: str, status: str, closed_at: Optional[datetime] = None, pnl: Optional[float] = None
    ) -> bool:
        """
        Update order status by client_order_id.

        Args:
            client_order_id: Client order ID
            status: New status
            closed_at: Optional close timestamp
            pnl: Optional P&L value

        Returns:
            True if updated, False otherwise
        """
        pass

    @abstractmethod
    def update(
        self,
        order_id: str,
        updates: Dict[str, Any],
        verify_programmatic: bool = True,
    ) -> bool:
        """
        Generic update for order fields.

        Args:
            order_id: Order ID
            updates: Dictionary of fields to update
            verify_programmatic: If True, only update programmatic orders

        Returns:
            True if updated, False otherwise
        """
        pass

    @abstractmethod
    def mark_be_moved(
        self,
        order_id: str,
        new_stop_loss: Optional[float] = None,
        new_take_profit: Optional[float] = None,
        verify_programmatic: bool = True,
    ) -> bool:
        """
        Mark break-even triggered for an order.

        Args:
            order_id: Order ID
            new_stop_loss: New stop loss price
            new_take_profit: New take profit price
            verify_programmatic: If True, only update programmatic orders

        Returns:
            True if updated, False otherwise
        """
        pass

    @abstractmethod
    def get_all_programmatic_orders(
        self, status: Optional[str] = None, symbol: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all programmatic orders with pagination.

        Args:
            status: Optional status filter
            symbol: Optional symbol filter
            limit: Maximum results
            offset: Number to skip

        Returns:
            List of order dicts
        """
        pass

    @abstractmethod
    def get_orders_cursor(
        self, last_id: Optional[int] = None, limit: int = 50, status: Optional[str] = None, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get orders using cursor-based pagination.

        Args:
            last_id: Last order ID from previous page
            limit: Maximum results
            status: Optional status filter
            symbol: Optional symbol filter

        Returns:
            List of order dicts
        """
        pass

    @abstractmethod
    def get_last_closed_order(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get most recent closed order.

        Args:
            symbol: Optional symbol filter

        Returns:
            Order dict or None
        """
        pass

    @abstractmethod
    def get_orders_by_symbol(
        self, symbol: str, status: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get orders for a specific symbol.

        Args:
            symbol: Trading symbol
            status: Optional status filter
            limit: Maximum results
            offset: Number to skip

        Returns:
            List of order dicts
        """
        pass

    @abstractmethod
    def is_programmatic_order(self, order_id: str) -> bool:
        """
        Check if order is programmatic.

        Args:
            order_id: Order ID

        Returns:
            True if programmatic, False otherwise
        """
        pass


class SignalRepository(ABC):
    """Abstract repository for Signal operations."""

    @abstractmethod
    def save_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save a new signal.

        Args:
            data: Signal data (must include correlation_id, symbol, signal_type, confidence)

        Returns:
            Created signal as dict
        """
        pass

    @abstractmethod
    def get_recent_signals(
        self, limit: int = 50, symbol: Optional[str] = None, executed_only: bool = False, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get recent signals.

        Args:
            limit: Maximum results
            symbol: Optional symbol filter
            executed_only: If True, only return executed signals
            offset: Number to skip

        Returns:
            List of signal dicts
        """
        pass

    @abstractmethod
    def mark_signal_executed(self, correlation_id: str, order_id: str) -> bool:
        """
        Mark signal as executed.

        Args:
            correlation_id: Signal correlation ID
            order_id: Executed order ID

        Returns:
            True if updated, False otherwise
        """
        pass

    @abstractmethod
    def update_signal_outcome(
        self,
        correlation_id: str,
        outcome: str,
        outcome_pnl: Optional[float] = None,
        outcome_duration_minutes: Optional[int] = None,
    ) -> bool:
        """
        Update signal outcome.

        Args:
            correlation_id: Signal correlation ID
            outcome: 'WIN', 'LOSS', or 'BREAKEVEN'
            outcome_pnl: Optional final P&L
            outcome_duration_minutes: Duration until outcome

        Returns:
            True if updated, False otherwise
        """
        pass

    @abstractmethod
    def get_signal_performance_stats(self, symbol: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        Get signal performance statistics.

        Args:
            symbol: Optional symbol filter
            days: Number of days to analyze

        Returns:
            Dict with performance metrics
        """
        pass

    @abstractmethod
    def get_signals_cursor(
        self,
        last_id: Optional[int] = None,
        limit: int = 50,
        symbol: Optional[str] = None,
        executed: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get signals using cursor-based pagination.

        Args:
            last_id: Last signal ID from previous page
            limit: Maximum results
            symbol: Optional symbol filter
            executed: Optional executed filter

        Returns:
            List of signal dicts
        """
        pass


class MartingaleRepository(ABC):
    """Abstract repository for Martingale chain operations."""

    @abstractmethod
    def find_or_create_martingale_chain(
        self,
        symbol: str,
        initial_order_id: str,
        loss: float,
        chain_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Find existing or create new Martingale chain.

        Args:
            symbol: Trading symbol
            initial_order_id: First order in chain
            loss: Initial loss amount
            chain_id: Optional unique chain identifier (for backward compatibility)

        Returns:
            Martingale chain dict
        """
        pass

    @abstractmethod
    def get_martingale_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get active Martingale chain for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Active chain dict or None
        """
        pass

    @abstractmethod
    def update_martingale_chain(self, chain_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update Martingale chain.

        Args:
            chain_id: Chain ID
            updates: Fields to update (current_step, latest_order_id, total_loss, etc.)

        Returns:
            True if updated, False otherwise
        """
        pass

    @abstractmethod
    def get_active_martingale_chains(self) -> List[Dict[str, Any]]:
        """
        Get all active Martingale chains.

        Returns:
            List of active chain dicts
        """
        pass

    @abstractmethod
    def get_martingale_chains_cursor(self, last_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get Martingale chains using cursor-based pagination.

        Args:
            last_id: Last chain ID from previous page
            limit: Maximum results

        Returns:
            List of chain dicts
        """
        pass


class GradualRecoveryRepository(ABC):
    """Abstract repository for Gradual Recovery operations."""

    @abstractmethod
    def create_gradual_recovery(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a new gradual recovery.

        Args:
            data: Recovery payload dict (supports recovery_id, initial_loss, config, symbol)

        Returns:
            Created recovery dict
        """
        pass

    @abstractmethod
    def get_active_gradual_recovery(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get active gradual recovery.

        Args:
            symbol: Optional symbol filter (None for global)

        Returns:
            Active recovery dict or None
        """
        pass

    @abstractmethod
    def update_gradual_recovery(self, recovery_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update gradual recovery.

        Args:
            recovery_id: Recovery ID
            updates: Fields to update

        Returns:
            True if updated, False otherwise
        """
        pass

    @abstractmethod
    def cancel_gradual_recovery(self, recovery_id: str) -> bool:
        """
        Cancel a gradual recovery.

        Args:
            recovery_id: Recovery ID

        Returns:
            True if cancelled, False otherwise
        """
        pass

    @abstractmethod
    def get_gradual_recovery_by_id(self, recovery_id: str) -> Optional[Dict[str, Any]]:
        """
        Get recovery by ID.

        Args:
            recovery_id: Recovery ID

        Returns:
            Recovery dict or None
        """
        pass

    @abstractmethod
    def get_all_gradual_recoveries(
        self, status: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all gradual recoveries.

        Args:
            status: Optional status filter
            limit: Maximum results
            offset: Number to skip

        Returns:
            List of recovery dicts
        """
        pass


class SystemStateRepository(ABC):
    """Abstract repository for system state operations."""

    @abstractmethod
    def get_system_state(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Get system state value by key.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            State value with correct type or default
        """
        pass

    @abstractmethod
    def set_system_state(
        self,
        key: str,
        value: Any,
        value_type: str = "string",
        description: Optional[str] = None,
        category: Optional[str] = None,
    ) -> bool:
        """
        Set system state value.

        Args:
            key: State key
            value: State value
            value_type: Type of value ('string', 'integer', 'float', 'boolean', 'json')
            description: Optional description
            category: Optional category

        Returns:
            True if set successfully
        """
        pass


class AuditLogRepository(ABC):
    """Abstract repository for audit log operations."""

    @abstractmethod
    def create_audit_log(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create audit log entry.

        Args:
            data: Audit log payload dict

        Returns:
            Created audit log dict
        """
        pass

    @abstractmethod
    def get_recent_audit_logs(
        self, limit: int = 100, severity: Optional[str] = None, event_type: Optional[str] = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get recent audit logs.

        Args:
            limit: Maximum results
            severity: Optional severity filter
            event_type: Optional event type filter
            offset: Number to skip

        Returns:
            List of audit log dicts
        """
        pass

    @abstractmethod
    def get_audit_log_cursor(
        self,
        last_id: Optional[int] = None,
        limit: int = 50,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get audit logs using cursor-based pagination.

        Args:
            last_id: Last log ID from previous page
            limit: Maximum results
            event_type: Optional event type filter
            severity: Optional severity filter

        Returns:
            List of audit log dicts
        """
        pass


__all__ = [
    "PaginatedResult",
    "OrderRepository",
    "SignalRepository",
    "MartingaleRepository",
    "GradualRecoveryRepository",
    "SystemStateRepository",
    "AuditLogRepository",
]
