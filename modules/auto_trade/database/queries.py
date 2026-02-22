"""
Database Query Layer for Auto Trading System
=============================================

This module provides query functions that delegate to DynamoDB via RepositoryContext.

CRITICAL: All order queries ONLY return PROGRAMMATIC orders by default.
Manual trades from Binance are excluded from queries.

Usage (new code):
    from modules.auto_trade.database import RepositoryContext

    ctx = RepositoryContext.from_env()
    orders = ctx.orders.get_open_positions()

Usage (legacy compatible):
    from modules.auto_trade.database import get_open_positions

    # No session needed - uses DynamoDB internally
    orders = get_open_positions()

Created: 2026-02-03
Refactored: 2026-02-20 (DynamoDB only)
"""

from typing import Any, Dict, List, Optional

from .repository import RepositoryContext

_ctx: Optional[RepositoryContext] = None


def _get_ctx() -> RepositoryContext:
    """Get or create RepositoryContext (singleton)."""
    global _ctx
    if _ctx is None:
        _ctx = RepositoryContext.from_env()
    return _ctx


def get_open_positions(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get all open positions (programmatic orders only).

    Args:
        symbol: Optional symbol filter

    Returns:
        List of open orders as dicts
    """
    ctx = _get_ctx()
    return ctx.orders.get_open_positions(symbol=symbol)


def get_last_closed_order(symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get the most recent closed order (programmatic only).

    Args:
        symbol: Optional symbol filter

    Returns:
        Most recent closed order dict or None
    """
    ctx = _get_ctx()
    return ctx.orders.get_last_closed_order(symbol=symbol)


def get_all_programmatic_orders(
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Fetch all programmatic orders.

    Args:
        status: Optional status filter ('OPEN', 'CLOSED', etc.)
        symbol: Optional symbol filter
        limit: Maximum number of orders to return
        offset: Number of orders to skip (for pagination)

    Returns:
        List of programmatic order dicts
    """
    ctx = _get_ctx()
    return ctx.orders.get_all_programmatic_orders(
        status=status,
        symbol=symbol,
        limit=limit,
        offset=offset,
    )


def get_orders_cursor(
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 50,
    last_key: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Get paginated orders.

    Args:
        status: Optional status filter
        symbol: Optional symbol filter
        limit: Page size
        last_key: Pagination cursor (not used in DynamoDB impl, kept for API compat)

    Returns:
        List of order dicts
    """
    ctx = _get_ctx()
    return ctx.orders.get_orders_cursor(
        status=status,
        symbol=symbol,
        limit=limit,
    )


def is_programmatic_order(order_id: str) -> bool:
    """
    Check if an order is programmatic.

    Args:
        order_id: Order ID

    Returns:
        True if order is programmatic
    """
    ctx = _get_ctx()
    order = ctx.orders.get_order_by_id(order_id, verify_programmatic=False)
    if order:
        return order.get("order_source") == "PROGRAMMATIC"
    return False


def get_order_by_id(order_id: str) -> Optional[Dict[str, Any]]:
    """
    Get order by order_id (programmatic only by default).

    Args:
        order_id: Order ID

    Returns:
        Order dict or None
    """
    ctx = _get_ctx()
    return ctx.orders.get_order_by_id(order_id, verify_programmatic=True)


def get_order_by_client_id(client_order_id: str) -> Optional[Dict[str, Any]]:
    """
    Get order by client_order_id.

    Args:
        client_order_id: Client order ID

    Returns:
        Order dict or None
    """
    ctx = _get_ctx()
    return ctx.orders.get_order_by_client_id(client_order_id)


def update_order_status(
    order_id: str,
    status: str,
    pnl: Optional[float] = None,
) -> bool:
    """
    Update order status.

    Args:
        order_id: Order ID
        status: New status
        pnl: Optional P&L value

    Returns:
        True if updated successfully
    """
    ctx = _get_ctx()
    return ctx.orders.update_order_status(order_id, status, pnl=pnl)


def update_order_status_by_client_id(
    client_order_id: str,
    status: str,
    pnl: Optional[float] = None,
) -> bool:
    """
    Update order status by client_order_id.

    Args:
        client_order_id: Client order ID
        status: New status
        pnl: Optional P&L value

    Returns:
        True if updated successfully
    """
    ctx = _get_ctx()
    return ctx.orders.update_order_status_by_client_id(client_order_id, status, pnl=pnl)


def mark_be_moved(
    order_id: str,
    new_stop_loss: Optional[float] = None,
    new_take_profit: Optional[float] = None,
    verify_programmatic: bool = True,
) -> bool:
    """
    Mark order as moved to breakeven.

    Args:
        order_id: Order ID
        new_stop_loss: New stop loss price
        new_take_profit: New take profit price
        verify_programmatic: If True, only update programmatic orders

    Returns:
        True if marked successfully
    """
    ctx = _get_ctx()
    return ctx.orders.mark_be_moved(
        order_id,
        new_stop_loss=new_stop_loss,
        new_take_profit=new_take_profit,
        verify_programmatic=verify_programmatic,
    )


def create_order(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new order.

    Args:
        data: Order data dictionary

    Returns:
        Created order dict
    """
    ctx = _get_ctx()
    return ctx.orders.create_order(data)


def get_orders_by_symbol(symbol: str, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get orders by symbol.

    Args:
        symbol: Trading symbol
        status: Optional status filter
        limit: Maximum number of results

    Returns:
        List of order dicts
    """
    ctx = _get_ctx()
    return ctx.orders.get_orders_by_symbol(symbol, status=status, limit=limit)


# ============================================================================
# MARTINGALE QUERIES
# ============================================================================


def get_martingale_state(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get martingale chain state for a symbol.

    Args:
        symbol: Trading symbol

    Returns:
        Martingale state dict or None
    """
    ctx = _get_ctx()
    return ctx.martingale.get_martingale_state(symbol)


def find_or_create_martingale_chain(
    symbol: str,
    initial_order_id: str,
    loss: float,
    chain_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Find or create a martingale chain.

    Args:
        symbol: Trading symbol
        initial_order_id: First order in the chain
        loss: Initial loss amount
        chain_id: Optional unique chain identifier

    Returns:
        Martingale chain dict
    """
    ctx = _get_ctx()
    return ctx.martingale.find_or_create_martingale_chain(
        symbol=symbol,
        initial_order_id=initial_order_id,
        loss=loss,
        chain_id=chain_id,
    )


def update_martingale_chain(chain_id: str, updates: Dict[str, Any]) -> bool:
    """
    Update martingale chain.

    Args:
        chain_id: Chain ID
        updates: Fields to update

    Returns:
        True if updated successfully
    """
    ctx = _get_ctx()
    return ctx.martingale.update_martingale_chain(chain_id, updates)


def get_active_martingale_chains() -> List[Dict[str, Any]]:
    """
    Get all active martingale chains.

    Returns:
        List of active martingale chain dicts
    """
    ctx = _get_ctx()
    return ctx.martingale.get_active_martingale_chains()


def get_martingale_chains_cursor(limit: int = 50, last_key: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Get paginated martingale chains.

    Args:
        limit: Page size
        last_key: Pagination cursor (not used in DynamoDB impl, kept for API compat)

    Returns:
        List of chain dicts
    """
    ctx = _get_ctx()
    return ctx.martingale.get_martingale_chains_cursor(limit=limit)


# ============================================================================
# SIGNAL QUERIES
# ============================================================================


def save_signal(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save a new signal.

    Args:
        data: Signal data

    Returns:
        Saved signal dict
    """
    ctx = _get_ctx()
    return ctx.signals.save_signal(data)


def mark_signal_executed(correlation_id: str, order_id: str) -> bool:
    """
    Mark signal as executed.

    Args:
        correlation_id: Signal correlation ID
        order_id: Associated order ID

    Returns:
        True if marked successfully
    """
    ctx = _get_ctx()
    return ctx.signals.mark_signal_executed(correlation_id, order_id)


def update_signal_outcome(
    correlation_id: str,
    outcome: str,
    pnl: Optional[float] = None,
) -> bool:
    """
    Update signal outcome.

    Args:
        correlation_id: Signal correlation ID
        outcome: Outcome ('WIN', 'LOSS', 'BREAKEVEN')
        pnl: Optional P&L

    Returns:
        True if updated successfully
    """
    ctx = _get_ctx()
    return ctx.signals.update_signal_outcome(correlation_id, outcome, outcome_pnl=pnl)


def get_recent_signals(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get recent signals.

    Args:
        limit: Maximum number of results

    Returns:
        List of signal dicts
    """
    ctx = _get_ctx()
    return ctx.signals.get_recent_signals(limit=limit)


def get_signal_performance_stats(days: int = 30) -> Dict[str, Any]:
    """
    Get signal performance statistics.

    Args:
        days: Number of days to analyze

    Returns:
        Stats dict
    """
    ctx = _get_ctx()
    return ctx.signals.get_signal_performance_stats(days=days)


def get_signals_cursor(limit: int = 50, last_key: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Get paginated signals.

    Args:
        limit: Page size
        last_key: Pagination cursor (not used in DynamoDB impl, kept for API compat)

    Returns:
        List of signal dicts
    """
    ctx = _get_ctx()
    return ctx.signals.get_signals_cursor(limit=limit)


# ============================================================================
# SYSTEM STATE QUERIES
# ============================================================================


def get_system_state(key: str) -> Optional[Any]:
    """
    Get system state by key.

    Args:
        key: State key

    Returns:
        State value or None
    """
    ctx = _get_ctx()
    return ctx.system_state.get_system_state(key)


def set_system_state(key: str, value: Any) -> bool:
    """
    Set system state.

    Args:
        key: State key
        value: State value

    Returns:
        True if set successfully
    """
    ctx = _get_ctx()
    return ctx.system_state.set_system_state(key, value)


# ============================================================================
# AUDIT LOG QUERIES
# ============================================================================


def create_audit_log(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an audit log entry.

    Args:
        data: Audit log data

    Returns:
        Created audit log dict
    """
    ctx = _get_ctx()
    return ctx.audit_log.create_audit_log(data)


def get_recent_audit_logs(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get recent audit logs.

    Args:
        limit: Maximum number of results

    Returns:
        List of audit log dicts
    """
    ctx = _get_ctx()
    return ctx.audit_log.get_recent_audit_logs(limit=limit)


def get_audit_log_cursor(limit: int = 50, last_key: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Get paginated audit logs.

    Args:
        limit: Page size
        last_key: Pagination cursor (not used in DynamoDB impl, kept for API compat)

    Returns:
        List of audit log dicts
    """
    ctx = _get_ctx()
    return ctx.audit_log.get_audit_log_cursor(limit=limit)


# ============================================================================
# STATISTICS QUERIES
# ============================================================================


def get_daily_stats(date: Optional[str] = None) -> Dict[str, Any]:
    """
    Get daily statistics.

    Note: DynamoDB OrderRepository does not expose a dedicated get_daily_stats
    method. This function returns an empty dict as a safe stub. Override by
    querying get_all_programmatic_orders and aggregating in the caller if needed.

    Args:
        date: Date string (YYYY-MM-DD), defaults to today (currently unused)

    Returns:
        Daily stats dict (empty – no dedicated implementation)
    """
    return {}


def get_overall_stats() -> Dict[str, Any]:
    """
    Get overall statistics.

    Note: DynamoDB OrderRepository does not expose a dedicated get_overall_stats
    method. This function returns an empty dict as a safe stub. Override by
    querying get_all_programmatic_orders and aggregating in the caller if needed.

    Returns:
        Overall stats dict (empty – no dedicated implementation)
    """
    return {}


# ============================================================================
# GRADUAL RECOVERY QUERIES
# ============================================================================


def get_active_gradual_recovery(symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get active gradual recovery.

    Args:
        symbol: Optional symbol filter (None for global)

    Returns:
        Recovery dict or None
    """
    ctx = _get_ctx()
    return ctx.gradual_recovery.get_active_gradual_recovery(symbol=symbol)


def create_gradual_recovery(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new gradual recovery.

    Args:
        data: Recovery data

    Returns:
        Created recovery dict
    """
    ctx = _get_ctx()
    return ctx.gradual_recovery.create_gradual_recovery(data)


def update_gradual_recovery(recovery_id: str, updates: Dict[str, Any]) -> bool:
    """
    Update gradual recovery.

    Args:
        recovery_id: Recovery ID
        updates: Fields to update

    Returns:
        True if updated successfully
    """
    ctx = _get_ctx()
    return ctx.gradual_recovery.update_gradual_recovery(recovery_id, updates)


def cancel_gradual_recovery(recovery_id: str) -> bool:
    """
    Cancel a gradual recovery.

    Args:
        recovery_id: Recovery ID

    Returns:
        True if cancelled successfully
    """
    ctx = _get_ctx()
    return ctx.gradual_recovery.cancel_gradual_recovery(recovery_id)


def get_gradual_recovery_by_id(recovery_id: str) -> Optional[Dict[str, Any]]:
    """
    Get gradual recovery by ID.

    Args:
        recovery_id: Recovery ID

    Returns:
        Recovery dict or None
    """
    ctx = _get_ctx()
    return ctx.gradual_recovery.get_gradual_recovery_by_id(recovery_id)


def get_all_gradual_recoveries(status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get all gradual recoveries.

    Args:
        status: Optional status filter
        limit: Maximum number of results

    Returns:
        List of recovery dicts
    """
    ctx = _get_ctx()
    return ctx.gradual_recovery.get_all_gradual_recoveries(status=status, limit=limit)


__all__ = [
    # Order queries
    "get_open_positions",
    "get_last_closed_order",
    "get_all_programmatic_orders",
    "get_orders_cursor",
    "is_programmatic_order",
    "get_order_by_id",
    "get_order_by_client_id",
    "update_order_status",
    "update_order_status_by_client_id",
    "mark_be_moved",
    "create_order",
    "get_orders_by_symbol",
    # Martingale queries
    "get_martingale_state",
    "find_or_create_martingale_chain",
    "update_martingale_chain",
    "get_active_martingale_chains",
    "get_martingale_chains_cursor",
    # Signal queries
    "save_signal",
    "mark_signal_executed",
    "update_signal_outcome",
    "get_recent_signals",
    "get_signal_performance_stats",
    "get_signals_cursor",
    # System state queries
    "get_system_state",
    "set_system_state",
    # Audit log queries
    "create_audit_log",
    "get_recent_audit_logs",
    "get_audit_log_cursor",
    # Statistics queries
    "get_daily_stats",
    "get_overall_stats",
    # Gradual recovery queries
    "get_active_gradual_recovery",
    "create_gradual_recovery",
    "update_gradual_recovery",
    "cancel_gradual_recovery",
    "get_gradual_recovery_by_id",
    "get_all_gradual_recoveries",
]
