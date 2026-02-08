"""
Database Query Layer for Auto Trading System
==============================================

DEPRECATED: This module is maintained for backward compatibility.
New code should import directly from `modules.auto_trade.database.queries.*`

CRITICAL: All order queries ONLY return PROGRAMMATIC orders by default.
Manual trades from Binance are excluded from queries.

Created: 2026-02-03
Refactored: 2026-02-08
"""

# Import all functions from sub-modules
from .queries.audit_logs import (
    create_audit_log,
    get_audit_log_cursor,
    get_recent_audit_logs,
)
from .queries.gradual_recovery import (
    cancel_gradual_recovery,
    create_gradual_recovery,
    get_active_gradual_recovery,
    get_all_gradual_recoveries,
    get_gradual_recovery_by_id,
    update_gradual_recovery,
)
from .queries.martingale import (
    find_or_create_martingale_chain,
    get_active_martingale_chains,
    get_martingale_chains_cursor,
    get_martingale_state,
    update_martingale_chain,
)
from .queries.orders import (
    create_order,
    get_all_programmatic_orders,
    get_last_closed_order,
    get_open_positions,
    get_order_by_client_id,
    get_order_by_id,
    get_orders_by_symbol,
    get_orders_cursor,
    is_programmatic_order,
    mark_be_moved,
    update_order_status,
    update_order_status_by_client_id,
)
from .queries.signals import (
    get_recent_signals,
    get_signal_performance_stats,
    get_signals_cursor,
    mark_signal_executed,
    save_signal,
    update_signal_outcome,
)
from .queries.statistics import (
    get_daily_stats,
    get_overall_stats,
)
from .queries.system_state import (
    get_system_state,
    set_system_state,
)

__all__ = [
    # Order queries
    "get_open_positions",
    "get_last_closed_order",
    "get_all_programmatic_orders",
    "get_orders_cursor",
    "is_programmatic_order",
    "get_order_by_id",
    "get_order_by_client_id",
    "update_order_status_by_client_id",
    "update_order_status",
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
