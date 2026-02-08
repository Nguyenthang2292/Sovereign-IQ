"""
Modular query package for auto_trade database operations.

Each module contains related query functions:
- orders: Order management and tracking
- martingale: Martingale chain management
- signals: Signal lifecycle and performance
- system_state: Key-value state storage
- audit_logs: Audit trail management
- statistics: Performance analytics
- gradual_recovery: Gradual recovery strategy

Usage:
    from modules.auto_trade.database import queries

    # Option 1: Import from facade (backward compatible)
    orders = queries.get_open_positions(session)

    # Option 2: Import from specific module (recommended for new code)
    from modules.auto_trade.database.queries import orders as order_queries
    orders = order_queries.get_open_positions(session)
"""

# Re-export all functions from sub-modules for convenience (explicit imports for static analysis)
from .audit_logs import create_audit_log, get_audit_log_cursor, get_recent_audit_logs
from .gradual_recovery import (
    cancel_gradual_recovery,
    create_gradual_recovery,
    get_active_gradual_recovery,
    get_all_gradual_recoveries,
    get_gradual_recovery_by_id,
    update_gradual_recovery,
)
from .martingale import (
    find_or_create_martingale_chain,
    get_active_martingale_chains,
    get_martingale_chains_cursor,
    get_martingale_state,
    update_martingale_chain,
)
from .orders import (
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
from .signals import (
    get_recent_signals,
    get_signal_performance_stats,
    get_signals_cursor,
    mark_signal_executed,
    save_signal,
    update_signal_outcome,
)
from .statistics import get_daily_stats, get_overall_stats
from .system_state import get_system_state, set_system_state

__all__ = [
    "create_audit_log",
    "get_audit_log_cursor",
    "get_recent_audit_logs",
    "cancel_gradual_recovery",
    "create_gradual_recovery",
    "get_active_gradual_recovery",
    "get_all_gradual_recoveries",
    "get_gradual_recovery_by_id",
    "update_gradual_recovery",
    "find_or_create_martingale_chain",
    "get_active_martingale_chains",
    "get_martingale_chains_cursor",
    "get_martingale_state",
    "update_martingale_chain",
    "create_order",
    "get_all_programmatic_orders",
    "get_last_closed_order",
    "get_order_by_client_id",
    "get_order_by_id",
    "get_orders_by_symbol",
    "get_orders_cursor",
    "get_open_positions",
    "is_programmatic_order",
    "mark_be_moved",
    "update_order_status",
    "update_order_status_by_client_id",
    "get_recent_signals",
    "get_signal_performance_stats",
    "get_signals_cursor",
    "mark_signal_executed",
    "save_signal",
    "update_signal_outcome",
    "get_daily_stats",
    "get_overall_stats",
    "get_system_state",
    "set_system_state",
]
