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

# Re-export all functions from sub-modules for convenience
from .orders import *
from .martingale import *
from .signals import *
from .system_state import *
from .audit_logs import *
from .statistics import *
from .gradual_recovery import *
