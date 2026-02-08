# Database Queries Module

Modular query package for auto_trade database operations.

## Overview

The `queries` module provides a clean, organized interface to database operations for the auto_trade system. Previously housed in a single 1243-line `queries.py` file, the functionality has been refactored into focused sub-modules with clear responsibilities.

**Key Features:**
- ✅ **100% Backward Compatible** - All existing imports continue to work
- ✅ **Modular Organization** - 7 focused modules with single responsibilities
- ✅ **Type-Safe** - Full type hints throughout
- ✅ **Performance Optimized** - Cursor-based pagination and SQL aggregation
- ✅ **Programmatic-First** - Defaults to filtering system-generated orders

## Module Structure

```
queries/
├── __init__.py              # Package exports (use for new code)
├── _shared.py               # Common imports and constants
├── orders.py                # Order management (15 functions)
├── martingale.py            # Martingale chain tracking (5 functions)
├── signals.py               # Signal lifecycle (7 functions)
├── system_state.py          # Key-value state (2 functions)
├── audit_logs.py            # Audit trail (3 functions)
├── statistics.py            # Performance analytics (2 functions)
└── gradual_recovery.py      # Recovery strategies (7 functions)
```

## Sub-Modules

### orders.py - Order Management

**15 functions** for managing trading orders with programmatic filtering.

**Key Functions:**
- `get_open_positions()` - Retrieve all open positions
- `get_last_closed_order()` - Get most recent closed order
- `create_order()` - Create new programmatic order with validation
- `update_order_status()` - Update order status and P&L
- `mark_be_moved()` - Mark break-even triggered
- `get_orders_cursor()` - Cursor-based pagination for performance

**Critical Feature:** All order queries default to `order_source="PROGRAMMATIC"` to filter out manual Binance trades.

**Example:**
```python
from modules.auto_trade.database.queries.orders import get_open_positions

# Get all open programmatic positions
positions = get_open_positions(session)

# Get open positions for specific symbol
btc_positions = get_open_positions(session, symbol="BTCUSDT")
```

### martingale.py - Martingale Chain Tracking

**5 functions** for managing Martingale recovery chains.

**Key Functions:**
- `get_martingale_state()` - Get active chain for symbol
- `find_or_create_martingale_chain()` - Get or create chain
- `update_martingale_chain()` - Update chain progress
- `get_active_martingale_chains()` - Get all active chains
- `get_martingale_chains_cursor()` - Cursor pagination

**Example:**
```python
from modules.auto_trade.database.queries.martingale import (
    get_martingale_state,
    update_martingale_chain
)

# Check if symbol has active martingale chain
chain = get_martingale_state(session, "BTCUSDT")

if chain:
    # Update chain progress
    update_martingale_chain(
        session,
        chain_id=chain.chain_id,
        current_step=2,
        latest_order_id="ORDER_123",
        total_loss=50.0
    )
```

### signals.py - Signal Lifecycle Management

**7 functions** for tracking trading signals from generation to outcome.

**Key Functions:**
- `save_signal()` - Create new signal
- `mark_signal_executed()` - Mark signal as executed with order ID
- `update_signal_outcome()` - Record outcome (WIN/LOSS/BREAKEVEN)
- `get_signal_performance_stats()` - Calculate win rate and P&L metrics
- `get_signals_cursor()` - Cursor pagination

**Example:**
```python
from modules.auto_trade.database.queries.signals import (
    save_signal,
    mark_signal_executed,
    get_signal_performance_stats
)

# Save new signal
signal = save_signal(
    session,
    correlation_id="SIG_123",
    symbol="BTCUSDT",
    signal_type="LONG",
    confidence=0.85,
    atc_score=0.9,
    xgboost_score=0.8
)

# Mark as executed
mark_signal_executed(session, "SIG_123", "ORDER_456")

# Get performance stats
stats = get_signal_performance_stats(session, symbol="BTCUSDT", days=30)
print(f"Win rate: {stats['win_rate']:.2f}%")
```

### system_state.py - Key-Value State Management

**2 functions** for storing and retrieving system configuration state.

**Key Functions:**
- `get_system_state()` - Retrieve typed state value
- `set_system_state()` - Store typed state value

**Supported Types:** string, integer, float, boolean, json

**Example:**
```python
from modules.auto_trade.database.queries.system_state import (
    get_system_state,
    set_system_state
)

# Store configuration
set_system_state(
    session,
    key="max_positions",
    value=5,
    value_type="integer",
    description="Maximum concurrent positions"
)

# Retrieve configuration (returns typed value)
max_positions = get_system_state(session, "max_positions")  # Returns int: 5
```

### audit_logs.py - Audit Trail Management

**3 functions** for creating and querying audit logs.

**Key Functions:**
- `create_audit_log()` - Create audit entry
- `get_recent_audit_logs()` - Query logs with filters
- `get_audit_log_cursor()` - Cursor pagination

**Example:**
```python
from modules.auto_trade.database.queries.audit_logs import (
    create_audit_log,
    get_recent_audit_logs
)

# Log important event
create_audit_log(
    session,
    event_type="ORDER_CREATED",
    event_category="TRADING",
    severity="INFO",
    event_summary="Created LONG order for BTCUSDT",
    order_id="ORDER_789"
)

# Query critical errors
errors = get_recent_audit_logs(
    session,
    severity="ERROR",
    limit=50
)
```

### statistics.py - Performance Analytics

**2 functions** for calculating trading performance metrics.

**Key Functions:**
- `get_daily_stats()` - Daily statistics with database aggregation
- `get_overall_stats()` - All-time statistics

**Performance Note:** Uses SQL aggregation for efficiency with large datasets.

**Example:**
```python
from modules.auto_trade.database.queries.statistics import (
    get_daily_stats,
    get_overall_stats
)

# Get last 30 days of statistics
daily_stats = get_daily_stats(session, days=30)
for day in daily_stats:
    print(f"{day['date']}: {day['total_trades']} trades, {day['win_rate']:.2f}% win rate")

# Get overall statistics
overall = get_overall_stats(session)
print(f"Total P&L: ${overall['total_pnl']:.2f}")
print(f"Win Rate: {overall['win_rate']:.2f}%")
print(f"Best Trade: ${overall['best_trade']:.2f}")
```

### gradual_recovery.py - Gradual Recovery Strategies

**7 functions** for managing gradual recovery from losses.

**Key Functions:**
- `get_active_gradual_recovery()` - Get active recovery (global or per-symbol)
- `create_gradual_recovery()` - Start new recovery
- `update_gradual_recovery()` - Update recovery progress
- `cancel_gradual_recovery()` - Cancel recovery
- `get_gradual_recovery_by_id()` - Get specific recovery
- `get_all_gradual_recoveries()` - Query all recoveries

**Modes:**
- **Global Recovery** (`symbol=None` or `symbol="GLOBAL"`) - Tracks portfolio-wide recovery
- **Per-Symbol Recovery** (`symbol="BTCUSDT"`) - Tracks symbol-specific recovery

**Example:**
```python
from modules.auto_trade.database.queries.gradual_recovery import (
    get_active_gradual_recovery,
    create_gradual_recovery,
    update_gradual_recovery
)

# Check for active global recovery
recovery = get_active_gradual_recovery(session, symbol=None)

if not recovery:
    # Start new recovery
    recovery = create_gradual_recovery(
        session,
        recovery_id="REC_001",
        initial_loss=100.0,
        config={"target_percentage": 0.3, "max_trades": 50}
    )

# Update progress
update_gradual_recovery(
    session,
    recovery_id="REC_001",
    remaining_loss=80.0,
    total_profit_accumulated=20.0,
    recovery_percentage=20.0,
    trades_count=5
)
```

### _shared.py - Common Utilities

**Internal module** containing shared imports, types, and constants used across all query modules.

**Contents:**
- Common type imports (datetime, typing)
- SQLAlchemy imports (desc, func, Session)
- Model re-exports (Order, Signal, etc.)
- Constants (DEFAULT_ORDER_SOURCE, DEFAULT_EXECUTION_MODE)

**Note:** This is an internal module. Import from specific query modules instead.

## Usage Patterns

### Pattern 1: Backward Compatible (Existing Code)

All existing imports continue to work without modification:

```python
# OLD: Still works (facade pattern)
from modules.auto_trade.database.queries import (
    get_open_positions,
    save_signal,
    get_daily_stats
)

# Or relative imports (also still work)
from database.queries import get_open_positions, mark_be_moved
```

### Pattern 2: Modular Imports (Recommended for New Code)

Import directly from specific modules for better organization:

```python
# NEW: Import from specific modules
from modules.auto_trade.database.queries.orders import get_open_positions
from modules.auto_trade.database.queries.signals import save_signal
from modules.auto_trade.database.queries.statistics import get_daily_stats

# Or import entire module
from modules.auto_trade.database.queries import orders, signals, statistics

positions = orders.get_open_positions(session)
stats = statistics.get_daily_stats(session, days=30)
```

### Pattern 3: Package-Level Imports

Import from package for convenience:

```python
# Import from package __init__
from modules.auto_trade.database import queries

positions = queries.get_open_positions(session)
signal = queries.save_signal(session, ...)
```

## Migration Guide

### For Existing Code

**No changes required!** The facade pattern ensures 100% backward compatibility.

```python
# This continues to work exactly as before
from modules.auto_trade.database.queries import get_open_positions

orders = get_open_positions(session)
```

### For New Code

**Recommended:** Use direct module imports for clarity:

```python
# Clear and explicit - recommended for new code
from modules.auto_trade.database.queries.orders import (
    get_open_positions,
    create_order,
    update_order_status
)

from modules.auto_trade.database.queries.signals import (
    save_signal,
    mark_signal_executed
)
```

### Best Practices

1. **New Code:** Import directly from sub-modules
   ```python
   from modules.auto_trade.database.queries.orders import get_open_positions
   ```

2. **Existing Code:** No changes needed - keep using facade
   ```python
   from modules.auto_trade.database.queries import get_open_positions
   ```

3. **Module Imports:** Use when calling multiple functions from same module
   ```python
   from modules.auto_trade.database.queries import orders

   positions = orders.get_open_positions(session)
   order = orders.create_order(session, order_data)
   ```

4. **Type Hints:** Import types from models, not from queries
   ```python
   from modules.auto_trade.database.models import Order, Signal
   from modules.auto_trade.database.queries.orders import get_open_positions

   def process_orders() -> list[Order]:
       return get_open_positions(session)
   ```

## Backward Compatibility

### Facade Pattern

The parent `queries.py` file acts as a **facade** that re-exports all functions from sub-modules:

```python
# queries.py
from .queries.orders import get_open_positions, create_order, ...
from .queries.signals import save_signal, mark_signal_executed, ...
# ... all 41 functions re-exported

__all__ = [
    "get_open_positions",
    "create_order",
    "save_signal",
    # ... all functions listed
]
```

### Import Compatibility Guarantees

✅ **All existing imports work without modification**

| Current Import | Status |
|----------------|--------|
| `from modules.auto_trade.database.queries import get_open_positions` | ✅ Works |
| `from database.queries import get_open_positions` | ✅ Works |
| `from modules.auto_trade.database.queries import save_signal` | ✅ Works |
| `from database.queries import mark_be_moved` | ✅ Works |

**Verified Files** (no changes needed):
- ✅ `modules/auto_trade/strategies/recovery_manager.py`
- ✅ `modules/auto_trade/execution/trailing_stop_job.py`
- ✅ `modules/auto_trade/execution/negative_breakeven_ws_handler.py`
- ✅ `modules/auto_trade/execution/negative_breakeven_job.py`
- ✅ `tests/auto_trade/test_performance_10k_orders.py`

## Function Index

### Orders (15 functions)
- `get_open_positions(session, symbol=None, order_source="PROGRAMMATIC")`
- `get_last_closed_order(session, symbol=None, order_source="PROGRAMMATIC")`
- `get_all_programmatic_orders(session, status=None, symbol=None, limit=100, offset=0)`
- `get_orders_cursor(session, last_id=None, limit=50, order_source="PROGRAMMATIC", status=None, symbol=None)`
- `is_programmatic_order(session, order_id)`
- `get_order_by_id(session, order_id, verify_programmatic=True)`
- `get_order_by_client_id(session, client_order_id)`
- `update_order_status_by_client_id(session, client_order_id, status, closed_at=None, pnl=None)`
- `update_order_status(session, order_id, status, pnl=None, verify_programmatic=True)`
- `mark_be_moved(session, order_id, new_stop_loss=None, new_take_profit=None, verify_programmatic=True)`
- `create_order(session, order_data)`
- `get_orders_by_symbol(session, symbol, status=None, limit=50, offset=0)`

### Martingale (5 functions)
- `get_martingale_state(session, symbol)`
- `find_or_create_martingale_chain(session, chain_id, symbol, original_loss, initial_order_id)`
- `update_martingale_chain(session, chain_id, current_step, latest_order_id, total_loss, recovered=False, recovery_pnl=0.0)`
- `get_active_martingale_chains(session)`
- `get_martingale_chains_cursor(session, last_id=None, limit=50)`

### Signals (7 functions)
- `save_signal(session, correlation_id, symbol, signal_type, confidence, atc_score=None, xgboost_score=None, gemini_score=None, **kwargs)`
- `mark_signal_executed(session, correlation_id, order_id)`
- `update_signal_outcome(session, correlation_id, outcome, outcome_pnl, outcome_duration_minutes=None)`
- `get_recent_signals(session, limit=50, executed_only=False, offset=0)`
- `get_signal_performance_stats(session, symbol=None, days=30)`
- `get_signals_cursor(session, last_id=None, limit=50, symbol=None, executed=None)`

### System State (2 functions)
- `get_system_state(session, key)`
- `set_system_state(session, key, value, value_type="string", description=None, category=None)`

### Audit Logs (3 functions)
- `create_audit_log(session, event_type, event_category, severity, event_summary, **kwargs)`
- `get_recent_audit_logs(session, limit=100, severity=None, event_type=None, offset=0)`
- `get_audit_log_cursor(session, last_id=None, limit=50, event_type=None, severity=None)`

### Statistics (2 functions)
- `get_daily_stats(session, days=30)`
- `get_overall_stats(session)`

### Gradual Recovery (7 functions)
- `get_active_gradual_recovery(session, symbol=None)`
- `create_gradual_recovery(session, recovery_id, initial_loss, config, symbol=None)`
- `update_gradual_recovery(session, recovery_id, remaining_loss=None, total_profit_accumulated=None, recovery_percentage=None, trades_count=None, win_streak=None, estimated_trades_remaining=None, status=None)`
- `cancel_gradual_recovery(session, recovery_id)`
- `get_gradual_recovery_by_id(session, recovery_id)`
- `get_all_gradual_recoveries(session, status=None, limit=50, offset=0)`

## Performance Considerations

### Cursor-Based Pagination

For large datasets, use cursor-based pagination instead of offset-based:

```python
from modules.auto_trade.database.queries.orders import get_orders_cursor

# First page
orders = get_orders_cursor(session, last_id=None, limit=50)

# Next page (using last order's ID)
if orders:
    next_orders = get_orders_cursor(session, last_id=orders[-1].id, limit=50)
```

**Benefits:**
- ✅ Consistent performance regardless of page number
- ✅ No expensive offset calculations
- ✅ Tested with 10,000+ records

### Database Aggregation

Statistics queries use SQL aggregation for performance:

```python
# Efficient: Database performs aggregation
stats = get_overall_stats(session)

# Efficient: Database groups by date
daily = get_daily_stats(session, days=30)
```

**Benefits:**
- ✅ Minimal memory usage
- ✅ Fast execution even with thousands of orders
- ✅ Returns aggregated results directly

## Testing

See test files in `tests/auto_trade/database/`:
- `test_queries_orders.py` - Order queries tests
- `test_queries_martingale.py` - Martingale queries tests
- `test_queries_signals.py` - Signal queries tests
- `test_queries_system_state.py` - System state queries tests
- `test_queries_audit_logs.py` - Audit log queries tests
- `test_queries_statistics.py` - Statistics queries tests
- `test_queries_gradual_recovery.py` - Gradual recovery queries tests
- `test_queries_backward_compat.py` - Backward compatibility tests

## Related Documentation

- **Design Document**: `QUERIES_REFACTORING_DESIGN.md` - Complete refactoring design and rationale
- **Database Models**: `models.py` - ORM model definitions
- **Database Config**: `config.py` - Database connection configuration
- **Main README**: `../../../README.md` - Project overview

## Version History

- **v2.0** (2026-02-08) - Refactored into modular sub-package structure
- **v1.0** (2026-02-03) - Original monolithic `queries.py` file

---

**Note:** This module is part of the Sovereign-IQ cryptocurrency trading analysis system. All functions default to filtering **PROGRAMMATIC** orders to exclude manual Binance trades.
