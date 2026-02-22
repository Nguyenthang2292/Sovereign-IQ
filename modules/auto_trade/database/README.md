# Auto Trade Database Module

**Phase 5: Module DATABASE** - Lightweight database system for order tracking, signal management, and Martingale chain monitoring.

## 📋 Overview

This module provides a complete database layer for the auto_trade system with:

- **SQLAlchemy ORM Models** - Type-safe database models
- **Programmatic Order Filtering** - Separates auto_trade orders from manual Binance trades
- **Query Layer** - High-level query functions
- **Migration System** - Schema versioning and updates
- **Backup & Recovery** - Automated backups with compression
- **Statistics** - Performance metrics and analytics

## 🗄️ Database Schema

### Core Tables

1. **`orders`** - Trading orders (PROGRAMMATIC only)
   - Order identification (order_id, client_order_id)
   - Order source tracking (`order_source`, `execution_mode`)
   - P&L tracking
   - Break-even management
   - Martingale chain linking

2. **`signals`** - Trading signals from pipeline
   - Signal quality metrics
   - Component scores (ATC, XGBoost, Gemini)
   - Execution status
   - Outcome tracking

3. **`martingale_chain`** - Martingale recovery sequences
   - Loss tracking
   - Recovery progress
   - Safety limits

4. **`system_state`** - System configuration (key-value store)

5. **`audit_log`** - Comprehensive audit trail

## 🚀 Quick Start

### Initialize Database

```python
from modules.auto_trade.database import initialize_database

# Initialize database with schema
initialize_database(db_path='data/auto_trade.db')
```

### Basic Usage

```python
from modules.auto_trade.database import (
    session_scope,
    create_order,
    get_open_positions,
    save_signal
)

# Create an order
with session_scope() as session:
    order = create_order(session, {
        'order_id': 'ORDER_123',
        'client_order_id': 'AT_20260203_BTCUSDT_abc123',
        'symbol': 'BTCUSDT',
        'side': 'LONG',
        'entry_price': 50000.0,
        'amount': 0.01,
        'leverage': 2,
        'stop_loss': 45000.0,
        'take_profit': 52500.0,
        'status': 'OPEN',
        'order_source': 'PROGRAMMATIC',  # CRITICAL
        'execution_mode': 'AUTO'
    })

# Get open positions (PROGRAMMATIC only)
with session_scope() as session:
    positions = get_open_positions(session)
    for pos in positions:
        print(f"{pos.symbol}: {pos.side} @ ${pos.entry_price}")

# Save signal
with session_scope() as session:
    signal = save_signal(
        session,
        correlation_id='SIGNAL_001',
        symbol='BTCUSDT',
        signal_type='LONG',
        confidence=0.85,
        atc_score=0.75,
        xgboost_score=0.90
    )
```

## 🔑 Key Features

### DynamoDB Single-Table Design

Module hỗ trợ backend DynamoDB qua `DB_BACKEND=dynamodb` với mô hình single-table:

- Primary Key: `pk`, `sk`
- `GSI1`: truy vấn theo symbol + trạng thái/time
- `GSI2`: global timeline theo entity type
- `GSI3`: truy vấn order programmatic theo trạng thái

### Access Patterns (6 patterns chính)

1. `get_open_positions(symbol?)` → `GSI1`/`GSI3`
2. `get_recent_signals(symbol?)` → `GSI1`/`GSI2`
3. `get_signal_performance_stats(days)` → `GSI2`
4. `get_martingale_state(symbol)` → `GSI1`
5. `get_active_gradual_recovery(symbol)` → `GSI1`
6. `get_recent_audit_logs(limit, severity?)` → `GSI2` (+ TTL)

### GSI Usage Guide

- `GSI1(gsi1pk, gsi1sk)`: symbol-centric queries (ORDER/SIGNAL/CHAIN/RECOVERY)
- `GSI2(gsi2pk, gsi2sk)`: timeline-centric queries (`ORDER`, `SIGNAL`, `CHAIN`, `RECOVERY`, `AUDIT`)
- `GSI3(gsi3pk, gsi3sk)`: programmatic order state (`PROGRAMMATIC#OPEN`, `PROGRAMMATIC#CLOSED`)

### 1. Programmatic Order Filtering

**All order queries filter by `order_source='PROGRAMMATIC'` by default.**

This ensures that:
- Only auto_trade system orders are tracked in the database
- Manual trades on Binance are excluded
- Martingale chains only track programmatic orders
- Statistics are accurate for the trading system

```python
# Get open positions - PROGRAMMATIC only by default
positions = get_open_positions(session)

# Check if an order is programmatic
if is_programmatic_order(session, order_id):
    print("This order was created by auto_trade system")
```

### 2. Client Order ID Tagging

All programmatic orders use a unique client_order_id with prefix `AT_`:

```python
client_order_id = f"AT_{timestamp}_{symbol}_{random_suffix}"
# Example: AT_20260203_BTCUSDT_a1b2c3
```

This enables:
- Fast identification of auto_trade orders
- Position reconciliation with Binance
- Filtering manual trades during sync

### 3. Transaction Management

```python
from modules.auto_trade.database import transaction

with session_scope() as session:
    with transaction(session):
        # Multiple operations in single transaction
        order1 = create_order(session, {...})
        order2 = create_order(session, {...})
        # Auto-commit on success, rollback on error
```

### 4. Statistics & Analytics

```python
from modules.auto_trade.database import (
    get_overall_stats,
    get_daily_stats,
    get_signal_performance_stats
)

# Overall trading statistics
stats = get_overall_stats(session)
print(f"Win rate: {stats['win_rate']:.1f}%")
print(f"Total P&L: ${stats['total_pnl']:.2f}")

# Daily performance
daily = get_daily_stats(session, days=30)

# Signal performance
signal_perf = get_signal_performance_stats(session, symbol='BTCUSDT')
```

### 5. Backup & Recovery

```python
from modules.auto_trade.database import (
    create_database_backup,
    restore_latest_backup,
    list_all_backups
)

# Create backup (compressed)
backup_path = create_database_backup(compress=True)

# List backups
backups = list_all_backups()

# Restore from latest backup
restore_latest_backup(db_path='data/auto_trade.db')
```

## 📊 Query Examples

### Order Queries

```python
# Get all programmatic orders
orders = get_all_programmatic_orders(session, status='CLOSED', limit=50)

# Get last closed order
last_order = get_last_closed_order(session, symbol='BTCUSDT')

# Update order status
update_order_status(session, order_id, 'CLOSED', pnl=125.50)

# Mark break-even moved
mark_be_moved(session, order_id, new_stop_loss=entry_price)
```

### Martingale Queries

```python
# Get active Martingale chain for symbol
chain = get_martingale_state(session, 'BTCUSDT')

# Create or find chain
chain = find_or_create_martingale_chain(
    session, chain_id, symbol, original_loss, initial_order_id
)

# Update chain progress
update_martingale_chain(
    session, chain_id, current_step=2,
    latest_order_id=order_id, total_loss=-200.0
)
```

### Signal Queries

```python
# Mark signal as executed
mark_signal_executed(session, correlation_id, order_id)

# Update outcome after order closes
update_signal_outcome(session, correlation_id, 'WIN', outcome_pnl=150.0)

# Get recent signals
signals = get_recent_signals(session, limit=20, executed_only=True)
```

## 🛠️ Database Utilities

### Data Export

```python
from modules.auto_trade.database import DataExporter

# Export to CSV
DataExporter.export_to_csv(
    session, Order, 'exports/orders.csv',
    filters={'status': 'CLOSED'}
)

# Export to JSON
DataExporter.export_to_json(
    session, Signal, 'exports/signals.json', pretty=True
)

# Export all data
export_all_data(session, output_dir='data/exports')
```

### Database Maintenance

```python
from modules.auto_trade.database import DatabaseCleaner

# Cleanup old audit logs (keep 90 days)
deleted = DatabaseCleaner.cleanup_old_records(
    session, AuditLog, days_to_keep=90
)

# Archive old closed orders
archived = DatabaseCleaner.archive_old_orders(
    session, days_to_keep=90, archive_path='data/archive'
)
```

## 🔧 Configuration

### Database Path

Default: `data/auto_trade.db`

```python
# Custom database path
from modules.auto_trade.database import get_db_manager

db_manager = get_db_manager(db_path='custom/path/database.db')
```

### Connection Pool

```python
db_manager = DatabaseManager(
    db_path='data/auto_trade.db',
    pool_size=5,
    max_overflow=10,
    echo=False  # Set True for SQL logging
)
```

## 📁 Module Structure

```
modules/auto_trade/database/
├── __init__.py           # Module exports
├── models.py             # SQLAlchemy ORM models
├── queries.py            # Query functions
├── schema.sql            # Database schema
├── migrations.py         # Migration system
├── backup.py             # Backup & recovery
└── utils.py              # Utilities & helpers
```

## ✅ Testing

Run the test script:

```bash
python test_database_phase5.py
```

This verifies:
- Database initialization
- Order CRUD operations
- Signal management
- Martingale chains
- Statistics queries
- Backup system

## 🔐 Security & Best Practices

1. **Programmatic Filtering**: Always use default filters to exclude manual trades
2. **Client Order ID**: Always tag orders with `AT_` prefix
3. **Transactions**: Use `transaction()` context manager for multi-step operations
4. **Backups**: Enable automated daily backups
5. **Cleanup**: Regularly archive and cleanup old records
6. **Connection Pool**: Use session_scope() for proper connection management

## 📈 Performance

- **WAL Mode**: Enabled for better concurrent access
- **Indexes**: Optimized for common queries
- **Connection Pool**: Reuses connections efficiently
- **Batch Operations**: Supported for bulk inserts

## 🔗 Integration with Other Phases

- **Phase 3 (Execution)**: Creates orders in database
- **Phase 4 (Monitoring)**: Updates order status, manages Martingale
- **Phase 6 (Integration)**: Main loop uses all database functions
- **Phase 7 (Deployment)**: Backup system for production

## 📝 License

Part of Sovereign-IQ Auto Trading System
Created: 2026-02-03
