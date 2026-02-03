# Order Tagging System

**Task 3.2 Implementation** - Unique client order ID generation and metadata tagging for programmatic orders.

## 📋 Overview

The Order Tagging System provides:
- Generate unique client order IDs with `AT_` prefix
- Distinguish programmatic orders from manual Binance trades
- Track order metadata for database synchronization
- Support Martingale chain and signal correlation IDs

## 🎯 Key Features

### 1. **Unique Client Order ID Generation**
Format: `AT_{timestamp}_{symbol}_{random_suffix}`

Example: `AT_1707043200_BTCUSDT_a1b2c3`

## 🚀 Quick Start

```python
from modules.auto_trade.execution.order_tagging import (
    generate_order_id,
    tag_programmatic_order,
    is_auto_trade_order
)

# Generate client order ID
client_order_id = generate_order_id('BTCUSDT')
# Output: AT_1707043200_BTCUSDT_abc123

# Tag order with full metadata
metadata = tag_programmatic_order('BTCUSDT', signal_id='SIGNAL_001')

# Check if order is programmatic
is_programmatic = is_auto_trade_order(client_order_id)
```

## ✅ Benefits

1. **Clear Identification**: AT_ prefix instantly identifies auto_trade orders
2. **Database Filtering**: Only programmatic orders synced and queried
3. **Position Reconciliation**: Match Binance orders with database records
4. **Audit Trail**: Complete tracking of all automated trades

## 🧪 Testing

```bash
python test_order_tagging.py
```

**Status**: ✅ Production Ready (10/10 tests passing)
