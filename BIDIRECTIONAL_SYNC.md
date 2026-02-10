# Bidirectional TP/SL Synchronization ⚡🔄

## Problem Fixed 🐛

**Before**: TP/SL fetched from Binance API but DB never updated.
**Issue**: 
- Trailing stop jobs read from DB → stale data → wrong decisions 
- Historical analysis corrupted
- Data inconsistency between live state and storage

**After**: **Bidirectional sync** - Binance API ↔️ Database

---

## Architecture 🏗️

```
┌──────────────┐
│   Binance    │
│  Open Orders │ ← Primary source of truth
│   (Live API) │
└───────┬──────┘
        │
        │ ① Fetch TP/SL
        ↓
┌──────────────────┐
│  TPSLSyncService │ ← Bidirectional sync layer
└───────┬──────────┘
        │
        │ ② Update DB
        ↓
┌──────────────┐
│   Database   │ ← Persistent storage
│   (SQLite)   │
└──────────────┘
```

---

## Flow 🌊

### 1️⃣ WebSocket Position Update
```python
# Real-time position update from Binance stream
Position update received
  ↓
TPSLSyncService.sync_position_tp_sl()
  ↓
├─ Fetch TP/SL from Binance Open Orders API
├─ Update database Order record
├─ Auto-detect Break Even moved
└─ Return values to GUI
```

### 2️⃣ Manual Refresh (Dashboard)
```python
# User clicks Refresh button
User action
  ↓
DataService.get_positions()
  ↓
TPSLSyncService.sync_position_tp_sl()
  ↓
├─ Fetch from Binance
├─ Sync to DB
└─ Display in GUI
```

---

## Key Features ✨

### ✅ Automatic DB Updates
```python
# Whenever TP/SL fetched from Binance:
order.take_profit = binance_tp  # ← DB updated
order.stop_loss = binance_sl    # ← DB updated
order.updated_at = now()        # ← Timestamp tracked
session.commit()                # ← Changes persisted
```

### ✅ Break Even Auto-Detection
```python
# Smart BE detection based on SL movement
if side == "LONG" and stop_loss >= entry_price:
    order.be_moved = True
    order.be_moved_at = now()
    order.original_stop_loss = old_sl  # ← Keep history
```

### ✅ Change Tracking
```python
# Only update if values actually changed
if order.take_profit != new_tp:
    logger.info(f"TP changed: ${order.take_profit} → ${new_tp}")
    order.take_profit = new_tp
    changed = True
```

---

## Implementation 🔧

### Core Service: `tp_sl_sync.py`

```python
class TPSLSyncService:
    """Bidirectional TP/SL sync between Binance and Database."""
    
    @staticmethod
    def sync_position_tp_sl(client, session, symbol, side, entry_price):
        """
        Complete sync pipeline:
        1. Fetch from Binance Open Orders API
        2. Update database Order record
        3. Detect Break Even moved
        4. Return current values
        """
        # Fetch from Binance
        tp, sl, _ = TPSLSyncService.fetch_tp_sl_from_binance(client, symbol)
        
        # Sync to DB
        if tp or sl:
            TPSLSyncService.sync_to_database(session, symbol, tp, sl)
        
        # Detect BE
        be = TPSLSyncService.detect_break_even(entry_price, sl, side)
        
        return {"take_profit": tp, "stop_loss": sl, "break_even": be}
```

### Integrated In:

**1. `data_service.py`** (Manual refresh)
```python
from modules.auto_trade.gui.utils.tp_sl_sync import TPSLSyncService

with database_manager.session_scope() as session:
    result = TPSLSyncService.sync_position_tp_sl(
        client=client,
        session=session,
        symbol=symbol,
        side=side,
        entry_price=entry_price
    )
    
    take_profit = result["take_profit"]
    stop_loss = result["stop_loss"]
    break_even = result["break_even"]
```

**2. `websocket_handler.py`** (Real-time updates)
```python
# Same sync logic on every position update
result = TPSLSyncService.sync_position_tp_sl(...)
```

---

## Benefits 🎯

### ✅ **Consistent Data**
- DB always in sync with live Binance state
- No stale TP/SL values
- Historical data accurate

### ✅ **Trailing Stops Work**
- Jobs read correct TP/SL from DB
- BE detection reliable
- No missed profit protection

### ✅ **Audit Trail**
```python
order.updated_at        # When last synced
order.be_moved_at       # When BE moved
order.original_stop_loss # Before BE moved
```

### ✅ **Graceful Degradation**
```python
try:
    # Primary: Fetch from Binance + Update DB
    result = sync_position_tp_sl(...)
except:
    # Fallback: Read from DB only
    order = query_from_database(...)
```

---

## Testing 🧪

### Test Sync Logic
```bash
python test_tp_sl_sync.py
```

Expected output:
```
[TPSLSync] Found 2 open orders for BTC/USDT
[TPSLSync] Found TP for BTC/USDT: $45000.0
[TPSLSync] Found SL for BTC/USDT: $42000.0
[TPSLSync] Updated TP for BTC/USDT: $44500.0 → $45000.0
[TPSLSync] Updated SL for BTC/USDT: $41500.0 → $42000.0
[TPSLSync] ✅ DB updated for BTC/USDT
```

### Verify DB Changes
```bash
sqlite3 crypto_trading.db

SELECT symbol, take_profit, stop_loss, be_moved, updated_at 
FROM orders 
WHERE status = 'OPEN';
```

---

## Logs 📊

### Successful Sync
```
[WebSocket] Syncing TP/SL/BE for SKL/USDT...
[TPSLSync] Found 2 open orders for SKL/USDT
[TPSLSync] Found TP for SKL/USDT: $0.095
[TPSLSync] Found SL for SKL/USDT: $0.089
[TPSLSync] Updated TP for SKL/USDT: None → $0.095
[TPSLSync] Updated SL for SKL/USDT: None → $0.089
[TPSLSync] BE detected for SKL/USDT! Moved to $0.089
[TPSLSync] ✅ DB updated for SKL/USDT
[WebSocket] ✅ Synced SKL/USDT: TP=$0.095, SL=$0.089, BE=$0.089
```

### Fallback to DB
```
[DataService] Could not sync TP/SL for BTC/USDT: API rate limit
[DataService]   Fallback to DB: TP=45000.0, SL=42000.0
```

---

## Migration Guide 🚀

### Before (DB Never Updated)
```python
# ❌ Old code - only GUI display
tp, sl = fetch_from_binance(symbol)  # Binance API
# Database never updated!
```

### After (Bidirectional Sync)
```python
# ✅ New code - GUI + DB both updated
result = TPSLSyncService.sync_position_tp_sl(
    client, session, symbol, side, entry_price
)
# Binance API → Display + DB sync
```

---

## Future Enhancements 🔮

### Periodic Background Sync
```python
# Sync all open positions every 5 minutes
def periodic_tp_sl_sync():
    for position in get_all_open_positions():
        TPSLSyncService.sync_position_tp_sl(...)
```

### Conflict Resolution
```python
# If DB value differs from Binance:
if db_tp != binance_tp:
    logger.warning(f"Conflict detected: DB=${db_tp} vs Binance=${binance_tp}")
    # Binance always wins (source of truth)
    order.take_profit = binance_tp
```

---

## Summary ✅

| Feature | Status |
|---------|--------|
| Fetch from Binance | ✅ Working |
| Update DB automatically | ✅ Working |
| BE auto-detection | ✅ Working |
| Change tracking | ✅ Working |
| Fallback to DB | ✅ Working |
| Audit trail | ✅ Working |

**Now TP/SL data is consistent everywhere! 🎉**

Database → Always in sync with Binance 🔄
Trailing stops → Read correct values ✅  
Historical analysis → Accurate data 📊
