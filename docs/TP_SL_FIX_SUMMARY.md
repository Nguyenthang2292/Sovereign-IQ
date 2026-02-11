# TP/SL Detection Fix - Complete Summary

## 🐛 Vấn đề ban đầu

TP/SL hiển thị **None** dù trên Binance UI có orders "Take Profit Market" và "Stop Market".

```python
[DataService] Synced TP/SL for SKL/USDT: TP=$None, SL=$None, BE=$None
```

## 🔍 Root Causes Discovered

### 1. **Order Type Detection Issue**

**Vấn đề:** 
- ccxt normalize `order['type']` từ `'TAKE_PROFIT_MARKET'` → `'market'`
- Code đang check `order.get('type')` → chỉ nhận được `'market'`
- Detection logic không match → Không nhận diện được TP orders!

**Example từ Binance API:**
```json
{
  "type": "market",              // ❌ ccxt normalized (generic)
  "info": {
    "type": "TAKE_PROFIT_MARKET" // ✅ Binance raw type (specific)
  }
}
```

**Fix:**
```python
# OLD (sai):
order_type = order.get("type", "").upper()  # → "MARKET"

# NEW (đúng):
order_type_info = order.get("info", {}).get("type", "").upper()  # → "TAKE_PROFIT_MARKET"
order_type = order_type_info if order_type_info else order_type_main
```

### 2. **Symbol Format Mismatch**

**Vấn đề:**
- Binance API trả về: `"SKL/USDT"` hoặc `"SKL/USDT:USDT"`
- Database lưu: `"SKLUSDT:USDT"`
- Database query với: `Order.symbol == "SKL/USDT"` → ❌ No match!

**Fix:**
```python
@staticmethod
def _normalize_symbol_for_db(symbol: str) -> str:
    """
    Normalize: "SKL/USDT" → "SKLUSDT"
    """
    return symbol.replace("/", "").split(":")[0]

# Query with multiple formats:
order = session.query(Order).filter(
    Order.status == "OPEN"
).filter(
    (Order.symbol == symbol) | 
    (Order.symbol == symbol_normalized) |
    (Order.symbol.like(f"{symbol_normalized}%"))  # SKLUSDT:USDT
).first()
```

## ✅ Files Fixed

### 1. `modules/auto_trade/gui/utils/position_sync_service.py`
- ✅ Use `order['info']['type']` for accurate detection
- ✅ Check both `stopPrice` and `triggerPrice`
- ✅ Improved logging with both type fields

### 2. `modules/auto_trade/gui/utils/tp_sl_sync.py`
- ✅ Same detection logic as position_sync
- ✅ Added `_normalize_symbol_for_db()` helper
- ✅ Query with multiple symbol formats (LIKE pattern)
- ✅ Enhanced debug logging

## 🧪 Test Results

### Test Script: `test_inspect_binance_order_structure.py`

**Found actual order structure:**
```json
{
  "type": "market",
  "info": {
    "type": "TAKE_PROFIT_MARKET",  ← This is the real type!
    "stopPrice": "0.00668",
    "status": "NEW"
  }
}
```

### Test Script: `test_tp_sl_direct.py`

**Before fix:**
```
TP (before):  None
SL (before):  None
```

**After fix:**
```
✅ Fetch TP từ Binance: $0.00668
✅ Sync vào database:    TP = $0.00668
✅ Database updated!
```

## 📋 Detection Logic - Complete Guide

### Binance Order Types (TP/SL)

**Take Profit Orders:**
- `TAKE_PROFIT` (limit)
- `TAKE_PROFIT_MARKET` ← Most common
- `TAKE_PROFIT_LIMIT`

**Stop Loss Orders:**
- `STOP` (stop-limit)
- `STOP_MARKET` ← Most common
- `STOP_LOSS`
- `STOP_LOSS_MARKET`
- `TRAILING_STOP_MARKET`

### Detection Code

```python
# Get raw Binance type from info field
order_type = order.get("info", {}).get("type", "").upper()

# TP Detection
if "TAKE_PROFIT" in order_type:
    tp_price = order.get("stopPrice") or order.get("triggerPrice")
    
# SL Detection  
elif "STOP" in order_type and ("MARKET" in order_type or "LOSS" in order_type):
    sl_price = order.get("stopPrice") or order.get("triggerPrice")
```

## 🔧 How to Use

### 1. Manual Sync (GUI Button)
1. Click **🔄 Sync from Binance** button
2. System fetches positions + TP/SL orders
3. Syncs to local database
4. UI updates automatically

### 2. Auto Sync (Background)
- Runs every refresh cycle in GUI
- Fetches TP/SL from Binance
- Updates database if changed
- Detects Break Even moves

### 3. Standalone Test
```bash
# Test sync manually
python test_tp_sl_direct.py

# Inspect order structure
python test_inspect_binance_order_structure.py

# Check database content
python test_check_database.py
```

## 📊 Technical Diagrams

### Order Type Detection Flow

```
Binance API Response
        ↓
┌─────────────────────┐
│ order['type']       │ → "market" (generic)
│ order['info']['type']│ → "TAKE_PROFIT_MARKET" (specific) ✅
└─────────────────────┘
        ↓
Check info['type'] FIRST
        ↓
if "TAKE_PROFIT" in type:
    → Extract stopPrice → Save as TP
        ↓
Database: order.take_profit = 0.00668 ✅
```

### Symbol Normalization Flow

```
Various formats from Binance:
- "SKL/USDT"
- "SKL/USDT:USDT"
- "SKLUSDT"
        ↓
Normalize → "SKLUSDT"
        ↓
Database Query:
  symbol == "SKL/USDT" OR
  symbol == "SKLUSDT" OR
  symbol LIKE "SKLUSDT%" ✅
        ↓
Match: "SKLUSDT:USDT" in DB ✅
```

## ⚠️ Important Notes

### Why SL might be None

1. **Not set on Binance** - Only TP was configured
2. **Already filled** - SL order executed and closed
3. **Cancelled** - Order was manually cancelled
4. **Different position side** - Using HEDGE mode with separate orders

### Symbol Format Handling

Always use **slash format** (`"SKL/USDT"`) when calling:
- `client.fetch_open_orders(symbol)`
- `client.fetch_positions(symbols=[symbol])`

Database will **auto-normalize** internally:
- Queries work with any format
- Stored format: `"SKLUSDT:USDT"` (with market suffix)

## 🎯 Expected Behavior

**When TP/SL orders exist on Binance:**
```
GUI Display:
  SKL/USDT | 3121 | TP: $0.00668 | SL: None | BE: --
                    ^^^^^^^^^^^^^
                    Now shows correctly!
```

**When no orders exist:**
```
GUI Display:
  SKL/USDT | 3121 | TP: -- | SL: -- | BE: --
  
(This is correct - no orders to display)
```

## 🚀 Next Steps

1. **Restart GUI** to see updated code
2. **Click Sync button** to pull latest from Binance
3. **Add SL order** on Binance if desired
4. **Verify** both TP and SL display correctly

## 📝 Code Quality Improvements

- ✅ Comprehensive logging for debugging
- ✅ Multiple test scripts for validation
- ✅ Symbol normalization utility
- ✅ Robust error handling
- ✅ Type detection from raw API data
- ✅ Database query flexibility
- ✅ Documentation and examples

---

**Status:** ✅ FIXED and TESTED
**Date:** 2026-02-11
**Tested On:** Binance Futures (Production)
