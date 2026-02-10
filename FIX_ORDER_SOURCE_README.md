# Fix Order Source in Database

## 🐛 Problem

AutoTrade orders are marked as `order_source='MANUAL'` instead of `order_source='PROGRAMMATIC'` in the database, causing TP/SL/BE to show as "N/A" in the GUI.

## ✅ Solution

Run these scripts to fix the issue safely.

---

## 📋 Step-by-Step Guide

### Step 1: Check Current Status

```bash
python test_check_order_source.py
```

**Expected output:**
- Shows all orders and their `order_source` values
- Identifies which orders need fixing

### Step 2: Backup Database (IMPORTANT!)

```bash
python backup_database.py
```

**This creates a backup in `data/backups/` directory**
- Backup format: `crypto_trading_backup_YYYYMMDD_HHMMSS.db`
- Shows last 5 backups
- Safe to run multiple times

### Step 3: Dry Run (Preview Changes)

```bash
python fix_order_source.py
```

**This shows what WOULD be changed WITHOUT changing it**
- Lists all orders that will be updated
- Shows before/after values
- 100% safe, doesn't modify database

### Step 4: Apply Fix (LIVE MODE)

```bash
python fix_order_source.py --live
```

**This ACTUALLY updates the database**
- 3-second countdown to cancel (Ctrl+C)
- Updates all AutoTrade orders to `order_source='PROGRAMMATIC'`
- Verifies fix after completion

### Step 5: Verify Fix

```bash
python test_check_order_source.py
```

**Check that:**
- All AutoTrade orders now show `order_source='PROGRAMMATIC'`
- No more "MANUAL" orders with `execution_mode='AUTO'`

### Step 6: Test in GUI

```bash
python run_auto_trade_gui.py
```

**Expected result:**
- TP/SL/BE now display correctly (no more N/A)
- Click "🔄 Refresh" to reload positions
- WebSocket updates show TP/SL values

---

## 🔄 Restore from Backup (if needed)

If something goes wrong:

```bash
python backup_database.py restore data/backups/crypto_trading_backup_YYYYMMDD_HHMMSS.db
```

**Replace `YYYYMMDD_HHMMSS` with actual backup timestamp**

---

## 📊 What the Fix Does

### SQL Equivalent:
```sql
UPDATE orders 
SET order_source = 'PROGRAMMATIC' 
WHERE execution_mode = 'AUTO' 
  AND order_source = 'MANUAL';
```

### Logic:
- Finds all orders with `execution_mode='AUTO'` (placed by AutoTrade)
- But have `order_source='MANUAL'` (incorrect)
- Updates them to `order_source='PROGRAMMATIC'` (correct)

### Why This Fixes TP/SL Display:
- Before: Query searched for `order_source='PROGRAMMATIC'` → Found 0 → TP/SL = N/A
- After: Query finds orders with any source → Found 1 → TP/SL displayed ✅

---

## 🎯 Quick Command Summary

```bash
# 1. Check status
python test_check_order_source.py

# 2. Backup (REQUIRED!)
python backup_database.py

# 3. Preview changes (safe)
python fix_order_source.py

# 4. Apply fix (LIVE)
python fix_order_source.py --live

# 5. Verify
python test_check_order_source.py

# 6. Restore if needed
python backup_database.py restore data/backups/crypto_trading_backup_YYYYMMDD_HHMMSS.db
```

---

## ⚠️ Safety Features

1. **Backup First**: Always create backup before applying changes
2. **Dry Run**: Default mode shows preview without changing
3. **Countdown**: 3-second cancel window in live mode
4. **Verification**: Automatic check after update
5. **Restore**: Easy restore from backup if needed

---

## 🔍 Troubleshooting

### Issue: "No orders need fixing"
- ✅ Good! Your database is already correct
- No action needed

### Issue: "Database not found"
- Check if app is using a different database path
- Look in `data/`, `modules/auto_trade/data/`, or root directory

### Issue: TP/SL still showing N/A after fix
- Restart the app: `python run_auto_trade_gui.py`
- Click "🔄 Refresh" button in GUI
- Check logs: Should see `[WebSocket] Found X orders for SYMBOL`

### Issue: Want to undo changes
- Use restore command with backup file
- Backups are kept forever (manual cleanup needed)

---

## 📝 Notes

- This only affects **existing orders** in database
- **New orders** placed after the code fix will be correct automatically
- The GUI code fix (querying all orders) works regardless of order_source
- This database fix is optional but recommended for consistency

---

## 💡 Prevention

To prevent this issue in the future, the code has been updated:
1. **GUI Query Fix**: Now queries ALL open orders (not just PROGRAMMATIC)
2. **Order Creation**: Verify `OrderManager.execute_signal` sets `order_source='PROGRAMMATIC'`

Both fixes are already applied in your codebase!
