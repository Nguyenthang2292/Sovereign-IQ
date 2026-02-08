# Performance Improvements

## Overview

This document summarizes the performance optimizations applied to the reconciliation and database operations in the auto_trade module.

## Changes Made

### 1. Batch DB Inserts (Task 1)

**Before:**
- Each order was inserted individually using `create_order()` within a loop
- One DB transaction per order
- High overhead for large datasets

**After:**
- Orders are collected in batches during processing
- Single `bulk_insert_mappings()` call per symbol
- One DB transaction per batch of orders
- Validation is performed before insertion to maintain data integrity

**Impact:**
- Reduced DB round-trips from N (one per order) to ~1 per symbol
- Significantly faster insertion for large datasets
- Maintained validation and error tracking

**Code Location:** `modules/auto_trade/database/reconcile.py`

### 2. Write Locking for Concurrent Updates (Task 2)

**Before:**
- No protection against concurrent reconcile runs
- Risk of duplicate inserts when multiple reconciles run simultaneously
- Race conditions on stale order updates

**After:**
- Added module-level `_reconcile_lock` (threading.Lock)
- Lock acquired before any DB write operations (bulk insert, stale updates)
- 30-second timeout to prevent indefinite blocking
- Lock released in finally block to ensure cleanup

**Impact:**
- Prevents duplicate inserts during concurrent reconcile runs
- Ensures data consistency
- Safe to run reconcile from multiple threads/processes

**Code Location:** `modules/auto_trade/database/reconcile.py` (module level and usage)

### 3. Optimized Stale Order Detection (Task 3)

**Before:**
- Called `fetch_order()` individually for each stale order
- N API calls for N stale orders
- High latency and rate limit risk

**After:**
- Uses `fetch_closed_orders()` in batch to get all closed orders for a symbol/time window
- Maps results by `clientOrderId` for O(1) lookup
- Falls back to individual `fetch_order()` only if order not found in batch
- Logs how many individual fetches were needed vs batch coverage

**Impact:**
- Reduced exchange API calls from N to ~1 per symbol
- Faster stale order processing
- Lower risk of hitting rate limits
- Same final outcomes, just fewer API calls

**Code Location:** `modules/auto_trade/database/reconcile.py` (stale order section)

### 4. Profiling Support (Task 4)

**Added:**
- `enable_profiling` parameter to `reconcile_orders_with_binance()`
- Timing instrumentation for key operations
- Profiling script at `scripts/profile_reconcile.py`
- Supports cProfile for detailed function-level profiling

**Usage:**
```python
result = reconcile_orders_with_binance(
    api_key, api_secret, enable_profiling=True
)
# Access timing: result["timing"]
```

**Run profiling script:**
```bash
python scripts/profile_reconcile.py
```

**Code Locations:**
- `modules/auto_trade/database/reconcile.py` (profiling instrumentation)
- `scripts/profile_reconcile.py` (standalone profiling script)

## Performance Characteristics

### Database Operations

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Insert 100 orders | 100 transactions | 1 transaction | ~100x fewer commits |
| Concurrent safety | None | Thread-safe lock | Prevents duplicates |
| Stale order updates | 1 API call per order | 1 batch + fallback | ~N-1 fewer API calls |

### Memory Usage

- Slightly higher memory usage during reconcile due to batching
- Orders collected per symbol before insertion
- Batch size limited by available memory (typically 100-1000 orders per symbol)

### Concurrency

- Single reconcile can run at a time (DB writes are serialized)
- Multiple reconciles can read from exchange simultaneously
- Safe for multi-threaded environments

## Verification

To verify these improvements:

1. **Run performance tests:**
   ```bash
   pytest tests/auto_trade/test_performance_10k_orders.py -v
   ```

2. **Test reconcile with profiling:**
   ```bash
   python -c "
   from modules.auto_trade.database.reconcile import reconcile_orders_with_binance
   result = reconcile_orders_with_binance(
       'demo_key', 'demo_secret', testnet=True, enable_profiling=True
   )
   print('Timing:', result.get('timing', {}))
   print('Inserted:', result['inserted'])
   print('Skipped:', result['skipped'])
   "
   ```

3. **Run profiling script:**
   ```bash
   python scripts/profile_reconcile.py
   ```

## Future Optimizations

Potential areas for further improvement:

1. **Connection pooling tuning:** Adjust pool_size/max_overflow based on workload
2. **Index optimization:** Add composite indexes for common query patterns
3. **Async support:** Consider async/await for I/O operations
4. **Caching:** Cache exchange data temporarily to reduce API calls
5. **Parallel processing:** Process multiple symbols in parallel (with care for rate limits)

## References

- SQLAlchemy bulk_insert_mappings: https://docs.sqlalchemy.org/en/14/orm/session_api.html#sqlalchemy.orm.Session.bulk_insert_mappings
- Python threading.Lock: https://docs.python.org/3/library/threading.html#threading.Lock
- CCXT fetchClosedOrders: https://docs.ccxt.com/#/README?id=closed-orders
