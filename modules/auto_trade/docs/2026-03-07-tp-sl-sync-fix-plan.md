# Fix TP/SL Order Sync And Race Condition

## Goal

Fix race conditions and sync errors that leave orphan TP/SL conditional orders open on Binance after a position has been closed either manually or by a TP/SL fill.

## Tasks

- [x] Task 1: Initialize per-symbol threading locks (`self._close_locks`) and implement a unified `_cancel_and_close_position` method in `websocket_data_service.py` → Verify: The new method accepts symbol, PnL, exit price, and source, and operates idempotently.
- [x] Task 2: Refactor `_handle_order_update` and `_handle_position_update` in `websocket_data_service.py` to use the shared `_cancel_and_close_position` within the thread lock → Verify: Both Path A (order fill) and Path B (position size 0) call this method correctly under a lock.
- [x] Task 3: Improve `POSITION_CLOSED` event publishing in `WebSocketDataService` to fallback to `order_id` when `client_order_id` is generic or missing, preventing memory leaks and false negatives → Verify: Events publish correctly regardless of missing client IDs.
- [x] Task 4: Fix DB record lookup in `EnsureTPSLJob._cleanup_closed_position` (`ensure_tp_sl_job.py`) by checking for multiple possible primary key formats (e.g., `order_id`, `pk`, `PK`, `id`) → Verify: The job successfully locates and marks closed records in DynamoDB.
- [x] Task 5: Use CCXT futures symbol format in `PositionSyncService.sync_all_positions()` (`position_sync_service.py`) prior to calling `cancel_all_orders()` → Verify: The exchange client receives the properly formatted symbol (e.g., `BTC/USDT:USDT`).

## Done When

- [x] `_handle_order_update` and `_handle_position_update` execution logic is decoupled from `db_orders` states to prevent race conditions.
- [x] Orphan TP/SL conditional orders are strictly cancelled when positions hit size zero.
- [x] DB sync updates and positions' CLOSED events fire reliably and perfectly reflect Binance states.