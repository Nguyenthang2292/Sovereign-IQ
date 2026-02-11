# Position Sync Guide

## Tổng quan

Chức năng **Manual Position Sync** cho phép bạn đồng bộ các positions đang mở trên Binance vào database local của AutoTrade system. Điều này rất hữu ích khi:

- ✅ Positions được mở thủ công trên Binance (không qua AutoTrade)
- ✅ Database bị clear/reset nhưng positions vẫn còn trên Binance
- ✅ Bị miss-sync do lỗi kết nối hoặc lỗi hệ thống
- ✅ Muốn import existing positions vào tracking system

## Cách sử dụng trong GUI

### Phương pháp 1: Sync từ GUI (Khuyến nghị)

1. Mở AutoTrade Dashboard:
   ```bash
   python run_auto_trade_gui.py
   ```

2. Trong tab **Dashboard**, tìm panel "Open Positions"

3. Click button **"🔄 Sync from Binance"** (màu xanh lá)

4. Đợi dialog hiển thị kết quả:
   ```
   ✅ Sync completed!
   
   Found: 1 positions
   Synced: 1 new
   Existing: 0 already in DB
   Failed: 0
   ```

5. Positions sẽ tự động refresh và hiển thị với đầy đủ thông tin TP/SL/BE

### Phương pháp 2: Sync từ command line

Nếu không muốn mở GUI, bạn có thể chạy script test:

```bash
python test_position_sync_manual.py
```

Script này sẽ:
- Kết nối tới Binance API
- Fetch tất cả open positions
- Sync vào database
- Hiển thị thống kê chi tiết

## Cách hoạt động

### 1. Fetch Positions từ Binance
Service sẽ gọi Binance Futures API để lấy:
- Position size (contracts)
- Entry price
- Leverage
- Side (LONG/SHORT)

### 2. Fetch TP/SL từ Open Orders
Với mỗi position, service sẽ:
- Scan các open orders của symbol đó
- Detect TAKE_PROFIT orders → TP price
- Detect STOP_MARKET orders → SL price

### 3. Sync vào Database
Tạo Order record mới với:
- `order_source`: "MANUAL"
- `execution_mode`: "MANUAL"
- `status`: "OPEN"
- Full TP/SL/BE tracking

### 4. Duplicate Detection
Nếu position đã tồn tại trong DB (cùng symbol + OPEN status):
- Không tạo duplicate
- Báo cáo "already exists"

## Các trường hợp sử dụng

### Case 1: Position mở thủ công trên Binance

**Tình huống**: Bạn đã mở position SKL/USDT trực tiếp trên Binance app.

**Giải pháp**:
1. Set TP/SL cho position đó trên Binance (nếu chưa có)
2. Click "Sync from Binance" trong AutoTrade GUI
3. Position sẽ được import với đầy đủ TP/SL info

### Case 2: Database bị reset

**Tình huống**: Database bị xóa hoặc corrupted, nhưng positions vẫn còn trên Binance.

**Giải pháp**:
1. Chạy `python init_database.py` để tạo lại database
2. Chạy `python test_position_sync_manual.py` để sync tất cả positions
3. Mở GUI để verify

### Case 3: Miss-sync do lỗi

**Tình huống**: AutoTrade place order nhưng không track được do network error.

**Giải pháp**:
1. Click "Sync from Binance" để re-fetch
2. System sẽ auto-detect và sync missing positions

## Technical Details

### Service Architecture

```
PositionSyncService
├── fetch_binance_positions()     # Get all open positions
│   └── _fetch_tp_sl_orders()     # Get TP/SL from orders
├── sync_position_to_db()         # Sync single position
└── sync_all_positions()          # Bulk sync with stats
```

### Database Schema

Synced positions được lưu với các fields:
```python
Order(
    order_id="SYNC_<timestamp>",          # Synthetic ID
    client_order_id="SYNC_SYMBOL_<ts>",   # Unique identifier
    symbol="SKLUSDT",                     # Normalized symbol
    side="LONG",                          # Position side
    order_source="MANUAL",                # Mark as manual
    execution_mode="MANUAL",              # Not auto-traded
    status="OPEN",                        # Active position
    take_profit=0.007,                    # From Binance orders
    stop_loss=0.0033,                     # From Binance orders
    ...
)
```

### Error Handling

Service xử lý các errors:
- ❌ **No API credentials**: Báo lỗi "Exchange manager unavailable"
- ❌ **Network timeout**: Retry logic built-in
- ❌ **Invalid response**: Skip position và log warning
- ❌ **Database error**: Rollback transaction và báo lỗi

## Troubleshooting

### Sync button không hoạt động

**Nguyên nhân**: Exchange manager không được khởi tạo.

**Giải pháp**:
1. Kiểm tra API credentials trong Settings tab
2. Đảm bảo mode không phải DRY_RUN
3. Check logs: `logs/auto_trade_gui.log`

### Không fetch được TP/SL

**Nguyên nhân**: Position không có TP/SL orders trên Binance.

**Giải pháp**:
1. Mở Binance Futures app
2. Set TP/SL cho position đó
3. Re-sync lại

### Position bị duplicate

**Nguyên nhân**: Đã sync trước đó nhưng chưa refresh UI.

**Giải pháp**:
- Click "🔄 Refresh" để reload positions
- Duplicate detection sẽ prevent multiple syncs

## Logs

Tất cả sync operations được log:

```
[PositionSync] Starting manual position sync...
[PositionSync] Fetched 1 open positions from Binance
[PositionSync] ✅ Synced SKLUSDT to DB (ID=123)
[PositionSync] Sync completed: 1 synced, 0 existing, 0 failed
```

Check logs tại: `logs/auto_trade_gui.log`

## API Reference

### PositionSyncService.sync_all_positions()

```python
stats = PositionSyncService.sync_all_positions(client, db_manager)

# Returns:
{
    "fetched": 1,      # Total positions found on Binance
    "synced": 1,       # New positions synced to DB
    "existing": 0,     # Positions already in DB
    "failed": 0        # Sync failures
}
```

### GUI Callback: on_sync_positions()

```python
# Called when "Sync from Binance" button clicked
# Runs in background thread
# Shows messagebox with results
# Auto-refreshes positions display
```

## Best Practices

1. **Sync trước khi trade**: Luôn sync positions khi bắt đầu session mới
2. **Set TP/SL trên Binance**: Đảm bảo có TP/SL orders để sync được đầy đủ
3. **Regular sync**: Sync định kỳ nếu hay trade thủ công
4. **Check logs**: Monitor logs để catch errors sớm

## Security Notes

- ❗ API keys cần quyền "Read" để fetch positions
- ❗ API keys cần quyền "Read" để fetch open orders
- ✅ Không cần quyền "Trade" cho sync operation
- ✅ Credentials stored securely trong `.env` file

## Support

Nếu gặp vấn đề:
1. Check logs: `logs/auto_trade_gui.log`
2. Run test script: `python test_position_sync_manual.py`
3. Verify API permissions trên Binance
4. Report issue với full error logs

---

**Created**: 2026-02-11  
**Version**: 1.0.0  
**Maintainer**: AutoTrade Team
