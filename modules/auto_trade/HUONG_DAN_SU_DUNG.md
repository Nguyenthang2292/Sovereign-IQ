# Hướng Dẫn Sử Dụng Module Auto Trade

**Ngôn ngữ**: Tiếng Việt | [English](USER_GUIDE.md)

**Phiên bản**: 1.0.0
**Cập nhật**: 2026-02-03

---

## 📋 Mục Lục

1. [Yêu Cầu Hệ Thống](#-yêu-cầu-hệ-thống)
2. [Cài Đặt Database SQLite](#-cài-đặt-database-sqlite)
3. [Cấu Hình API Binance Demo](#-cấu-hình-api-binance-demo)
4. [Cấu Hình API Binance Thực](#-cấu-hình-api-binance-thực)
5. [Chạy Hệ Thống](#-chạy-hệ-thống)
6. [Kiểm Tra & Giám Sát](#-kiểm-tra--giám-sát)

---

## 📦 Yêu Cầu Hệ Thống

### Phần Mềm Cần Thiết

```bash
# Python phiên bản 3.9 trở lên
python --version  # Phải >= 3.9

# Cài đặt dependencies
pip install -r requirements.txt
```

### Thư Viện Python Quan Trọng

- `SQLAlchemy` - ORM cho database
- `ccxt` - Kết nối với Binance API
- `python-dotenv` - Quản lý biến môi trường

---

## 🗄️ Cài Đặt Database SQLite

### Bước 1: Khởi Tạo Database

SQLite là database file-based, **không cần chạy server riêng**.

```bash
# Di chuyển vào thư mục auto_trade
cd modules/auto_trade

# Database sẽ tự động được tạo khi chạy lần đầu
python main.py --init-db
```

Hoặc sử dụng Python:

```python
from modules.auto_trade.database import initialize_database

# Khởi tạo database với schema
initialize_database(db_path='data/auto_trade.db')
print("✅ Database đã được khởi tạo thành công!")
```

### Bước 2: Kiểm Tra Database

```python
from modules.auto_trade.database import get_db_manager

# Lấy database manager
db_manager = get_db_manager()

# Kiểm tra kết nối
if db_manager.check_connection():
    print("✅ Database hoạt động bình thường")

    # Xem thống kê
    stats = db_manager.get_database_stats()
    print(f"Tổng số orders: {stats['total_orders']}")
    print(f"Orders đang mở: {stats['open_orders']}")
else:
    print("❌ Không thể kết nối database")
```

### Bước 3: Cấu Hình Database (Tùy Chọn)

Tạo file `.env` trong thư mục `modules/auto_trade/`:

```bash
# Database Configuration
AUTO_TRADE_DB_DIR=data
AUTO_TRADE_DB_NAME=auto_trade.db
AUTO_TRADE_DB_POOL_SIZE=5
AUTO_TRADE_MAX_BACKUPS=30
AUTO_TRADE_BACKUP_COMPRESS=true
```

### Vị Trí Files

```
modules/auto_trade/
├── data/
│   ├── auto_trade.db          # Database chính
│   ├── backups/               # Các bản backup
│   │   ├── auto_trade_backup_20260203_120000.db.gz
│   │   └── ...
│   └── exports/               # Dữ liệu xuất ra
```

---

## 🧪 Cấu Hình API Binance Demo

### Về Demo Account

Binance cung cấp **Demo Trading** (endpoint `demo-fapi.binance.com`). Trong code: dùng `testnet=True` và trong `.env` đặt `BINANCE_TESTNET=true`. Chi tiết từng bước lấy key, cấu hình và xử lý lỗi: **[BINANCE_DEMO_GUIDE.md](BINANCE_DEMO_GUIDE.md)**.

**Đặc điểm Demo Account:**
- ✅ Giống môi trường real trading, tiền ảo
- ✅ Dữ liệu giá real-time, đầy đủ tính năng futures
- ✅ Không rủi ro tài chính
- ⚠️ API keys riêng cho Demo, không dùng cho tài khoản thật

### Bước 1: Lấy Demo API Keys

1. Đăng nhập Binance → **Derivatives** → **USDT-M Futures** → bật **Demo Trading**.
2. Trong chế độ Demo: **Profile** → **API Management** → **Create API**.
3. Bật **Enable Futures** và **Enable Reading**; không bật Withdrawal.
4. Lưu API Key và Secret Key (Secret chỉ hiện một lần).

### Bước 2: Cấu Hình Demo API

Thêm vào file `.env` trong `modules/auto_trade/`:

```bash
# Binance Demo (demo-fapi.binance.com)
BINANCE_API_KEY=your_demo_api_key_here
BINANCE_API_SECRET=your_demo_secret_here
BINANCE_TESTNET=true

# Database Configuration (tùy chọn)
AUTO_TRADE_DB_DIR=data
AUTO_TRADE_DB_NAME=auto_trade.db
AUTO_TRADE_MAX_BACKUPS=30
```

**Quan trọng:** Demo dùng endpoint `demo-fapi.binance.com` → trong code phải dùng `testnet=True` (ví dụ trong `test_demo_api.py`).

### Bước 3: Kiểm Tra Demo API

```bash
python modules/auto_trade/test_demo_api.py
```

Nếu lỗi -2008 (Invalid Api-Key): key hết hạn hoặc sai → tạo key mới trong Demo Trading và cập nhật `.env`. Nếu lỗi -2015: kiểm tra quyền Futures và IP (xem [BINANCE_DEMO_GUIDE.md](BINANCE_DEMO_GUIDE.md)).

### Khi Demo Keys không hoạt động

- **Option 1:** Lấy key Demo mới theo [BINANCE_DEMO_GUIDE.md](BINANCE_DEMO_GUIDE.md).
- **Option 2:** Dùng Real API keys (cẩn trọng) — xem mục **Cấu Hình API Binance Thực** bên dưới.
- **Option 3:** Tạm thời bỏ qua test API, chỉ chạy DB/backend: `python main.py --dry-run --init-db`.

---

## 💰 Cấu Hình API Binance Thực

### ⚠️ CẢNH BÁO QUAN TRỌNG

```
🚨 ĐÂY LÀ TÀI KHOẢN THẬT VỚI TIỀN THẬT!
   - Bắt đầu với số tiền nhỏ
   - Kiểm tra kỹ tất cả cài đặt
   - Đặt stop-loss cho mọi lệnh
   - KHÔNG BAO GIỜ chia sẻ API keys
```

### Trước khi giao dịch thật (bắt buộc đọc)

1. **Bật Dry Run trước:** Khởi tạo `BinanceClient(..., testnet=False, dry_run=True)` để mô phỏng lệnh, không gửi lệnh thật.
2. **Số dư tối thiểu:** Nên có ít nhất $50–100 USDT để test thật; nếu chỉ vài USDT thì chỉ nên dùng dry_run.
3. **Kích thước lệnh:** Bắt đầu với size nhỏ nhất (1 contract hoặc notional tối thiểu).
4. **Stop-loss:** Mọi vị thế phải có stop-loss; không risk quá 1–2% mỗi lệnh.
5. **Kiểm tra logic:** Chạy backtest, test pipeline với dry_run, kiểm tra `order_builder.py`, `signal_selector.py`, `xgboost_filter.py` trước khi tắt dry_run.

**Khi sẵn sàng giao dịch thật:** Nạp đủ vốn → đặt giới hạn position (ví dụ MAX_POSITION_USDT=10, MAX_RISK_PCT=1.0) → đổi `dry_run=False` → theo dõi sát vài lệnh đầu.

**Chuyển lại sang Demo:** Lấy key Demo theo [BINANCE_DEMO_GUIDE.md](BINANCE_DEMO_GUIDE.md), cập nhật `.env` với demo keys và `BINANCE_TESTNET=true`, trong script đặt `testnet=True`.

### Bước 1: Tạo Real API Keys

1. Đăng nhập Binance: https://www.binance.com/
2. Vào **Account** → **API Management**
3. Tạo API Key mới:
   - Đặt tên: `Auto Trading Bot`
   - Chọn quyền:
     - ✅ **Enable Reading** (Bắt buộc)
     - ✅ **Enable Futures** (Bắt buộc cho futures trading)
     - ⚠️ **Enable Spot & Margin Trading** (Tùy chọn)
     - ❌ **KHÔNG CHỌN "Enable Withdrawals"** (Bảo mật)
4. Xác thực 2FA
5. Lưu lại API Key và Secret Key **ngay lập tức**

### Bước 2: Bảo Mật API Keys

**KHÔNG lưu trực tiếp trong config.json!**

Tạo file `.env` (file này **KHÔNG** được commit vào Git):

```bash
# .env
BINANCE_API_KEY=your_real_api_key_here
BINANCE_API_SECRET=your_real_api_secret_here
```

Thêm `.env` vào `.gitignore`:

```bash
echo ".env" >> .gitignore
```

### Bước 3: Cấu Hình Production

Chỉnh sửa `config.json`:

```json
{
  "exchange": "binance",
  "mode": "production",
  "testnet": false,
  "api": {
    "production": {
      "api_key": "${BINANCE_API_KEY}",
      "api_secret": "${BINANCE_API_SECRET}",
      "futures_api_url": "https://fapi.binance.com"
    }
  },
  "trading": {
    "enabled": false,  // Bắt đầu với FALSE để giám sát
    "max_positions": 2,
    "leverage": 2,
    "risk_per_trade": 0.01,  // 1% risk mỗi lệnh
    "max_daily_loss": 0.05,  // Dừng nếu loss 5%/ngày
    "position_size_usdt": 50  // Bắt đầu với $50
  },
  "risk_management": {
    "stop_loss_percentage": 2.0,
    "take_profit_percentage": 4.0,
    "trailing_stop": true,
    "break_even_after": 1.5
  }
}
```

### Bước 4: Kiểm Tra Real API

```python
import os
from dotenv import load_dotenv
from modules.auto_trade.core.exchange_client import ExchangeClient

# Load environment variables
load_dotenv()

# Kết nối production
client = ExchangeClient(
    api_key=os.getenv('BINANCE_API_KEY'),
    api_secret=os.getenv('BINANCE_API_SECRET'),
    testnet=False  # Production
)

# Kiểm tra quyền API
try:
    # Kiểm tra quyền đọc
    account = client.get_account_info()
    print("✅ API Key có quyền đọc")

    # Kiểm tra balance
    balance = client.get_futures_balance()
    total_usdt = balance['USDT']['free'] + balance['USDT']['locked']
    print(f"💰 Tổng USDT: {total_usdt:.2f}")

    # Kiểm tra quyền futures
    positions = client.get_futures_positions()
    print(f"✅ API Key có quyền futures")
    print(f"📊 Vị thế đang mở: {len(positions)}")

except Exception as e:
    print(f"❌ Lỗi: {e}")
    print("Kiểm tra lại API permissions")
```

### Bước 5: Whitelist IP (Khuyến Nghị)

Để tăng bảo mật:

1. Vào Binance **API Management**
2. Chọn API Key vừa tạo
3. Click **Edit restrictions**
4. Thêm IP address của server:
   ```
   ✅ Restrict access to trusted IPs only
   IP: YOUR_SERVER_IP
   ```

---

## 🚀 Chạy Hệ Thống

### Chế Độ Giám Sát (Monitoring Mode)

**Khuyến nghị bắt đầu với chế độ này!**

```bash
# Chỉ giám sát, KHÔNG giao dịch
python modules/auto_trade/main.py \
  --mode monitoring \
  --config config.json
```

Hoặc trong Python:

```python
from modules.auto_trade.main import AutoTradeSystem

# Khởi tạo hệ thống
system = AutoTradeSystem(
    config_path='modules/auto_trade/config.json',
    trading_enabled=False  # Chỉ giám sát
)

# Chạy monitoring
system.run_monitoring()
```

### Chế Độ Backtest

Test chiến lược với dữ liệu lịch sử:

```bash
python modules/auto_trade/main.py \
  --mode backtest \
  --symbol BTCUSDT \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

### Chế Độ Paper Trading (Demo Account)

Trading với tiền ảo trên demo account:

```bash
# Sử dụng demo account
python modules/auto_trade/main.py \
  --mode live \
  --config config.json
```

Đặt trong `.env`:

```bash
BINANCE_API_KEY=YhCgXF4wzDTx7fpXLx72rMt3P18SI7Ai1sUD0CCIVkjTCfG8Nka18BvAL0i2AyWo
BINANCE_API_SECRET=kpk6SFB6q47oxxf0AXgcpTWFHCKoyA0xFUie7C3sMLCu9MO0nOqn6GyaJlzXOZ6N
BINANCE_TESTNET=false
DRY_RUN=true  # Bật chế độ dry-run để an toàn hơn
```

### Chế Độ Live Trading (Production)

**⚠️ CHỈ SAU KHI ĐÃ TEST KỸ!**

```bash
# CẢNH BÁO: ĐÂY LÀ TIỀN THẬT!
python modules/auto_trade/main.py \
  --mode live \
  --config config.json
```

Đặt trong config:

```json
{
  "testnet": false,
  "trading": {
    "enabled": true  // BẬT trading thực
  }
}
```

### Chạy Như Service (Background)

```bash
# Linux/Mac - Sử dụng nohup
nohup python modules/auto_trade/main.py --mode live > auto_trade.log 2>&1 &

# Windows - Sử dụng pythonw
start /B pythonw modules/auto_trade/main.py --mode live

# Hoặc sử dụng screen (Linux)
screen -S auto_trade
python modules/auto_trade/main.py --mode live
# Ctrl+A, D để detach
```

---

## 📊 Kiểm Tra & Giám Sát

### 1. Kiểm Tra Database

```python
from modules.auto_trade.database import get_db_manager, get_overall_stats, session_scope

db_manager = get_db_manager()

# Thống kê tổng quan
with session_scope() as session:
    stats = get_overall_stats(session)
    print(f"📊 Thống kê giao dịch")
    print(f"   Tổng lệnh: {stats['total_trades']}")
    print(f"   Thắng: {stats['winning_trades']}")
    print(f"   Tỷ lệ thắng: {stats['win_rate']:.1f}%")
    print(f"   P&L tổng: ${stats['total_pnl']:.2f}")
```

### 2. Xem Vị Thế Hiện Tại

```python
from modules.auto_trade.database import get_open_positions, session_scope

with session_scope() as session:
    positions = get_open_positions(session)

    if positions:
        print(f"📈 {len(positions)} vị thế đang mở:")
        for pos in positions:
            pnl_color = "🟢" if pos.unrealized_pnl > 0 else "🔴"
            print(f"   {pnl_color} {pos.symbol}: {pos.side}")
            print(f"      Entry: ${pos.entry_price:.2f}")
            print(f"      P&L: ${pos.unrealized_pnl:.2f}")
    else:
        print("✅ Không có vị thế nào đang mở")
```

### 3. Xem Logs

```bash
# Xem logs realtime
tail -f logs/auto_trade.log

# Xem logs lỗi
grep ERROR logs/auto_trade.log

# Xem 100 dòng cuối
tail -n 100 logs/auto_trade.log
```

### 4. Giám Sát Performance

```python
from modules.auto_trade.database import get_daily_stats, session_scope

with session_scope() as session:
    daily_stats = get_daily_stats(session, days=7)

    print("📅 Thống kê 7 ngày gần nhất:")
    for day in daily_stats:
        print(f"\n{day['date']}:")
        print(f"   Lệnh: {day['total_trades']}")
        print(f"   Win rate: {day['win_rate']:.1f}%")
        print(f"   P&L: ${day['total_pnl']:.2f}")
```

### 5. Backup Database

```python
from modules.auto_trade.database import create_database_backup

# Tạo backup
backup_path = create_database_backup(compress=True)
print(f"✅ Backup đã lưu tại: {backup_path}")

# Backup tự động (chạy mỗi ngày)
from modules.auto_trade.database import BackupScheduler, BackupManager

backup_mgr = BackupManager('data/auto_trade.db')
scheduler = BackupScheduler(backup_mgr)

# Kiểm tra và backup nếu cần
if scheduler.should_backup(interval_hours=24):
    scheduler.run_if_needed()
    print("✅ Auto backup completed")
```

---

## 🛑 Dừng Hệ Thống

### Dừng An Toàn

```bash
# Gửi tín hiệu SIGTERM để hệ thống dừng an toàn
kill -TERM <process_id>

# Hoặc sử dụng Ctrl+C nếu chạy foreground
```

### Dừng Khẩn Cấp

```python
from modules.auto_trade.main import AutoTradeSystem

system = AutoTradeSystem.get_instance()

# Đóng tất cả vị thế
system.close_all_positions(reason="Emergency stop")

# Dừng hệ thống
system.shutdown()
```

---

## ⚠️ Lưu Ý Quan Trọng

### Bảo Mật

1. **KHÔNG BAO GIỜ** commit API keys vào Git
2. **LUÔN** sử dụng file `.env` cho sensitive data
3. **BẬT** 2FA cho tài khoản Binance
4. **WHITELIST** IP nếu có thể
5. **TRÁNH** cấp quyền withdrawal cho API keys

### Risk Management

1. **BẮT ĐẦU NHỎ**: Test với $50-100 USDT
2. **ĐẶT STOP-LOSS**: Luôn có stop-loss cho mọi lệnh
3. **GIỚI HẠN LEVERAGE**: Bắt đầu với leverage 2x-3x
4. **DAILY LOSS LIMIT**: Dừng nếu loss 5% trong ngày
5. **GIÁM SÁT THƯỜNG XUYÊN**: Kiểm tra ít nhất 2 lần/ngày

### Monitoring Best Practices

1. **Chạy monitoring mode 1 tuần** trước khi enable trading
2. **Backtest 3-6 tháng** data trước khi live
3. **Paper trade 2 tuần** trên testnet
4. **Bắt đầu với 1-2 cặp** trading (BTCUSDT, ETHUSDT)
5. **Tăng dần position size** sau khi có kết quả tốt

---

## 🆘 Xử Lý Lỗi Thường Gặp

### Lỗi 1: Database không khởi tạo được

```bash
# Xóa database cũ và tạo lại
rm data/auto_trade.db
python main.py --init-db
```

### Lỗi 2: API Key không hợp lệ

```python
# Kiểm tra lại API key
print(f"API Key: {os.getenv('BINANCE_API_KEY')[:10]}...")
print(f"Testnet: {config['testnet']}")

# Đảm bảo sử dụng đúng API key (testnet vs production)
```

### Lỗi 3: Insufficient balance

```python
# Kiểm tra balance
balance = client.get_futures_balance()
print(f"Free USDT: {balance['USDT']['free']}")

# Giảm position size trong config
```

### Lỗi 4: Database locked

```bash
# Đóng tất cả connections
pkill -f "python.*auto_trade"

# Hoặc restart database connection
```

---

## 📞 Hỗ Trợ

### Documentation

- **Module README**: `modules/auto_trade/README.md`
- **Database Guide**: `modules/auto_trade/database/README.md`
- **Improvements Log**: `modules/auto_trade/database/IMPROVEMENTS.md`

### Logs

- **Application logs**: `logs/auto_trade.log`
- **Error logs**: `logs/error.log`
- **Database logs**: `logs/database.log`

### Testing

```bash
# Chạy test suite
pytest modules/auto_trade/tests/

# Test cụ thể
pytest modules/auto_trade/tests/test_database.py
pytest modules/auto_trade/tests/test_execution.py
```

---

## ✅ Checklist Trước Khi Live Trading

- [ ] Database đã khởi tạo thành công
- [ ] Demo API keys đã test hoạt động
- [ ] Backtest cho kết quả tích cực (>55% win rate)
- [ ] Demo trading 2 tuần không lỗi
- [ ] Stop-loss và take-profit đã cấu hình
- [ ] Risk per trade <= 2%
- [ ] Daily loss limit đã đặt
- [ ] Backup tự động đã bật
- [ ] Monitoring đã test ít nhất 1 tuần
- [ ] IP đã whitelist (nếu có)
- [ ] 2FA đã bật cho Binance account
- [ ] API không có quyền withdrawal
- [ ] Đã đọc kỹ tất cả cảnh báo

---

**🎯 Khuyến Nghị**: Bắt đầu với monitoring mode → demo account → paper trade → live với số tiền nhỏ

**⚠️ Disclaimer**: Crypto trading có rủi ro cao. Chỉ trade với số tiền bạn có thể chấp nhận mất.

---

**Tạo bởi**: Auto Trade Development Team
**Phiên bản**: 1.0.0
**Cập nhật**: 2026-02-03
