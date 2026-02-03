# Binance API Test – Quick Reference

Cách nhanh để kiểm tra kết nối Binance API cho module auto_trade.

## Chạy test

```bash
# Từ project root
python modules/auto_trade/test_demo_api.py
```

**Demo:** Trong `.env` đặt `BINANCE_TESTNET=true` và dùng API keys từ Binance Futures → Demo Trading.  
**Real:** `BINANCE_TESTNET=false` và dùng API keys tài khoản thật (cẩn trọng).

## Scripts

| Script | Mục đích |
|--------|----------|
| `test_demo_api.py` | Test đầy đủ: Balance, Positions, Market data (BinanceClient) |
| `test_demo_simple.py` | Test nhanh kết nối CCXT |

## Lỗi thường gặp

- **-2008 Invalid Api-Key ID:** Key hết hạn hoặc sai → tạo key mới.
- **-2015 Invalid API-key, IP, or permissions:** Kiểm tra quyền Futures và IP.
- **-1022 Signature:** Kiểm tra Secret trong `.env` và đồng bộ giờ máy.

Chi tiết lấy key, cấu hình endpoint và xử lý lỗi: **[BINANCE_DEMO_GUIDE.md](BINANCE_DEMO_GUIDE.md)**.

## Bước tiếp theo

1. Test OK → xem [HUONG_DAN_SU_DUNG.md](HUONG_DAN_SU_DUNG.md) để cấu hình hệ thống.
2. Khởi tạo DB: `python modules/auto_trade/main.py --init-db`.
3. Chạy monitoring trước (không giao dịch).

---

**Last updated:** 2026-02
