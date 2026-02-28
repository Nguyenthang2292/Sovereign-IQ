# Auto-Close Timer — Design Document

**Date:** 2026-02-28  
**Status:** Approved  
**Author:** Brainstorming session

---

## 1. Tổng quan

Thêm cơ chế hẹn giờ đóng lệnh tự động vào hệ thống Auto-Trade. Tính năng hoạt động theo đúng pattern của các job hiện có (`TrailingStopJob`, `NegativeBreakevenJob`, `EnsureTPSLJob`): polling mỗi 30 giây, đọc open positions từ DynamoDB, thực thi qua `BinanceClient`.

**Hai trigger độc lập:**
- **Trigger A — Time-in-trade timeout:** Lệnh mở quá X giờ → tự đóng
- **Trigger B — Daily close window:** Tất cả lệnh đóng vào khung giờ cố định UTC mỗi ngày

**Cơ chế đóng:** Hủy TP hiện tại → đặt TP mới sát giá thị trường (quasi-market, không dùng market order trực tiếp).

---

## 2. Kiến trúc

### 2.1 File mới

```
modules/auto_trade/execution/
  auto_close_timer_job.py       ← NEW: AutoCloseTimerJob (core logic)
  auto_close_timer.py           ← NEW: AutoCloseLogic (helper calculations)

modules/auto_trade/gui/components/
  scheduled_exits_panel.py      ← NEW: Panel "Scheduled Exits" (tab mới)

modules/auto_trade/gui/components/config_panel_parts/
  auto_close_settings.py        ← NEW: Settings UI cho auto_close section
```

### 2.2 File sửa đổi

```
modules/auto_trade/settings.yaml                     ← Thêm section auto_close
modules/auto_trade/gui/main_window/main_window.py    ← Thêm tab Scheduled Exits
modules/auto_trade/gui/main_window/lifecycle_actions_mixin.py  ← Khởi động/dừng job
modules/auto_trade/gui/components/config_panel.py    ← Link tới auto_close_settings
```

### 2.3 Luồng hoạt động

```
QTimer (30s)
  └── AutoCloseTimerJob.run()
        ├── Kiểm tra: auto_close.enabled == True?
        ├── Lấy tất cả open positions từ DynamoDB
        └── Với mỗi lệnh:
              ├── Bỏ qua nếu auto_close_triggered == True (idempotent)
              ├── Bỏ qua nếu tuổi lệnh < grace_period_minutes
              ├── [Trigger A] Tính thời gian mở lệnh → so với deadline
              │     deadline = auto_close_deadline_utc (per-order override)
              │              hoặc opened_at + max_duration_hours (global)
              ├── [Trigger B] Kiểm tra daily_close_time có khớp không
              └── Nếu trigger → AutoCloseLogic.execute_close(order)
```

---

## 3. Logic đóng lệnh

```python
def execute_close(order, binance_client, tp_offset_pct=0.05):
    symbol = order["symbol"]                        # e.g. "BTCUSDT"
    side   = order["side"]                          # "LONG" | "SHORT"

    # 1. Lấy giá hiện tại
    current_price = binance_client.get_mark_price(symbol)

    # 2. Tính TP sát giá (quasi-market)
    if side == "LONG":
        new_tp = current_price * (1 - tp_offset_pct / 100)
    else:
        new_tp = current_price * (1 + tp_offset_pct / 100)

    # 3. Hủy TP cũ
    binance_client.cancel_order(symbol, order["tp_order_id"])

    # 4. Đặt TP mới
    binance_client.place_tp_order(symbol, side, new_tp, order["quantity"])

    # 5. Ghi DynamoDB
    update_order(order["id"], {
        "auto_close_triggered": True,
        "auto_close_reason": reason,          # "max_duration" | "daily_close" | "manual"
        "auto_close_triggered_at": utcnow(),
    })
```

**Độ ưu tiên khi xung đột:**
- Nếu cả hai trigger fire cùng lúc → chỉ thực thi một lần (flag `auto_close_triggered`)
- `auto_close_deadline_utc` (per-order) override hoàn toàn `max_duration_hours` (global)
- Lệnh tuổi < `grace_period_minutes` → bỏ qua dù daily_close đang active

---

## 4. Config — `settings.yaml`

```yaml
auto_close:
  enabled: false

  # Trigger A: time-in-trade timeout
  max_duration_enabled: true
  max_duration_hours: 4.0

  # Trigger B: daily close window
  daily_close_enabled: true
  daily_close_time: "22:00"     # UTC, format HH:MM
  daily_close_days: "1234567"   # 1=Mon ... 7=Sun, mặc định mỗi ngày

  # Safety
  grace_period_minutes: 5       # bỏ qua lệnh mới mở < N phút
  tp_offset_pct: 0.05           # % offset để đặt TP quasi-market
```

---

## 5. DynamoDB — Per-order fields

Tất cả fields đều **optional** (schema-less, không cần migration):

| Field | Type | Ghi bởi | Mô tả |
|---|---|---|---|
| `auto_close_deadline_utc` | ISO8601 string | GUI / user | Override global max_duration cho lệnh này |
| `auto_close_triggered` | bool | Job | Cờ idempotent, tránh re-trigger |
| `auto_close_reason` | string | Job | `"max_duration"` \| `"daily_close"` \| `"manual"` |
| `auto_close_triggered_at` | ISO8601 string | Job | Timestamp khi job fire |

**Logic đọc deadline trong job:**
```python
# Per-order override có priority cao hơn global
deadline = order.get("auto_close_deadline_utc")
if not deadline and max_duration_enabled:
    deadline = opened_at + timedelta(hours=max_duration_hours)
```

---

## 6. GUI — Tab "Scheduled Exits"

### 6.1 Bố cục

```
┌──────────────────────────────────────────────────────────┐
│  [✓] Auto-Close Enabled          [Open Settings]         │  ← Global toggle
├───────────────────────────┬──────────────────────────────┤
│  PENDING EXITS            │  CLOSE HISTORY               │
│  ┌──────────────────────┐ │  ┌───────────────────────┐   │
│  │Sym │Trigger│Countdown│ │  │Sym │Reason  │P&L% │Time│   │
│  │BTC │timer  │ 2h 14m  │ │  │ETH │daily   │-2.1%│... │   │
│  │ETH │daily  │ 0h 47m  │ │  │BTC │timer   │+1.3%│... │   │
│  └──────────────────────┘ │  └───────────────────────┘   │
│  [Override Deadline] [Cancel Auto-Close]                  │
└───────────────────────────┴──────────────────────────────┘
```

### 6.2 Pending Exits — Cột

| Cột | Mô tả |
|---|---|
| Symbol | Tên coin |
| Side | LONG / SHORT |
| Entry Price | Giá vào lệnh |
| Current P&L% | P&L thời gian thực |
| Trigger | `timer` hoặc `daily` |
| Deadline UTC | Thời điểm sẽ đóng |
| **Countdown** | `2h 14m 32s` — cập nhật mỗi 1 giây (QTimer local) |

### 6.3 Actions per-row

- **[Override Deadline]** — Mở datetime picker, ghi `auto_close_deadline_utc` vào DynamoDB
- **[Cancel Auto-Close]** — Xóa `auto_close_deadline_utc`, đặt flag tạm ngăn job fire

### 6.4 Close History — Cột

| Cột | Mô tả |
|---|---|
| Symbol | Tên coin |
| Reason | `max_duration` / `daily_close` / `manual` |
| P&L% | Kết quả khi đóng |
| Triggered At | Thời điểm job fire |

---

## 7. Danh sách tasks implementation

### Phase 1 — Backend (không phụ thuộc GUI)
- [x] Tạo `auto_close_timer.py` — `AutoCloseLogic.execute_close()`
- [x] Tạo `auto_close_timer_job.py` — `AutoCloseTimerJob` theo pattern `TrailingStopJob`
- [x] Cập nhật `settings.yaml` — thêm section `auto_close`
- [x] Unit tests cho logic trigger A + trigger B + idempotency + grace period

### Phase 2 — Integration
- [x] Đăng ký job trong `lifecycle_actions_mixin.py` (start/stop cùng các job khác)
- [x] Đọc config từ `settings_manager.get("auto_close", {})`

### Phase 3 — GUI
- [x] Tạo `auto_close_settings.py` — settings UI (toggle, fields config)
- [x] Tạo `scheduled_exits_panel.py` — panel với Pending + History
- [x] Thêm tab "Scheduled Exits" vào `main_window.py`
- [x] Countdown QTimer (1s) trong Pending table

---

## 8. Quyết định đã chốt

| Vấn đề | Quyết định | Lý do |
|---|---|---|
| Cơ chế đóng | TP quasi-market (offset 0.05%) | Tránh slippage market order, vẫn fill nhanh |
| Config storage | `settings.yaml` + DynamoDB per-order | Nhất quán với pattern hiện tại |
| Trigger đồng thời | Idempotent flag, chỉ fire 1 lần | Tránh double execution |
| Grace period | 5 phút (configurable) | Tránh đóng lệnh vừa mở khi daily_close fire |
| Per-order override | `auto_close_deadline_utc` field | Schema-less, không breaking |
