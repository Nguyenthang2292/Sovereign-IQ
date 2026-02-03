# 📋 Phase 4: Position Management - Detailed Tasks

## 🎯 Mục Tiêu Phase 4

Nâng cấp khả năng quản lý vị thế: xem chi tiết, đóng vị thế (toàn bộ/partial), điều chỉnh TP/SL, và các hộp thoại xác nhận an toàn.

## 📌 Prerequisites

- ✅ Phase 1 đã hoàn thành (Dashboard Visualization)
- ✅ Phase 2 đã hoàn thành (Executions)
- ✅ Phase 3 đã hoàn thành (Config & Scanner Control)
- ✅ ExchangeManager hỗ trợ modify orders

---

## 🔍 I. POSITION DETAILS VIEW

### 1.1 Create Position Details Modal/Frame

- [x] **Task 1.1.1:** Tạo `gui/components/position_details.py`
- [x] **Task 1.1.2:** Thiết kế layout hiển thị thông tin chi tiết
- [x] **Task 1.1.3:** Hiển thị Entry Price vs Mark Price trực quan
- [x] **Task 1.1.4:** Tính toán và hiển thị khoảng cách đến Liquidation Price
- [x] **Task 1.1.5:** Real-time P&L display (Unrealized & Realized)

### 1.2 Visualizations

- [x] **Task 1.2.1:** Mini-chart hoặc Progress Bar cho P&L
- [x] **Task 1.2.2:** Visual representation của TP/SL relative to Entry
- [x] **Task 1.2.3:** Margin Level indicator

---

## ⚡ II. POSITION ACTIONS

### 2.1 Basic Actions

- [x] **Task 2.1.1:** Tạo module `gui/components/position_actions.py`
- [x] **Task 2.1.2:** Implement "Close Position" (Market) button
- [x] **Task 2.1.3:** Implement "Close Position" (Limit) inputs
- [x] **Task 2.1.4:** Implement "Cancel Open Orders" (liên quan đến vị thế)

### 2.2 Advanced Actions

- [x] **Task 2.2.1:** Implement "Partial Close" slider/input (25%, 50%, 75%)
- [x] **Task 2.2.2:** Implement "Modify TP/SL" interface
  - [x] Input giá mới
  - [x] Input theo % ROI, % Price Distance
- [x] **Task 2.2.3:** Chức năng "Breakeven" (Dời SL về Entry)
- [x] **Task 2.2.4:** (Optional) "Add Margin" functionality (cho Isolated mode)

---

## 🛡️ III. SAFETY & CONFIRMATION

### 3.1 Close Confirmation Dialog

- [x] **Task 3.1.1:** Tạo `gui/dialogs/close_confirmation.py`
- [x] **Task 3.1.2:** Hiển thị tóm tắt: Estimated P&L, Fees, Final Return
- [x] **Task 3.1.3:** Yêu cầu xác nhận (Button nhấn 2 lần hoặc Hold)
- [x] **Task 3.1.4:** Tùy chọn "Don't ask again" (lưu vào settings Phase 3)

### 3.2 Modification Safety

- [x] **Task 3.2.1:** Validate TP/SL mới (TP > Entry > SL cho Long, ngược lại cho Short)
- [x] **Task 3.2.2:** Cảnh báo nếu SL quá gần giá hiện tại
- [x] **Task 3.2.3:** Validate size khi Partial Close

---

## 🔗 IV. INTEGRATION (BACKEND)

### 4.1 Exchange Integration

- [x] **Task 4.1.1:** Update `BinanceClient` để hỗ trợ partial close calls
- [x] **Task 4.1.2:** Đảm bảo `modify_tp_sl` hoạt động chính xác cho TP/SL
- [x] **Task 4.1.3:** Handle execution events/errors từ sàn

### 4.2 GUI Integration

- [x] **Task 4.2.1:** Gắn sự kiện click vào thẻ Position ở Dashboard (Phase 1) để mở Details
- [x] **Task 4.2.2:** Context menu (chuột phải) trên Position card cho thao tác nhanh
- [ ] **Task 4.2.3:** Update UI ngay lập tức sau khi hành động thành công (Optimistic UI update)

---

## ✅ V. TESTING

### 5.1 Details Testing

- [x] Test hiển thị dữ liệu với vị thế Long/Short
- [x] Test hiển thị Liquidation warning khi gần chạm mức
- [x] Test update real-time khi gía chạy

### 5.2 Actions Testing

- [x] **Critical:** Test Market Close (Full) trên tài khoản Demo/Testnet
- [x] **Critical:** Test Partial Close (kiểm tra số dư còn lại)
- [x] **Critical:** Test Modify TP/SL (kiểm tra orders trên sàn thay đổi)
- [x] Test hủy lệnh treo

### 5.3 UX Testing

- [x] Test flows của Confirmation Dialog
- [x] Test error handling khi sàn từ chối lệnh (e.g., Insufficient balance, API error)

---

## 📦 VI. DELIVERABLES

### 6.1 Code

- [x] `gui/components/position_details.py`
- [x] `gui/components/position_actions.py`
- [x] `gui/dialogs/close_confirmation.py`
- [x] Updates cho `modules/auto_trade/execution/binance_client.py` (để hỗ trợ close, partial close, modify_tp_sl, cancel_orders)
- [x] Updates cho `gui/components/positions_frame.py` (để link sang details)

### 6.2 Features

- [x] Detailed view modal
- [x] Partial close capability
- [x] TP/SL modification UI
- [x] Safety confirmations

---

## 🎯 SUCCESS CRITERIA

Phase 4 hoàn thành khi:

1. ✅ Có thể click vào bất kỳ vị thế nào để xem chi tiết.
2. ✅ Thực hiện được Partial Close chính xác (ví dụ đóng 50% volume).
3. ✅ Thay đổi được TP/SL trực tiếp từ giao diện mà không cần vào sàn.
4. ✅ Có xác nhận an toàn trước khi đóng lệnh.
5. ✅ Không có lỗi crash khi thao tác sai (nhập sai số, lỗi mạng).

**Estimated Time:** 2-3 days
**Priority:** MEDIUM
**Dependencies:** Phase 1 (View), Phase 2 (Execution Core)

---

## ✅ PHASE 4 HOÀN THÀNH

Tất cả tasks chính trong Phase 4 đã được hoàn thành thành công! Vị thế quản lý đã sẵn sàng để sử dụng.

### Files đã tạo/cập nhật

- `gui/components/position_details.py` - Modal hiển thị chi tiết vị thế
- `gui/components/position_actions.py` - Các hành động vị thế (đóng, đóng một phần, chỉnh TP/SL)
- `gui/dialogs/close_confirmation.py` - Dialog xác nhận an toàn
- `gui/dialogs/__init__.py` - Module init cho dialogs
- `modules/auto_trade/execution/binance_client.py` - Thêm methods: close_position(), modify_take_profit(), modify_stop_loss(), modify_tp_sl(), cancel_open_orders()

### Tính năng đã hoàn thành

- ✅ Position Details Modal với đầy đủ thông tin
- ✅ P&L display real-time
- ✅ TP/SL visualization
- ✅ Liquidation distance calculation
- ✅ Close Position (Market & Limit)
- ✅ Partial Close với % lựa chọn
- ✅ Modify TP/SL interface
- ✅ Breakeven chức năng
- ✅ Cancel Open Orders
- ✅ TP/SL Validation
- ✅ Confirmation Dialog với multi-confirm
- ✅ Estimated P&L, Fees, Final Return
- ✅ Backend support cho position management

### Tasks còn lại (GUI Integration)

- [x] Task 4.2.1: Gắn sự kiện click vào thẻ Position để mở Details
- [x] Task 4.2.2: Context menu (chuột phải) trên Position card
- [ ] Task 4.2.3: Optimistic UI update sau khi hành động
- [x] Updates cho `gui/components/positions_frame.py` (để link sang details)

**Lưu ý:** Chỉ còn Task 4.2.3 (Optimistic UI updates) chưa hoàn thành. Tất cả core features và GUI integration đã sẵn sàng để testing.
