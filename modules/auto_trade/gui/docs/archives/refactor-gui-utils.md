# Refactor `gui/utils/` → `gui/services/`

## Goal

Di chuyển 7 file business-service sai chỗ từ `gui/utils/` sang `gui/services/`,
giữ nguyên hoàn toàn logic — chỉ thay đổi vị trí và import paths.

---

## Đợt 1 — Service data chính (ưu tiên cao)

### 1. `data_service.py`

- [x] Copy → `gui/services/data_service.py`
- [x] Cập nhật import trong `main_window/main_window.py` (line 16)
- [x] Cập nhật import trong `components/trade_form.py` (lines 163, 178, 438 — lazy imports)
- [x] Cập nhật import trong `tests/test_data_service.py` (line 5)
- [x] Xóa `gui/utils/data_service.py`
- [x] Export `DataService` trong `services/__init__.py`
- Verify: `grep -r "gui.utils.data_service" modules/` → 0 kết quả

### 2. `position_sync_service.py`

- [x] Copy → `gui/services/position_sync_service.py`
- [x] Cập nhật import trong `main_window/position_actions.py` (line 67)
- [x] Cập nhật import trong `main_window/updaters.py` (line 195)
- [x] Cập nhật import trong `components/database/actions_section.py` (line 208)
- [x] Cập nhật import trong `database/__init__.py` (line 137)
- [x] Xóa `gui/utils/position_sync_service.py`
- [x] Export `PositionSyncService` trong `services/__init__.py`
- Verify: `grep -r "gui.utils.position_sync_service" modules/` → 0 kết quả

### 3. `tp_sl_sync.py`

- [x] Copy → `gui/services/tp_sl_sync.py`
- [x] Cập nhật import trong `main_window/settings_recovery_mixin.py` (line 230 — lazy import)
- [x] Cập nhật import trong `gui/services/data_service.py` (line 214 — sau khi task 1 xong)
- [x] Cập nhật import trong `gui/services/position_sync_service.py` (line 104 — sau task 2 xong)
- [x] Xóa `gui/utils/tp_sl_sync.py`
- [x] Export `TPSLSyncService` trong `services/__init__.py`
- Verify: `grep -r "gui.utils.tp_sl_sync" modules/` → 0 kết quả

---

## Đợt 2 — Manager & DryRun layer

### 4. `settings_manager.py`

- [x] Copy → `gui/services/settings_manager.py`
- [x] Cập nhật import trong `main_window/main_window.py` (line 18)
- [x] Xóa `gui/utils/settings_manager.py`
- [x] Export `SettingsManager` trong `services/__init__.py`
- Verify: `grep -r "gui.utils.settings_manager" modules/` → 0 kết quả

### 5. `credential_manager.py`

- [x] Copy → `gui/services/credential_manager.py`
- [x] Cập nhật import trong `main_window/position_actions.py` (line 66)
- [x] Cập nhật import trong `gui/services/websocket_data_service.py` (line 22)
- [x] Cập nhật import trong `components/config_panel_parts/credentials.py` (lines 9, 49, 114)
- [x] Cập nhật import trong `gui/services/data_service.py` (line 159 — lazy import)
- [x] Xóa `gui/utils/credential_manager.py`
- [x] Export `CredentialManager` trong `services/__init__.py`
- Verify: `grep -r "gui.utils.credential_manager" modules/` → 0 kết quả

### 6. `dry_run_db.py` + `dry_run_executor.py`

- [x] Tạo sub-package `gui/services/dry_run/__init__.py`
- [x] Copy `dry_run_db.py` → `gui/services/dry_run/dry_run_db.py`
- [x] Copy `dry_run_executor.py` → `gui/services/dry_run/dry_run_executor.py`
- [x] Cập nhật import trong `dry_run_executor.py` (line 11 — internal import)
- [x] Cập nhật import lazy trong `gui/services/data_service.py` (lines 363, 507)
- [x] Xóa `gui/utils/dry_run_db.py` và `gui/utils/dry_run_executor.py`
- [x] Export `DryRunDB`, `DryRunExecutor` trong `services/dry_run/__init__.py`
- Verify: `grep -r "gui.utils.dry_run" modules/` → 0 kết quả

---

## Done When

- [x] `gui/utils/` chỉ còn ~12 file helper thuần túy (stateless)
- [x] `gui/services/` có đủ 7 service classes + sub-package `dry_run/`
- [x] Tất cả `grep "gui.utils.<tên-file>"` trả về 0 kết quả
- [x] App khởi động bình thường (không có `ImportError`)

---

## Lưu ý

- **Thứ tự**: Làm đợt 1 trước, đặc biệt `data_service` trước `tp_sl_sync` (vì `data_service` import `tp_sl_sync`)
- **Lazy imports**: Một số import nằm trong hàm (không phải top-level) — cần grep kỹ
- **Test**: Sau mỗi đợt chạy `pytest modules/auto_trade/tests/ -x` để verify
- **`mock_price_feed.py`**: Tạm thời giữ ở `utils/` — cả `services/` và `utils/` đều dùng
