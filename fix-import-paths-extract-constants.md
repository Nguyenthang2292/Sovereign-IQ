# Fix Import Paths & Extract Constants

## Goal
Chuẩn hóa import trong `modules/auto_trade` sang `modules.auto_trade.gui.*` và tách magic numbers vào `DatabasePanelConfig` cho Database panel.

## Tasks

- [x] **1. Fix imports – entry & main_window**  
  Trong `run_gui.py`: `from gui.main_window` → `from modules.auto_trade.gui.main_window`.  
  Trong `main_window.py`, `layout.py`: tất cả `from gui.` → `from modules.auto_trade.gui.`  
  → Verify: `python -c "from modules.auto_trade.run_gui import *"` không lỗi.

- [x] **2. Fix imports – components**  
  Trong `config_panel.py`, `trade_form.py`, `stats_frame.py`, `account_frame.py`, `auto_trade_control.py`, `scanner_control.py`, `logs_viewer.py`: mỗi `from gui.` → `from modules.auto_trade.gui.`  
  → Verify: `python -c "from modules.auto_trade.gui.main_window.layout import LayoutManager"` không lỗi.

- [x] **3. Fix imports – dialogs & main_window utils**  
  Trong `shortcuts_help.py`, `updaters.py`, `settings_handler.py`, `websocket_handler.py`: `from gui.` → `from modules.auto_trade.gui.`  
  → Verify: `rg "from gui\." modules/auto_trade --glob "*.py"` trả về 0 kết quả.

- [x] **4. Tạo config cho Database panel**  
  Tạo `modules/auto_trade/gui/config/__init__.py` (rỗng hoặc export).  
  Tạo `modules/auto_trade/gui/config/database_panel_config.py` với class `DatabasePanelConfig`: `DEFAULT_DB_NAME`, `DEFAULT_PAGE_SIZE`, `INITIAL_PAGE`, `DEFAULT_RECONCILE_HOURS`, `MAX_RECONCILE_ERRORS_SHOWN`, `DEFAULT_DAYS_TO_KEEP`, `STATS_REFRESH_INTERVAL_MS`, `TEXTBOX_FONT`, `TITLE_FONT`, layout weights/padx nếu dùng chung.  
  → Verify: `python -c "from modules.auto_trade.gui.config.database_panel_config import DatabasePanelConfig; print(DatabasePanelConfig.DEFAULT_PAGE_SIZE)"` in `20`.

- [x] **5. Dùng config trong database_panel.py**  
  Import `DatabasePanelConfig`, thay `"crypto_trading.db"` bằng `DatabasePanelConfig.DEFAULT_DB_NAME`, grid `weight=3`/`weight=2` và padx bằng constants (thêm vào config nếu cần).  
  → Verify: Mở tab Database trong GUI, panel hiển thị bình thường.

- [x] **6. Dùng config trong database sections**  
  Trong `data_viewer_section.py`: `page_size=20` → `DatabasePanelConfig.DEFAULT_PAGE_SIZE`, font tuple → `TEXTBOX_FONT`/`TITLE_FONT`.  
  Trong `actions_section.py`: `days_to_keep=90`, `since_hours=24` → config.  
  Các section khác dùng `TITLE_FONT` thay cho `("Roboto", 14, "bold")` nếu có.  
  → Verify: Chạy pytest `tests/auto_trade/gui` (hoặc nhóm test database panel) pass.

- [x] **7. Verification**  
  Chạy `pytest tests/auto_trade/gui/test_database_panel.py` - 9/9 tests passed. Không còn `from gui.` imports trong Python files. DatabasePanelConfig hoạt động đúng.

## Done When

- [x] Không còn `from gui.` trong `modules/auto_trade/**/*.py`.
- [x] `DatabasePanelConfig` tồn tại và được dùng trong `database_panel.py` + các database sections.
- [x] Pytest database_panel pass (9/9); GUI imports hoạt động bình thường.

## Notes

- Replace từng file một, chạy test/import sau mỗi nhóm file để tránh break.
- Nếu có script chạy từ repo root với `sys.path` chứa `modules/auto_trade`, giữ nguyên; chỉ đổi import bên trong package.
- REF: `modules/auto_trade/REFACTORING_RECOMMENDATIONS.md` mục "4. Fix Import Paths", "5. Extract Constants".
