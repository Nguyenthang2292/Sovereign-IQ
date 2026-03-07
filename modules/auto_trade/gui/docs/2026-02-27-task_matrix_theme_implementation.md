# TASK: Matrix Theme Implementation

> **Nguồn:** `2026-02-27-matrix_theme_design_spec.md`  
> **Ngày tạo:** 2026-03-07  
> **Loại:** Theme Reskin (không thay đổi layout/kiến trúc)  
> **Trạng thái:** 🔴 TODO

---

## Goal

Chuyển đổi toàn bộ GUI Auto Trade Dashboard từ dark-blue theme sang **Matrix Terminal Theme** (neon green on black). Thay đổi chỉ ở lớp visual (colors, fonts, borders) — không động đến layout hay kiến trúc.

---

## Phase 1 — Color System

> **Mục tiêu:** Cập nhật `utils/colors.py` — zero layout risk

- [x] **T1.1** Mở `utils/colors.py`, đổi giá trị các background constants:
  - `BG_DARK` = `#000000`
  - `BG_CARD_DARK` = `#0a0a0a`
  - `BG_HEADER_DARK` = `#050505`
  - → Verify: import module, in `Colors.BG_DARK` → `#000000`

- [x] **T1.2** Đổi text color constants:
  - `TEXT_PRIMARY_DARK` = `#00FF41`
  - `TEXT_SECONDARY_DARK` = `#00aa2a`
  - → Verify: `Colors.TEXT_PRIMARY_DARK` == `#00FF41`

- [x] **T1.3** Thêm **5 constants mới** (background):

  ```python
  BG_HIGHLIGHT = "#001a00"
  BG_INPUT     = "#0d0d0d"
  TEXT_DIM     = "#005500"
  TEXT_BRIGHT  = "#33FF66"
  ```

  → Verify: `hasattr(Colors, 'BG_HIGHLIGHT')` == True

- [x] **T1.4** Thêm **4 constants border/accent** mới:

  ```python
  BORDER_NEON   = "#003B00"
  BORDER_ACTIVE = "#00FF41"
  ACCENT        = "#00FF41"
  ACCENT_DIM    = "#004400"
  ```

  → Verify: `Colors.BORDER_NEON` == `#003B00`

- [x] **T1.5** Cập nhật button color constants theo bảng spec (section 2.4):
  - `BTN_PRIMARY` → `#00CC33`, text `#000000`
  - `BTN_PRIMARY_HOVER` → `#00FF41`
  - `BTN_SUCCESS_TEXT` → `#000000`
  - `BTN_DANGER_HOVER` → `#ff6666`
  - `BTN_DANGER_ALT` → `#4a0000`, text `#ff4444`
  - `BTN_DANGER_ALT_HOVER` → `#660000`, text `#ff6666`
  - `BTN_NEUTRAL` → `#1a1a1a`, text `#00FF41`
  - `BTN_NEUTRAL_HOVER` → `#003300`, text `#00FF41`
  - `BTN_WARNING_HOVER` → `#ffcc33`
  - → Verify: chạy GUI, kiểm tra màu nút Primary/Neutral

- [x] **T1.6** Cập nhật trading semantic colors:
  - `NEUTRAL` → `#005500` (từ `#888888`)
  - `DRY_RUN` → `#00FF41` (từ `#4488ff`)
  - → Verify: các status badge hiện đúng màu

- [x] **T1.7** Xóa tất cả `*_LIGHT` constants (`BG_LIGHT`, `BG_CARD_LIGHT`, ...):
  - → Verify: `grep -r "_LIGHT" utils/colors.py` trả về rỗng

- [x] **T1.8** Simplify class methods — bỏ if/else theme check, hardcode dark:

  ```python
  @classmethod
  def get_bg(cls) -> str: return cls.BG_DARK
  @classmethod
  def is_dark_mode(cls) -> bool: return True
  @classmethod
  def get_current_theme(cls) -> str: return "Dark"
  ```

  - Tương tự cho `get_card_bg()`, `get_text_primary()`, `get_accent()` (→ `#00FF41`)
  - → Verify: `Colors.is_dark_mode()` == True, `Colors.get_accent()` == `#00FF41`

---

## Phase 2 — Theme Config

> **Mục tiêu:** Load custom Matrix theme JSON

- [x] **T2.1** Tạo file `config/matrix_theme.json` với nội dung đầy đủ từ spec (section 6.2):
  - Bao gồm CTk, CTkFrame, CTkButton, CTkLabel, CTkEntry, CTkTextbox, CTkScrollbar, CTkSwitch, CTkOptionMenu, CTkComboBox, CTkProgressBar, CTkCheckBox, CTkSegmentedButton
  - → Verify: file hợp lệ JSON (`python -m json.tool config/matrix_theme.json`)

- [x] **T2.2** Cập nhật `main_window.py` (hoặc entry point) — load custom theme:

  ```python
  ctk.set_appearance_mode("dark")
  ctk.set_default_color_theme("path/to/config/matrix_theme.json")
  ```

  - → Verify: GUI khởi động không lỗi, CTkButton mặc định xanh lá

- [x] **T2.3** Cập nhật `settings_handler.py` — disable light mode toggle:
  - Option A: Ẩn hoàn toàn nút toggle theme
  - Option B: Giữ nút nhưng callback không làm gì (no-op)
  - → Verify: click toggle theme → không thay đổi giao diện

---

## Phase 3 — Font System

> **Mục tiêu:** Monospace toàn bộ UI — thay thế Arial bằng Consolas

- [x] **T3.1** Tạo `utils/fonts.py` với class `Fonts` (hoặc thêm vào `colors.py`):

  ```python
  class Fonts:
      FAMILY = "Consolas"
      H1     = (FAMILY, 16, "bold")
      H2     = (FAMILY, 14, "bold")
      H3     = (FAMILY, 12, "bold")
      BODY   = (FAMILY, 11)
      SMALL  = (FAMILY, 10)
      TINY   = (FAMILY, 9)
      DATA   = (FAMILY, 18, "bold")
      INPUT  = (FAMILY, 12)
      BUTTON = (FAMILY, 12, "bold")
  ```

  - → Verify: `from utils.fonts import Fonts; Fonts.H1` == `("Consolas", 16, "bold")`

- [x] **T3.2** Cập nhật `components/auto_trade_control.py` (~10 chỗ dùng `"Arial"`):
  - Thay hết `("Arial", ...)` → `Fonts.BODY` / `Fonts.H2` / `Fonts.SMALL` tùy context
  - → Verify: không còn string `"Arial"` trong file (`grep "Arial" components/auto_trade_control.py`)

- [x] **T3.3** Cập nhật `components/account_frame.py` (~3 chỗ):
  - → Verify: `grep "Arial" components/account_frame.py` rỗng

- [x] **T3.4** Cập nhật `dialogs/close_confirmation.py` (~10 chỗ):
  - → Verify: `grep "Arial" dialogs/close_confirmation.py` rỗng

- [x] **T3.5** Cập nhật `dialogs/shortcuts_help.py` (~4 chỗ):
  - → Verify: `grep "Arial" dialogs/shortcuts_help.py` rỗng

- [x] **T3.6** Cập nhật `config/database_panel_config.py` (3 chỗ — mix `Roboto`, `Consolas`):
  - Chuẩn hóa tất cả về `Fonts.FAMILY`
  - → Verify: `grep -E '"Roboto"|"Arial"' config/database_panel_config.py` rỗng

---

## Phase 4 — Border & Highlight

> **Mục tiêu:** Viền neon mờ cho cards + highlight cho dữ liệu quan trọng

- [x] **T4.1** Tìm tất cả `CTkFrame` dùng làm card/panel trong components:
  - Chạy: `grep -rn "CTkFrame" modules/auto_trade/gui/components/`
  - List ra các frame cần add border

- [x] **T4.2** Thêm `border_width=1, border_color=Colors.BORDER_NEON` vào card frames:
  - Tập trung vào: Account Overview, Scanner Panel, Position Panel
  - → Verify: chạy GUI → thấy viền xanh mờ xung quanh cards

- [x] **T4.3** Thêm highlight frames cho dữ liệu critical (PnL, Balance, Signal):

  ```python
  frame.configure(fg_color=Colors.BG_HIGHLIGHT, border_color=Colors.BORDER_NEON)
  ```

  - → Verify: số PnL/Balance có nền `#001a00`, viền `#003B00`

- [x] **T4.4** Style tabview theo spec (section 4.4):

  ```python
  tabview.configure(
      segmented_button_fg_color="#0a0a0a",
      segmented_button_selected_color="#003300",
      segmented_button_selected_hover_color="#004400",
      segmented_button_unselected_color="#0a0a0a",
      segmented_button_unselected_hover_color="#001a00",
      text_color=Colors.TEXT_PRIMARY_DARK,
  )
  ```

  - → Verify: tab được chọn có nền `#003300`, tab khác `#0a0a0a`

- [x] **T4.5** Style active/focus border khi frame được focus:

  ```python
  frame.configure(border_color=Colors.BORDER_ACTIVE)  # #00FF41
  ```

  - → Verify: click vào frame → viền sáng `#00FF41`

---

## Phase 5 — Icon Recoloring

> **Mục tiêu:** Tất cả SVG icons dùng màu `#00FF41` mặc định

- [x] **T5.1** Mở `utils/svg_icons.py`, set `DEFAULT_ICON_COLOR = Colors.ACCENT` (`#00FF41`):
  - → Verify: `svg_icons.DEFAULT_ICON_COLOR` == `#00FF41`

- [x] **T5.2** Kiểm tra tất cả `get_icon()` calls trong codebase:
  - Chạy: `grep -rn "get_icon" modules/auto_trade/gui/`
  - Xác định call nào đang hardcode color cũ (`#4488ff`, `#888888`, ...)

- [x] **T5.3** Cập nhật các `get_icon()` call về dùng default color mới:
  - Ngoại lệ giữ semantic color: icon cạnh profit → `Colors.PROFIT`, loss → `Colors.LOSS`, warning → `Colors.BTN_WARNING`
  - → Verify: chạy GUI, tất cả icons toolbar/navbar màu xanh `#00FF41`

---

## Phase 6 — Verification & Testing

> **Luôn là Phase cuối cùng**

- [x] **T6.1** Chạy GUI đầy đủ: `python run_auto_trade_gui.py`
  - → Verify: launch không có exception, không bị lỗi import

- [x] **T6.2** Kiểm tra visual tất cả 6 tabs:
  - Tab 1: Scanner — colors đúng, không bị white flash
  - Tab 2: Positions — PnL numbers có highlight background
  - Tab 3: Orders — buttons đúng semantic color
  - Tab 4: Account — Balance data dùng `DATA` font
  - Tab 5: Settings — toggles/switches màu xanh Matrix
  - Tab 6: Database — font monospace, borders hiện đúng

- [x] **T6.3** Kiểm tra contrast WCAG — neon green trên đen:
  - `#00FF41` trên `#000000` = ~15:1 ✅ (pass AA & AAA)
  - `#00aa2a` trên `#000000` = ~4.8:1 ✅ (pass AA)
  - `#005500` trên `#000000` — chỉ dùng cho disabled/placeholder (decorative)

- [x] **T6.4** Test button states — từng button:
  - Normal state → màu đúng
  - Hover state → sáng hơn đúng spec
  - Disabled state → `TEXT_DIM` (`#005500`)

- [x] **T6.5** Test readability với monospace font:
  - Kiểm tra alignment cho dữ liệu dạng bảng (numbers align đúng)
  - Kiểm tra không bị overflow text trong các label nhỏ

- [x] **T6.6** Regression — light mode không còn hoạt động:
  - Verify: không còn button toggle theme, hoặc toggle không có tác dụng

- [x] **T6.7** Commit sau khi tất cả checks pass:

  ```
  git add -A
  git commit -m "feat(gui): apply Matrix Terminal theme reskin"
  ```

---

## Done When

- [ ] GUI khởi động thành công với Matrix dark theme
- [ ] Tất cả 6 tabs hiển thị đúng màu neon green / đen
- [ ] Không còn hardcoded `"Arial"` trong 5 files listed
- [ ] `colors.py` không còn `*_LIGHT` constants
- [ ] `utils/fonts.py` tồn tại với class `Fonts`
- [ ] `config/matrix_theme.json` hợp lệ và được load
- [ ] SVG icons mặc định màu `#00FF41`
- [ ] Không có lỗi console khi chạy

---

## Files bị ảnh hưởng

| File | Phase | Loại thay đổi |
|------|-------|---------------|
| `utils/colors.py` | P1 | Sửa constants, xóa LIGHT, simplify methods |
| `config/matrix_theme.json` | P2 | Tạo mới |
| `main_window.py` | P2 | Load custom theme |
| `settings_handler.py` | P2 | Disable light mode toggle |
| `utils/fonts.py` | P3 | Tạo mới |
| `components/auto_trade_control.py` | P3 | Thay Arial → Fonts |
| `components/account_frame.py` | P3 | Thay Arial → Fonts |
| `dialogs/close_confirmation.py` | P3 | Thay Arial → Fonts |
| `dialogs/shortcuts_help.py` | P3 | Thay Arial → Fonts |
| `config/database_panel_config.py` | P3 | Thay Arial/Roboto → Fonts |
| `components/*` (card frames) | P4 | Add border_width, border_color |
| `utils/svg_icons.py` | P5 | Set default icon color |

## Notes

- **KHÔNG thay đổi:** layout 6-tab, kiến trúc Mixin/Facade/Observer, animation
- **Backward compat:** `is_dark_mode()` và `get_current_theme()` giữ lại, chỉ hardcode return value
- **Font fallback:** nếu Consolas không có → `"Courier New"` → `"monospace"`
- **Icon ngoại lệ:** profit/loss/warning icons giữ semantic color gốc
