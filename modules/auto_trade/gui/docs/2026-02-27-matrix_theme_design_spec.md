# Matrix Theme Design Specification

> **Loại:** Theme Reskin (không thay đổi layout/kiến trúc)  
> **Ngày:** 2026-03-07  
> **Trạng thái:** Approved — Sẵn sàng implement

---

## 1. Tổng quan

Chuyển đổi toàn bộ GUI Auto Trade Dashboard từ theme hiện tại (dark blue accent) sang **Matrix Terminal Theme** — lấy cảm hứng từ giao diện hacker/terminal trong phim The Matrix.

### Phạm vi thay đổi
- ✅ Bảng màu (colors.py)
- ✅ Font hệ thống → monospace toàn bộ
- ✅ Button colors (semantic + Matrix style)
- ✅ Border/highlight cho frames & dữ liệu quan trọng
- ✅ Icon colors (SVG)
- ❌ Layout 6-tab — giữ nguyên
- ❌ Kiến trúc code (Mixin, Facade, Observer) — giữ nguyên
- ❌ Animation / Matrix Rain — không thêm

### Nguyên tắc thiết kế
1. **Dark-only** — Loại bỏ light mode, cố định dark Matrix
2. **Monospace everywhere** — Toàn bộ UI dùng Consolas/Courier New
3. **Semantic colors preserved** — Xanh lá = profit/success, Đỏ = loss/danger, Vàng = warning
4. **Neon accent** — Viền xanh lá mờ + highlight background cho dữ liệu quan trọng
5. **Performance first** — Không animation, không hiệu ứng nặng

---

## 2. Bảng màu Matrix

### 2.1 Nền (Background)

| Token hiện tại | Giá trị cũ | Giá trị mới | Mô tả |
|---|---|---|---|
| `BG_DARK` | `#1a1a1a` | `#000000` | Nền chính — đen tuyệt đối |
| `BG_CARD_DARK` | `#2b2b2b` | `#0a0a0a` | Nền card/panel — đen nhẹ |
| `BG_HEADER_DARK` | `#1e1e1e` | `#050505` | Nền header — gần đen |
| *(mới)* `BG_HIGHLIGHT` | — | `#001a00` | Highlight background cho dữ liệu quan trọng |
| *(mới)* `BG_INPUT` | — | `#0d0d0d` | Nền input fields |

### 2.2 Text

| Token hiện tại | Giá trị cũ | Giá trị mới | Mô tả |
|---|---|---|---|
| `TEXT_PRIMARY_DARK` | `#ffffff` | `#00FF41` | Text chính — neon green (Matrix signature) |
| `TEXT_SECONDARY_DARK` | `#888888` | `#00aa2a` | Text phụ — green mờ hơn |
| *(mới)* `TEXT_DIM` | — | `#005500` | Text rất mờ — placeholder, disabled |
| *(mới)* `TEXT_BRIGHT` | — | `#33FF66` | Text sáng nhất — tiêu đề nổi bật |

### 2.3 Border & Accent

| Token | Giá trị | Mô tả |
|---|---|---|
| *(mới)* `BORDER_NEON` | `#003B00` | Viền card/frame — xanh lá mờ |
| *(mới)* `BORDER_ACTIVE` | `#00FF41` | Viền khi focus/active — neon sáng |
| *(mới)* `ACCENT` | `#00FF41` | Accent chính (thay `#4488ff`) |
| *(mới)* `ACCENT_DIM` | `#004400` | Accent mờ cho hover states |

### 2.4 Button Colors (Semantic + Matrix)

| Token | Giá trị cũ | Giá trị mới | Text Color | Mô tả |
|---|---|---|---|---|
| `BTN_PRIMARY` | `#4488ff` | `#00CC33` | `#000000` | Primary action — xanh lá Matrix |
| `BTN_PRIMARY_HOVER` | `#0066ff` | `#00FF41` | `#000000` | Primary hover — sáng hơn |
| `BTN_SUCCESS` | `#00a855` | `#00a855` | `#000000` | Success/LONG — **giữ nguyên** |
| `BTN_SUCCESS_HOVER` | `#007a3d` | `#00CC66` | `#000000` | Success hover |
| `BTN_SUCCESS_TEXT` | `white` | `#000000` | — | Text đen trên nền xanh |
| `BTN_DANGER` | `#ff4444` | `#ff4444` | `#ffffff` | Danger/SHORT — **giữ nguyên** |
| `BTN_DANGER_HOVER` | `#cc0000` | `#ff6666` | `#000000` | Danger hover — sáng hơn |
| `BTN_DANGER_ALT` | `#7f1d1d` | `#4a0000` | `#ff4444` | Danger alternative — đậm hơn |
| `BTN_DANGER_ALT_HOVER` | `#991b1b` | `#660000` | `#ff6666` | Danger alt hover |
| `BTN_NEUTRAL` | `#555555` | `#1a1a1a` | `#00FF41` | Neutral — nền tối, text xanh |
| `BTN_NEUTRAL_HOVER` | `#333333` | `#003300` | `#00FF41` | Neutral hover — hint xanh lá |
| `BTN_WARNING` | `#ffaa00` | `#ffaa00` | `#000000` | Warning — **giữ nguyên** |
| `BTN_WARNING_HOVER` | `#cc8800` | `#ffcc33` | `#000000` | Warning hover |

### 2.5 Trading Colors (Semantic — giữ nguyên)

| Token | Giá trị | Ghi chú |
|---|---|---|
| `LONG` / `PROFIT` | `#00ff88` | Giữ nguyên — tương thích Matrix palette |
| `SHORT` / `LOSS` | `#ff4444` | Giữ nguyên — semantic đỏ |
| `NEUTRAL` | `#005500` | Đổi từ `#888888` → xanh mờ (Matrix style) |
| `PRODUCTION` | `#ff4444` | Giữ nguyên |
| `DEMO` | `#ffaa00` | Giữ nguyên |
| `DRY_RUN` | `#00FF41` | Đổi từ `#4488ff` → neon green (Matrix) |

### 2.6 Loại bỏ Light Mode

Xóa toàn bộ các constant `*_LIGHT` và simplify các class method:

```python
# TRƯỚC (theme-aware):
BG_LIGHT = "#f0f0f0"
BG_CARD_LIGHT = "#ffffff"
...
@classmethod
def get_bg(cls) -> str:
    return cls.BG_DARK if cls.is_dark_mode() else cls.BG_LIGHT

# SAU (dark-only):
@classmethod
def get_bg(cls) -> str:
    return cls.BG_DARK
```

Các method `is_dark_mode()`, `get_current_theme()` vẫn giữ lại cho backward compat nhưng luôn trả `True`/`"Dark"`.

---

## 3. Font System

### 3.1 Quy tắc: Monospace toàn bộ

| Sử dụng | Font cũ | Font mới | Size | Weight |
|---|---|---|---|---|
| Tiêu đề lớn (H1) | `Arial 16 bold` | `Consolas 16 bold` | 16 | bold |
| Tiêu đề (H2) | `Arial 14 bold` | `Consolas 14 bold` | 14 | bold |
| Section title | `Arial 12 bold` | `Consolas 12 bold` | 12 | bold |
| Body text | `Arial 11` | `Consolas 11` | 11 | normal |
| Label | `Arial 10` | `Consolas 10` | 10 | normal |
| Data/numbers | `Arial 18 bold` | `Consolas 18 bold` | 18 | bold |
| Keyboard shortcuts | `Consolas 11 bold` | `Consolas 11 bold` | 11 | bold (đã OK) |
| Input fields | varies | `Consolas 12` | 12 | normal |

### 3.2 Font Constant (đề xuất thêm vào colors.py hoặc tạo fonts.py)

```python
class Fonts:
    """Matrix theme monospace font system"""
    FAMILY = "Consolas"  # Fallback: "Courier New", "monospace"
    
    H1     = (FAMILY, 16, "bold")   # Main titles
    H2     = (FAMILY, 14, "bold")   # Section headers
    H3     = (FAMILY, 12, "bold")   # Sub-headers
    BODY   = (FAMILY, 11)           # Body text
    SMALL  = (FAMILY, 10)           # Labels, secondary
    TINY   = (FAMILY, 9)            # Footnotes
    DATA   = (FAMILY, 18, "bold")   # Big data display (balances, PnL)
    INPUT  = (FAMILY, 12)           # Input fields
    BUTTON = (FAMILY, 12, "bold")   # Button text
```

### 3.3 Files cần sửa font

Các file đang hardcode `"Arial"` cần chuyển sang `Fonts.FAMILY`:

| File | Số lần dùng font |
|---|---|
| `components/auto_trade_control.py` | ~10 lần |
| `components/account_frame.py` | ~3 lần |
| `dialogs/close_confirmation.py` | ~10 lần |
| `dialogs/shortcuts_help.py` | ~4 lần |
| `config/database_panel_config.py` | 3 lần (`Roboto`, `Consolas`) |

---

## 4. Border & Highlight System

### 4.1 Card/Frame Borders

Tất cả `CTkFrame` dùng làm card/panel sẽ có:

```python
# Viền neon mờ
frame.configure(
    border_width=1,
    border_color=Colors.BORDER_NEON,  # #003B00
    fg_color=Colors.BG_CARD_DARK,     # #0a0a0a
)
```

### 4.2 Active/Focus Borders

Khi frame được focus hoặc tab được chọn:

```python
frame.configure(border_color=Colors.BORDER_ACTIVE)  # #00FF41
```

### 4.3 Data Highlight

Dữ liệu quan trọng (PnL, giá hiện tại, signals) có highlight background:

```python
# Highlight frame cho số liệu quan trọng
highlight_frame = ctk.CTkFrame(
    parent,
    fg_color=Colors.BG_HIGHLIGHT,       # #001a00
    border_width=1,
    border_color=Colors.BORDER_NEON,    # #003B00
)
```

### 4.4 Tab Strip

CustomTkinter tabview styling:

```python
tabview.configure(
    fg_color=Colors.BG_DARK,            # #000000
    segmented_button_fg_color="#0a0a0a",
    segmented_button_selected_color="#003300",
    segmented_button_selected_hover_color="#004400",
    segmented_button_unselected_color="#0a0a0a",
    segmented_button_unselected_hover_color="#001a00",
    text_color=Colors.TEXT_PRIMARY_DARK,  # #00FF41
)
```

---

## 5. Icon Colors

### 5.1 SVG Icon Recoloring

`utils/svg_icons.py` render Lucide icons với color parameter:

```python
# TRƯỚC: dùng đa màu
icon = get_icon("settings", color="#4488ff")

# SAU: mặc định neon green
DEFAULT_ICON_COLOR = Colors.ACCENT  # #00FF41
icon = get_icon("settings", color=DEFAULT_ICON_COLOR)
```

Ngoại lệ — icon trong context semantic vẫn dùng semantic color:
- Icon cạnh profit → `Colors.PROFIT` (#00ff88)
- Icon cạnh loss → `Colors.LOSS` (#ff4444)
- Icon cạnh warning → `Colors.BTN_WARNING` (#ffaa00)

---

## 6. CustomTkinter Theme Config

### 6.1 Thay đổi trong main_window.py

```python
# TRƯỚC:
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# SAU:
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")  # Hoặc custom theme JSON
```

### 6.2 Custom Theme JSON (nếu cần)

Tạo file `config/matrix_theme.json`:

```json
{
  "CTk": {
    "fg_color": ["#000000", "#000000"]
  },
  "CTkToplevel": {
    "fg_color": ["#000000", "#000000"]
  },
  "CTkFrame": {
    "fg_color": ["#0a0a0a", "#0a0a0a"],
    "border_color": ["#003B00", "#003B00"]
  },
  "CTkButton": {
    "fg_color": ["#00CC33", "#00CC33"],
    "hover_color": ["#00FF41", "#00FF41"],
    "text_color": ["#000000", "#000000"],
    "border_color": ["#003B00", "#003B00"]
  },
  "CTkLabel": {
    "text_color": ["#00FF41", "#00FF41"]
  },
  "CTkEntry": {
    "fg_color": ["#0d0d0d", "#0d0d0d"],
    "border_color": ["#003B00", "#003B00"],
    "text_color": ["#00FF41", "#00FF41"],
    "placeholder_text_color": ["#005500", "#005500"]
  },
  "CTkTextbox": {
    "fg_color": ["#0d0d0d", "#0d0d0d"],
    "border_color": ["#003B00", "#003B00"],
    "text_color": ["#00FF41", "#00FF41"]
  },
  "CTkScrollbar": {
    "fg_color": ["#0a0a0a", "#0a0a0a"],
    "button_color": ["#003B00", "#003B00"],
    "button_hover_color": ["#00FF41", "#00FF41"]
  },
  "CTkSwitch": {
    "fg_color": ["#003B00", "#003B00"],
    "progress_color": ["#00FF41", "#00FF41"],
    "button_color": ["#00CC33", "#00CC33"],
    "button_hover_color": ["#00FF41", "#00FF41"]
  },
  "CTkOptionMenu": {
    "fg_color": ["#0a0a0a", "#0a0a0a"],
    "button_color": ["#003B00", "#003B00"],
    "button_hover_color": ["#004400", "#004400"],
    "text_color": ["#00FF41", "#00FF41"]
  },
  "CTkComboBox": {
    "fg_color": ["#0d0d0d", "#0d0d0d"],
    "border_color": ["#003B00", "#003B00"],
    "button_color": ["#003B00", "#003B00"],
    "button_hover_color": ["#004400", "#004400"],
    "text_color": ["#00FF41", "#00FF41"]
  },
  "CTkProgressBar": {
    "fg_color": ["#0a0a0a", "#0a0a0a"],
    "progress_color": ["#00FF41", "#00FF41"]
  },
  "CTkCheckBox": {
    "fg_color": ["#003B00", "#003B00"],
    "hover_color": ["#004400", "#004400"],
    "checkmark_color": ["#00FF41", "#00FF41"],
    "text_color": ["#00FF41", "#00FF41"]
  },
  "CTkSegmentedButton": {
    "fg_color": ["#0a0a0a", "#0a0a0a"],
    "selected_color": ["#003300", "#003300"],
    "selected_hover_color": ["#004400", "#004400"],
    "unselected_color": ["#0a0a0a", "#0a0a0a"],
    "unselected_hover_color": ["#001a00", "#001a00"],
    "text_color": ["#00FF41", "#00FF41"],
    "text_color_disabled": ["#005500", "#005500"]
  }
}
```

---

## 7. Implementation Plan

### Phase 1: Color System (ít rủi ro nhất)
1. Cập nhật `utils/colors.py` — đổi tất cả color constants theo bảng trên
2. Xóa `*_LIGHT` constants, simplify class methods
3. Thêm constants mới: `BORDER_NEON`, `BORDER_ACTIVE`, `BG_HIGHLIGHT`, `BG_INPUT`, `TEXT_DIM`, `TEXT_BRIGHT`, `ACCENT`, `ACCENT_DIM`
4. Đổi `get_accent()` từ `#4488ff` → `#00FF41`

### Phase 2: Theme Config
1. Tạo `config/matrix_theme.json`
2. Cập nhật `main_window.py` — load custom theme
3. Cố định `set_appearance_mode("dark")`
4. Xử lý `settings_handler.py` — disable light mode toggle (hoặc remove)

### Phase 3: Font System
1. Tạo `utils/fonts.py` hoặc thêm `Fonts` class vào `colors.py`
2. Update tất cả hardcoded `"Arial"` → `Fonts.FAMILY` trong:
   - `components/auto_trade_control.py`
   - `components/account_frame.py`
   - `dialogs/close_confirmation.py`
   - `dialogs/shortcuts_help.py`
   - `config/database_panel_config.py`

### Phase 4: Border & Highlight
1. Thêm `border_width=1, border_color=BORDER_NEON` vào các card frames
2. Thêm highlight frames cho dữ liệu PnL, giá, signals
3. Style tabview theo spec

### Phase 5: Icon Recoloring
1. Set default icon color = `#00FF41` trong `svg_icons.py`
2. Kiểm tra tất cả `get_icon()` calls — chuyển sang mặc định mới

### Phase 6: Testing
1. Chạy GUI — kiểm tra tất cả 6 tabs
2. Verify WCAG contrast ratios (neon green trên đen = ~15:1 ✅)
3. Kiểm tra readability với monospace fonts
4. Test tất cả button states (normal, hover, disabled)

---

## 8. Color Preview (Text-based)

```
╔══════════════════════════════════════════════════╗
║  ▓▓ MATRIX AUTO-TRADE DASHBOARD ▓▓              ║  ← #00FF41 on #000000
╠══════════════════════════════════════════════════╣
║                                                  ║
║  ┌──────────────────────────────────┐            ║  ← Card: #0a0a0a, border #003B00
║  │  Account Overview                │            ║  ← H2: #33FF66 (TEXT_BRIGHT)
║  │  ┌────────┐  ┌────────┐         │            ║
║  │  │ $1,234 │  │ +$56.7 │         │            ║  ← Data on #001a00 highlight
║  │  │ Balance│  │  PnL   │         │            ║
║  │  └────────┘  └────────┘         │            ║
║  │  Status: • Connected             │            ║  ← #00aa2a (TEXT_SECONDARY)
║  └──────────────────────────────────┘            ║
║                                                  ║
║  [▶ START TRADING]  [⏹ STOP]  [⚙ Settings]     ║
║   #00CC33 (green)   #ff4444   #1a1a1a           ║
║   text: black       text:wht  text: #00FF41     ║
║                                                  ║
╚══════════════════════════════════════════════════╝
```

---

## 9. Decision Log

| # | Câu hỏi | Quyết định | Lý do |
|---|---------|-----------|-------|
| 1 | Phạm vi redesign | Theme-only + button redesign | Giảm rủi ro, giữ nguyên kiến trúc đang hoạt động |
| 2 | Button color approach | Matrix + Semantic | Giữ UX quen thuộc (đỏ=nguy hiểm, xanh=OK) trong context Matrix |
| 3 | Font scope | Monospace toàn bộ GUI | Nhất quán terminal aesthetic, không mix font |
| 4 | Matrix Rain animation | Không animation | Ưu tiên hiệu năng, trading GUI cần nhẹ |
| 5 | Neon effects | Border + highlight accent | Tăng visual depth, highlight dữ liệu quan trọng |
| 6 | Light mode | Loại bỏ | Matrix theme chỉ hợp dark, simplify code |
