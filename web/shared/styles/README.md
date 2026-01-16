# Shared Styles Documentation

Hệ thống CSS tập trung cho tất cả các ứng dụng trong `web/apps/`.

## 📁 Cấu trúc

```
web/shared/styles/
├── variables.css    # CSS variables (colors, spacing, z-index, etc.)
├── base.css         # Reset, body, scrollbar, background layers
├── components.css   # UI components (glass panels, buttons, sidebar)
├── layouts.css      # Layout utilities, responsive breakpoints
├── effects.css      # Animations, neon effects, glow effects
└── README.md        # Tài liệu này
```

## 🎨 Sử dụng trong App

### Import vào app style.css

Trong file `style.css` của mỗi app (ví dụ: `web/apps/your_app/frontend/src/style.css`):

```css
/* Import shared styles */
@import url('../../../../../shared/styles/variables.css');
@import url('../../../../../shared/styles/base.css');
@import url('../../../../../shared/styles/components.css');
@import url('../../../../../shared/styles/layouts.css');
@import url('../../../../../shared/styles/effects.css');

/* App-specific styles below */
.your-custom-component {
  /* Your styles */
}
```

### Import riêng lẻ

Nếu chỉ cần một số file:

```css
/* Chỉ import variables và components */
@import url('../../../../../shared/styles/variables.css');
@import url('../../../../../shared/styles/components.css');
```

## 🧩 Các Components có sẵn

### Glassmorphism
```html
<div class="glass-panel">Content</div>
<nav class="glass-nav">Navigation</nav>
```

### Buttons
```html
<button class="btn-gradient">Click me</button>
```

### Sidebar (Full featured)
```html
<aside class="sidebar">
  <div class="sidebar-header">
    <div class="sidebar-logo">Logo</div>
    <h1>App Name</h1>
  </div>
  <nav class="sidebar-nav">
    <a href="#" class="sidebar-link active">
      <span class="sidebar-icon">🏠</span>
      <span class="sidebar-text">Home</span>
    </a>
  </nav>
  <div class="sidebar-footer">
    <button class="sidebar-footer-btn">Settings</button>
  </div>
</aside>
```

### Neon Text Effects
```html
<h1 class="neon-cyan">Cyan Glow</h1>
<h1 class="neon-purple">Purple Glow</h1>
<h1 class="neon-magenta">Magenta Glow</h1>
```

### Animations
```html
<div class="glow-effect">Glowing element</div>
<div class="fade-in">Fade in animation</div>
<div class="hover-lift">Lifts on hover</div>
```

## 🎨 CSS Variables

Sử dụng CSS variables trong code của bạn:

```css
.my-component {
  background: var(--color-glass-panel);
  border: 1px solid var(--color-border-medium);
  border-radius: var(--radius-md);
  padding: var(--spacing-lg);
  backdrop-filter: var(--blur-md);
  transition: all var(--transition-normal);
}
```

### Các biến quan trọng

**Colors:**
- `--color-bg-primary`, `--color-bg-secondary`
- `--color-text-primary`, `--color-text-secondary`
- `--color-blue`, `--color-purple`, `--color-cyan`
- `--color-border-light`, `--color-border-medium`

**Spacing:**
- `--spacing-xs` (0.25rem) đến `--spacing-3xl` (2rem)

**Border Radius:**
- `--radius-sm` (6px) đến `--radius-xl` (12px)

**Effects:**
- `--blur-sm` đến `--blur-xl`
- `--shadow-sm` đến `--shadow-xl`

**Z-Index:**
- `--z-background`, `--z-overlay`, `--z-content`
- `--z-sidebar`, `--z-sidebar-controls`

## 🔧 Tùy chỉnh cho từng App

### Override Variables

```css
/* Trong app style.css */
@import url('../../../../../shared/styles/variables.css');

/* Override colors */
:root {
  --color-bg-primary: #000000; /* Darker background */
  --color-purple: #9333ea;     /* Different purple */
}
```

### Override Components

```css
/* Override sidebar width */
.sidebar {
  width: 300px; /* Wider sidebar for this app */
}
```

### App-specific Background

```css
.app-background {
  background-image: url('./img/your-custom-bg.png');
  background-color: #123456; /* Fallback color */
}
```

## 📱 Responsive Breakpoints

Shared styles đã bao gồm responsive breakpoints:

- **Mobile:** `max-width: 767px`
- **Tablet:** `768px - 1024px`
- **Desktop:** `1025px+`

Sidebar tự động collapse trên mobile và tablet.

## ✅ Best Practices

1. **Luôn import `variables.css` đầu tiên** nếu dùng CSS variables
2. **Tránh override** các shared classes trừ khi thật sự cần thiết
3. **Sử dụng CSS variables** thay vì hardcode values
4. **Prefix app-specific classes** để tránh conflict: `.myapp-custom-btn`
5. **Test responsive** trên nhiều kích thước màn hình

## 🆕 Thêm App Mới

Khi tạo app mới:

1. Tạo file `style.css` trong `web/apps/your_app/frontend/src/`
2. Import shared styles:
```css
@import url('../../../../../shared/styles/variables.css');
@import url('../../../../../shared/styles/base.css');
@import url('../../../../../shared/styles/components.css');
@import url('../../../../../shared/styles/layouts.css');
@import url('../../../../../shared/styles/effects.css');
```
3. Thêm app-specific styles phía dưới

## 🐛 Troubleshooting

### CSS không load được
- Kiểm tra đường dẫn relative: `../../../../../shared/styles/`
- Đảm bảo file được build tool (Vite/Webpack) process

### Style bị override
- Kiểm tra thứ tự import CSS files
- Sử dụng CSS specificity cao hơn hoặc `!important` (chỉ khi cần thiết)

### Background không hiển thị
- Kiểm tra path đến file ảnh trong `.app-background`
- Override class trong app style.css với đường dẫn đúng

## 📝 Contribute

Khi thêm styles mới vào shared:

1. **Thêm vào file phù hợp**: Variables → variables.css, Components → components.css
2. **Document các class mới** trong README này
3. **Test trên tất cả apps** để đảm bảo không break
4. **Sử dụng CSS variables** thay vì hardcode
5. **Comment rõ ràng** trong CSS code

## 🎯 Examples

Xem implementation đầy đủ tại:
- `web/apps/gemini_analyzer/frontend/src/style.css`
- `web/apps/atc_visualizer/frontend/src/style.css`
