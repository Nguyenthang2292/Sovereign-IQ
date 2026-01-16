# Shared UI Components

Đây là thư viện component UI chia sẻ cho tất cả các app frontend trong dự án.

## Cấu trúc

```
web/shared/components/
├── index.js           # Export tất cả các component
├── Button.vue          # Button với các variant và loading state
├── Checkbox.vue        # Checkbox với styling đồng bộ
├── CustomDropdown.vue  # Custom dropdown với keyboard navigation
├── GlassPanel.vue      # Panel với glass morphism effect
├── Input.vue           # Input với icon support
└── LoadingSpinner.vue  # Loading spinner
```

## Sử dụng

### 1. Cấu hình alias trong vite.config.js

```javascript
import path from 'path'

export default defineConfig({
  resolve: {
    alias: {
      '@shared': path.resolve(__dirname, '../../../shared')
    }
  }
})
```

### 2. Import component

```javascript
import { Button, Input, Checkbox, CustomDropdown, GlassPanel } from '@shared/components'
// Hoặc import từng component
import Button from '@shared/components/Button.vue'
```

### 3. Sử dụng component

#### Button

```vue
<Button
  :loading="loading"
  :disabled="!isValid"
  variant="primary"
  fullWidth
  @click="handleClick"
>
  Load Data
</Button>
```

**Props:**
- `loading` (Boolean): Hiển thị spinner khi loading
- `disabled` (Boolean): Disable button
- `variant` (String): 'primary' | 'secondary' | 'danger' | 'success'
- `fullWidth` (Boolean): Button full width
- `loadingText` (String): Text hiển thị khi loading

#### Input

```vue
<Input
  v-model="symbol"
  type="text"
  placeholder="BTC/USDT"
  icon="💵"
  fullWidth
  :has-error="error"
  @input="handleInput"
/>
```

**Props:**
- `modelValue` (String|Number): Giá trị input
- `type` (String): Loại input (text, number, etc.)
- `placeholder` (String): Placeholder text
- `disabled` (Boolean): Disable input
- `icon` (String): Emoji icon hiển thị bên trái
- `rightIcon` (String): Emoji icon hiển thị bên phải
- `min`/`max`/`step` (Number|String): Number input constraints
- `hasError` (Boolean): Hiển thị error state
- `fullWidth` (Boolean): Input full width

#### Checkbox

```vue
<Checkbox
  v-model="checked"
  @change="handleChange"
>
  Show MA Lines
</Checkbox>
```

**Props:**
- `modelValue` (Boolean): Giá trị checkbox
- `disabled` (Boolean): Disable checkbox

#### CustomDropdown

```vue
<CustomDropdown
  v-model="timeframe"
  :options="timeframeOptions"
  placeholder="Select timeframe"
  :has-left-icon="true"
  option-label="label"
  option-value="value"
/>
```

**Props:**
- `modelValue` (String|Number): Giá trị được chọn
- `options` (Array): Mảng options (string, number, hoặc object)
- `placeholder` (String): Placeholder text
- `disabled` (Boolean): Disable dropdown
- `hasLeftIcon` (Boolean): Có space cho icon bên trái
- `optionLabel` (String): Key để lấy label từ object (default: 'label')
- `optionValue` (String): Key để lấy value từ object (default: 'value')

**Example options:**
```javascript
const timeframeOptions = [
  { value: '1m', label: '1 Minute' },
  { value: '5m', label: '5 Minutes' },
  { value: '1h', label: '1 Hour' }
]
// hoặc
const simpleOptions = ['1m', '5m', '1h']
```

#### GlassPanel

```vue
<GlassPanel padding="md" :highlighted="true">
  <h3>Content</h3>
  <p>Panel content here</p>
</GlassPanel>
```

**Props:**
- `padding` (String): 'sm' | 'md' | 'lg' | 'xl'
- `highlighted` (Boolean): Highlight panel với border purple

## Tailwind Configuration

Các app sử dụng shared components cần cấu hình Tailwind:

```javascript
// tailwind.config.js
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
    "../../../shared/**/*.{vue,js,ts,jsx,tsx}", // Thêm dòng này
  ],
  // ... rest of config
}
```

## Styling

Tất cả component sử dụng:
- **Glass morphism effect** với `backdrop-filter: blur(20px)`
- **Purple accent color** `#8b5cf6`
- **Dark theme** compatible
- **Tailwind CSS** utility classes
- **Responsive design** với `md:` breakpoint

## Tạo mới component

1. Tạo file component mới trong `web/shared/components/`
2. Export trong `web/shared/components/index.js`
3. Sử dụng Tailwind CSS và Glass effect style
4. Đảm bảo accessibility với proper ARIA attributes
5. Test với cả light và dark theme (nếu có)
