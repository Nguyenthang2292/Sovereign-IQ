/**
 * Type declarations for shared Vue components
 * Provides TypeScript support for shared components used across multiple apps
 */

// Use DefineComponent from Vue to provide proper component types
import type { DefineComponent } from 'vue'

// ============================================================================
// CustomDropdown Component
// ============================================================================
export interface CustomDropdownProps {
  modelValue?: string | number | null
  options: Array<string | number | Record<string, any>>
  placeholder?: string
  disabled?: boolean
  optionLabel?: string
  optionValue?: string
  hasLeftIcon?: boolean
}

declare module '@shared/components/CustomDropdown.vue' {
  const component: DefineComponent<CustomDropdownProps>
  export default component
}

// ============================================================================
// Input Component
// ============================================================================
export interface InputProps {
  modelValue?: string | number
  type?: string
  placeholder?: string
  disabled?: boolean
  icon?: string
  rightIcon?: string
  min?: string | number | null
  max?: string | number | null
  step?: string | number | null
  hasError?: boolean
  fullWidth?: boolean
}

declare module '@shared/components/Input.vue' {
  const component: DefineComponent<InputProps>
  export default component
}

// ============================================================================
// Button Component
// ============================================================================
export interface ButtonProps {
  disabled?: boolean
  loading?: boolean
  variant?: 'primary' | 'secondary' | 'danger' | 'success'
  loadingText?: string
  fullWidth?: boolean
}

declare module '@shared/components/Button.vue' {
  const component: DefineComponent<ButtonProps>
  export default component
}

// ============================================================================
// GlassPanel Component
// ============================================================================
export interface GlassPanelProps {
  variant?: 'default' | 'elevated' | 'subtle'
  padding?: 'none' | 'sm' | 'md' | 'lg'
}

declare module '@shared/components/GlassPanel.vue' {
  const component: DefineComponent<GlassPanelProps>
  export default component
}

// ============================================================================
// LoadingSpinner Component
// ============================================================================
export interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  color?: string
  text?: string
}

declare module '@shared/components/LoadingSpinner.vue' {
  const component: DefineComponent<LoadingSpinnerProps>
  export default component
}

// ============================================================================
// Checkbox Component
// ============================================================================
export interface CheckboxProps {
  modelValue?: boolean
  label?: string
  disabled?: boolean
  id?: string
}

declare module '@shared/components/Checkbox.vue' {
  const component: DefineComponent<CheckboxProps>
  export default component
}
