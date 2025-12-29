<template>
  <div
    class="custom-dropdown"
    :class="{ 'is-open': isOpen, 'is-disabled': disabled, 'has-left-icon': hasLeftIcon }"
    ref="rootRef"
    :data-testid="attrs['data-testid']"
  >
    <button
      ref="triggerRef"
      type="button"
      class="dropdown-trigger"
      @click="toggleDropdown"
      @keydown="handleTriggerKeydown"
      :class="{ 'is-focused': isOpen || isFocused }"
      :disabled="disabled"
      :aria-haspopup="'listbox'"
      :aria-expanded="isOpen ? 'true' : 'false'"
      :aria-controls="'dropdown-listbox-' + id"
      :aria-labelledby="'dropdown-label-' + id"
      :tabindex="disabled ? -1 : 0"
      :id="'dropdown-trigger-' + id"
    >
      <span :id="'dropdown-label-' + id" class="sr-only">{{ placeholder }}</span>
      <div class="dropdown-selected">
        <span v-if="selectedLabel" class="selected-text">{{ selectedLabel }}</span>
        <span v-else class="placeholder">{{ placeholder }}</span>
      </div>
      <svg
        class="dropdown-arrow"
        :class="{ 'is-open': isOpen }"
        width="12"
        height="12"
        viewBox="0 0 12 12"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
        focusable="false"
      >
        <path d="M6 9L1 4h10z" fill="currentColor" />
      </svg>
    </button>

    <Transition name="dropdown">
      <div
        v-if="isOpen"
        class="dropdown-menu"
        @click.stop
        :id="'dropdown-listbox-' + id"
        role="listbox"
        :aria-labelledby="'dropdown-label-' + id"
        ref="listboxRef"
        tabindex="-1"
      >
        <div class="dropdown-options">
          <div
            v-for="(option, index) in options"
            :key="getOptionValue(option, index)"
            class="dropdown-option"
            :id="getOptionId(index)"
            :class="{
              'is-selected': isSelected(option, index),
              'is-hovered': hoveredIndex === index
            }"
            role="option"
            :aria-selected="isSelected(option, index) ? 'true' : 'false'"
            @click="selectOption(option, index)"
            @mouseenter="setHoveredIndex(index)"
            @mouseleave="clearHoveredIndex"
            :tabindex="-1"
            :ref="el => setOptionRef(el, index)"
          >
            <span class="option-text">{{ getOptionLabel(option) }}</span>
            <span v-if="isSelected(option, index)" class="option-check">✓</span>
          </div>
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick, onMounted, onUnmounted, useAttrs } from 'vue'

const attrs = useAttrs()

const props = defineProps({
  modelValue: {
    type: [String, Number],
    default: null
  },
  options: {
    type: Array,
    required: true,
    validator(value) {
      if (!Array.isArray(value)) {
        // Not an array
        return false
      }
      // Allow primitive arrays (strings, numbers) and object arrays with at least label/value, but do not strictly enforce shape
      return value.every(
        (item) =>
          typeof item === 'string' ||
          typeof item === 'number' ||
          (
            typeof item === 'object' &&
            item !== null &&
            (
              (typeof item.label === 'string' || typeof item.label === 'number') ||
              (typeof item.value === 'string' || typeof item.value === 'number')
            )
          )
      )
    }
  },
  placeholder: {
    type: String,
    default: 'Select an option'
  },
  disabled: {
    type: Boolean,
    default: false
  },
  optionLabel: {
    type: String,
    default: 'label'
  },
  optionValue: {
    type: String,
    default: 'value'
  },
  hasLeftIcon: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['update:modelValue', 'change'])

const isOpen = ref(false)
const isFocused = ref(false)
const hoveredIndex = ref(null) // Tracks which option is focused (for arrow navigation)
const triggerRef = ref(null)
const listboxRef = ref(null)
const rootRef = ref(null)
const optionRefs = ref({})

// Provide a unique ID for ARIA attributes (important for multi-instance use)
let _id = 0
const id = `cdropdown${++_id + Math.floor(Math.random() * 100000)}`

// Compute selected option's label for rendering
const selectedLabel = computed(() => {
  if (props.modelValue === null || props.modelValue === undefined) return null

  const selectedOption = props.options.find((option, index) => {
    const value = getOptionValue(option, index)
    return value === props.modelValue
  })

  return selectedOption ? getOptionLabel(selectedOption) : null
})

function getOptionLabel(option) {
  if (typeof option === 'string' || typeof option === 'number') {
    return String(option)
  }
  return option[props.optionLabel] || option.value || option
}

function getOptionValue(option, index) {
  if (typeof option === 'string' || typeof option === 'number') {
    return option
  }
  return option[props.optionValue] !== undefined ? option[props.optionValue] : index
}

function isSelected(option, index) {
  const value = getOptionValue(option, index)
  return value === props.modelValue
}

function getOptionId(index) {
  return `dropdown-option-${id}-${index}`
}

function toggleDropdown() {
  if (props.disabled) return
  isOpen.value = !isOpen.value
  if (isOpen.value) {
    openDropdown()
  } else {
    closeDropdown()
  }
}

function openDropdown() {
  isOpen.value = true
  isFocused.value = true
  // Set hover to selected or first option
  hoveredIndex.value = getSelectedIndex()
  nextTick(() => {
    // Focus first option or selected option
    focusHoveredOption()
    scrollHoveredOptionIntoView()
  })
}

function closeDropdown() {
  isOpen.value = false
  isFocused.value = false
  hoveredIndex.value = null
  nextTick(() => {
    // Restore focus to trigger for seamless keyboard usage
    if (triggerRef.value) triggerRef.value.focus()
  })
}

function selectOption(option, index) {
  const value = getOptionValue(option, index)
  emit('update:modelValue', value)
  emit('change', value)
  closeDropdown()
}

// Focus helpers
function setHoveredIndex(index) {
  hoveredIndex.value = index
}

function clearHoveredIndex() {
  hoveredIndex.value = null
}

function getSelectedIndex() {
  return props.options.findIndex((option, index) =>
    getOptionValue(option, index) === props.modelValue
  )
}

function focusHoveredOption() {
  if (
    isOpen.value &&
    hoveredIndex.value !== null &&
    optionRefs.value[hoveredIndex.value]
  ) {
    optionRefs.value[hoveredIndex.value].focus()
  }
}

function scrollHoveredOptionIntoView() {
  if (
    isOpen.value &&
    hoveredIndex.value !== null &&
    optionRefs.value[hoveredIndex.value]
  ) {
    optionRefs.value[hoveredIndex.value].scrollIntoView({ block: 'nearest' })
  }
}

// Keyboard handling for the button/trigger
function handleTriggerKeydown(event) {
  if (props.disabled) return
  switch (event.key) {
    case ' ':
    case 'Enter':
      event.preventDefault()
      toggleDropdown()
      break
    case 'ArrowDown':
      event.preventDefault()
      if (!isOpen.value) {
        openDropdown()
      } else {
        moveHover(1)
        focusHoveredOption()
        scrollHoveredOptionIntoView()
      }
      break
    case 'ArrowUp':
      event.preventDefault()
      if (!isOpen.value) {
        openDropdown()
      } else {
        moveHover(-1)
        focusHoveredOption()
        scrollHoveredOptionIntoView()
      }
      break
    case 'Tab':
      // Let Tab work naturally
      closeDropdown()
      break
    default:
      // Could support type-ahead search here
      break
  }
}

// Keyboard handling for the dropdown menu/options
function handleOptionKeydown(event, index) {
  switch (event.key) {
    case 'ArrowDown':
      event.preventDefault()
      moveHover(1)
      focusHoveredOption()
      scrollHoveredOptionIntoView()
      break
    case 'ArrowUp':
      event.preventDefault()
      moveHover(-1)
      focusHoveredOption()
      scrollHoveredOptionIntoView()
      break
    case 'Enter':
    case ' ':
      event.preventDefault()
      selectOption(props.options[hoveredIndex.value], hoveredIndex.value)
      break
    case 'Tab':
      closeDropdown()
      break
    case 'Escape':
      event.preventDefault()
      closeDropdown()
      break
    default:
      break
  }
}

function moveHover(direction) {
  let idx = hoveredIndex.value
  if (idx === null || idx === undefined) idx = -1
  const optionsLength = props.options.length
  let newIdx = idx + direction
  if (newIdx < 0) newIdx = optionsLength - 1
  else if (newIdx >= optionsLength) newIdx = 0
  hoveredIndex.value = newIdx
}

// Outside click and escape
function handleClickOutside(event) {
  // Only close if click is truly outside THIS dropdown instance
  if (rootRef.value && !rootRef.value.contains(event.target)) {
    closeDropdown()
  }
}

function handleDocumentKeydown(event) {
  // Only act if menu is open; most key handling is local to components
  if (
    event.key === 'Escape' &&
    isOpen.value
  ) {
    closeDropdown()
  }
}

// Manage list of optionRefs for focus control; optionRefs is an array of DOM nodes
onMounted(() => {
  document.addEventListener('click', handleClickOutside)
  document.addEventListener('keydown', handleDocumentKeydown)
})
onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
  document.removeEventListener('keydown', handleDocumentKeydown)
})

watch(() => props.modelValue, () => {
  // Reset hover when value changes
  hoveredIndex.value = getSelectedIndex()
})

// Handle keyboard navigation on each option
function setOptionRef(el, index) {
  if (el) {
    optionRefs.value[index] = el
  } else {
    // Remove ref when element is unmounted
    delete optionRefs.value[index]
  }
}

// Accessibility: ensure optionRefs matches option count
watch(
  () => props.options.length,
  () => {
    // Clean up refs if option length changed
    optionRefs.value = {}
  }
)

</script>

<!-- Additional styles for focus, visually hidden label, etc -->
<style scoped>
.sr-only {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  padding: 0 !important;
  overflow: hidden !important;
  clip: rect(1px, 1px, 1px, 1px) !important;
  white-space: nowrap !important;
  border: 0 !important;
}

/* ...rest unchanged... */
.custom-dropdown {
  position: relative;
  width: 100%;
  flex-shrink: 0;
}

.dropdown-trigger {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  padding: 0.625rem 2.5rem 0.625rem 1rem;
  background-color: rgba(55, 65, 81, 0.5);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(107, 114, 128, 0.5);
  border-radius: 0.5rem;
  color: white;
  cursor: pointer;
  transition: all 0.3s ease;
  min-height: 2.5rem;
  gap: 0.5rem;
  font-size: 0.875rem;
}

/* Khi có icon bên ngoài, tăng padding-left */
.custom-dropdown.has-left-icon .dropdown-trigger {
  padding-left: 2.5rem;
}

.dropdown-trigger:hover,
.dropdown-trigger:focus-visible {
  background-color: rgba(55, 65, 81, 0.6);
  border-color: rgba(139, 92, 246, 0.5);
  outline: none;
}

.dropdown-trigger.is-focused,
.dropdown-trigger:focus {
  background-color: rgba(55, 65, 81, 0.7);
  border-color: rgba(139, 92, 246, 0.8);
  box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2);
  outline: none;
}

.custom-dropdown.is-disabled .dropdown-trigger {
  opacity: 0.5;
  cursor: not-allowed;
}

.dropdown-selected {
  flex: 1;
  display: flex;
  align-items: center;
  min-width: 0;
  overflow: hidden;
}

.selected-text {
  color: white;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 0.875rem;
  max-width: 100%;
}

.placeholder {
  color: rgba(156, 163, 175, 1);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 0.875rem;
  max-width: 100%;
}

.dropdown-arrow {
  color: white;
  transition: transform 0.3s ease;
  flex-shrink: 0;
  position: absolute;
  right: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  pointer-events: none;
  width: 12px;
  height: 12px;
}

.dropdown-arrow.is-open {
  transform: translateY(-50%) rotate(180deg);
}

.dropdown-menu {
  position: absolute;
  top: calc(100% + 0.5rem);
  left: 0;
  right: 0;
  z-index: 50;
  background: rgba(20, 20, 30, 0.5);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 0.5rem;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
  overflow: hidden;
  max-height: 300px;
  overflow-y: auto;
}

.dropdown-options {
  padding: 0.25rem 0;
}

.dropdown-option {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1rem;
  color: white;
  cursor: pointer;
  transition: all 0.2s ease;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  border-left: 3px solid transparent;
}
.dropdown-option:focus {
  outline: none;
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.6) 0%, rgba(139, 92, 246, 0.6) 100%);
}

.dropdown-option:first-child {
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.dropdown-option:last-child {
  border-bottom: none;
}

.dropdown-option.is-hovered {
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.5) 0%, rgba(139, 92, 246, 0.5) 100%);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-left-color: rgba(139, 92, 246, 0.8);
}

.dropdown-option.is-selected {
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.7) 0%, rgba(139, 92, 246, 0.7) 100%);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-left-color: rgba(139, 92, 246, 1);
  font-weight: 600;
}

.option-text {
  flex: 1;
}

.option-check {
  color: rgba(139, 92, 246, 1);
  font-weight: bold;
  margin-left: 0.5rem;
}

/* Transition animations */
.dropdown-enter-active,
.dropdown-leave-active {
  transition: all 0.3s ease;
}

.dropdown-enter-from {
  opacity: 0;
  transform: translateY(-10px);
}

.dropdown-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

/* Custom scrollbar for dropdown menu */
.dropdown-menu::-webkit-scrollbar {
  width: 6px;
}

.dropdown-menu::-webkit-scrollbar-track {
  background: rgba(20, 20, 30, 0.3);
}

.dropdown-menu::-webkit-scrollbar-thumb {
  background: rgba(139, 92, 246, 0.5);
  border-radius: 3px;
}

.dropdown-menu::-webkit-scrollbar-thumb:hover {
  background: rgba(139, 92, 246, 0.7);
}
</style>

