<template>
  <div class="relative" :class="{ 'w-full': fullWidth }">
    <span v-if="icon" class="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 z-10 pointer-events-none">
      {{ icon }}
    </span>
    <input :id="attrs.id" :value="localValue" :type="type" :placeholder="placeholder" :disabled="disabled" :min="min"
      :max="max" :step="step" @input="handleInput" @blur="handleBlur" :class="computedClasses" />
    <span v-if="rightIcon"
      class="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 pointer-events-none">
      {{ rightIcon }}
    </span>
  </div>
</template>

<script setup>
import { computed, useAttrs, ref, watch } from 'vue'

const attrs = useAttrs()

const props = defineProps({
  modelValue: {
    type: [String, Number],
    default: ''
  },
  type: {
    type: String,
    default: 'text'
  },
  placeholder: {
    type: String,
    default: ''
  },
  disabled: {
    type: Boolean,
    default: false
  },
  icon: {
    type: String,
    default: ''
  },
  rightIcon: {
    type: String,
    default: ''
  },
  min: {
    type: [String, Number],
    default: null
  },
  max: {
    type: [String, Number],
    default: null
  },
  step: {
    type: [String, Number],
    default: null
  },
  hasError: {
    type: Boolean,
    default: false
  },
  fullWidth: {
    type: Boolean,
    default: false
  }
})

// Local state for smooth typing without re-renders
const localValue = ref('')

// Sync localValue with prop changes from parent
watch(() => props.modelValue, (newVal) => {
  localValue.value = newVal
}, { immediate: true })

const emit = defineEmits(['update:modelValue', 'input', 'blur'])

const computedClasses = computed(() => {
  return [
    props.fullWidth ? 'w-full' : '',
    'px-4 py-3 bg-gray-700/50 border border-gray-600/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 backdrop-blur-sm transition-all duration-300',
    props.hasError ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : '',
    props.icon ? 'pl-10' : '',
    props.rightIcon ? 'pr-10' : ''
  ].filter(c => c).join(' ')
})

function handleInput(event) {
  // Update local value for smooth typing
  localValue.value = event.target.value
  emit('input', event)
}

function handleBlur(event) {
  // Only emit to parent on blur to prevent re-render during typing
  const value = event.target.value
  emit('update:modelValue', value)
  emit('blur', event)
}
</script>
