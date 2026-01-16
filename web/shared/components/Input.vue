<template>
  <div class="relative" :class="{ 'w-full': fullWidth }">
    <span
      v-if="icon"
      class="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 z-10 pointer-events-none"
    >
      {{ icon }}
    </span>
    <input
      v-model="localValue"
      :type="type"
      :placeholder="placeholder"
      :disabled="disabled"
      :min="min"
      :max="max"
      :step="step"
      @input="handleInput"
      @blur="handleBlur"
      :class="inputClasses"
    />
    <span
      v-if="rightIcon"
      class="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 pointer-events-none"
    >
      {{ rightIcon }}
    </span>
  </div>
</template>

<script setup>
import { computed } from 'vue'

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

const emit = defineEmits(['update:modelValue', 'input', 'blur'])

const localValue = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value)
})

const inputClasses = computed(() => {
  const classes = [
    'px-4 py-3 bg-gray-700/50 border rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 backdrop-blur-sm transition-all duration-300',
    props.fullWidth ? 'w-full' : ''
  ]

  if (props.hasError) {
    classes.push('border-red-500 focus:ring-red-500 focus:border-red-500')
  } else {
    classes.push('border-gray-600/50 focus:ring-purple-500 focus:border-purple-500')
  }

  if (props.icon) {
    classes.push('pl-10')
  }

  return classes.join(' ')
})

function handleInput(event) {
  let value = event.target.value
  // Convert to number if type is number
  if (props.type === 'number' && value !== '') {
    const numValue = Number(value)
    if (!isNaN(numValue)) {
      value = numValue
    }
  }
  emit('update:modelValue', value)
  emit('input', event)
}

function handleBlur(event) {
  emit('blur', event)
}
</script>
