<template>
  <button
    @click="handleClick"
    :disabled="disabled || loading"
    :class="buttonClasses"
  >
    <span v-if="loading" class="flex items-center justify-center gap-2">
      <svg class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>
      <span v-if="loadingText">{{ loadingText }}</span>
      <span v-else>Loading...</span>
    </span>
    <span v-else class="flex items-center justify-center gap-2">
      <slot></slot>
    </span>
  </button>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  disabled: {
    type: Boolean,
    default: false
  },
  loading: {
    type: Boolean,
    default: false
  },
  variant: {
    type: String,
    default: 'primary',
    validator: (value) => ['primary', 'secondary', 'danger', 'success'].includes(value)
  },
  loadingText: {
    type: String,
    default: ''
  },
  fullWidth: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['click'])

function handleClick(event) {
  if (!props.disabled && !props.loading) {
    emit('click', event)
  }
}

const buttonClasses = computed(() => {
  const classes = [
    'px-6 py-3 rounded-lg font-semibold text-white transition-all duration-300 flex items-center justify-center gap-2',
    props.fullWidth ? 'w-full' : ''
  ]

  if (props.disabled || props.loading) {
    classes.push('bg-gray-600/50 cursor-not-allowed border border-gray-600/50')
  } else {
    switch (props.variant) {
      case 'primary':
        classes.push('btn-gradient hover:shadow-glow-purple hover:scale-[1.02] active:scale-[0.98]')
        break
      case 'secondary':
        classes.push('bg-gray-700/50 text-gray-300 hover:bg-gray-600/50 border border-gray-600/50')
        break
      case 'danger':
        classes.push('bg-gradient-to-r from-red-600 to-red-500 hover:from-red-700 hover:to-red-600 hover:shadow-lg hover:scale-[1.02] active:scale-[0.98]')
        break
      case 'success':
        classes.push('bg-gradient-to-r from-green-600 to-green-500 hover:from-green-700 hover:to-green-600 hover:shadow-lg hover:scale-[1.02] active:scale-[0.98]')
        break
    }
  }

  return classes
})
</script>
