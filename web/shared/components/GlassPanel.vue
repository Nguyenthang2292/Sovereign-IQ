<template>
  <div
    class="glass-panel rounded-xl p-4 md:p-6"
    :class="[paddingClass, { [borderClass]: highlighted }]"
  >
    <slot></slot>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  padding: {
    type: String,
    default: 'md',
    validator: (value) => ['sm', 'md', 'lg', 'xl'].includes(value)
  },
  highlighted: {
    type: Boolean,
    default: false
  }
})

const paddingClass = computed(() => {
  const paddings = {
    sm: 'p-4',
    md: 'p-4 md:p-6',
    lg: 'p-6 md:p-8',
    xl: 'p-8 md:p-10'
  }
  return paddings[props.padding] || paddings.md
})

const borderClass = computed(() => {
  return props.highlighted ? 'border-purple-500 shadow-lg shadow-purple-500/50' : 'border-gray-700/50'
})
</script>

<style scoped>
.glass-panel {
  background: rgba(20, 20, 30, 0.5);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
}
</style>
