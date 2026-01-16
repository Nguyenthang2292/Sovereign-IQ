<template>
  <label class="flex items-center gap-3 cursor-pointer group">
    <div class="relative">
      <input
        type="checkbox"
        :checked="modelValue"
        @change="handleChange"
        :disabled="disabled"
        class="peer appearance-none w-5 h-5 text-purple-600 bg-gray-700/50 border border-gray-600/50 rounded focus:ring-2 focus:ring-purple-500 transition-all duration-300 checked:bg-purple-600 checked:border-purple-600"
        :class="{ 'opacity-50 cursor-not-allowed': disabled }"
      />
      <svg
        v-if="modelValue"
        class="absolute left-0.5 top-0.5 w-4 h-4 text-white pointer-events-none"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M5 13l4 4L19 7"
        />
      </svg>
    </div>
    <span
      v-if="$slots.default"
      class="text-sm font-medium text-gray-300 group-hover:text-white transition-colors"
      :class="{ 'text-gray-500': disabled }"
    >
      <slot></slot>
    </span>
  </label>
</template>

<script setup>
const props = defineProps({
  modelValue: {
    type: Boolean,
    default: false
  },
  disabled: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['update:modelValue', 'change'])

function handleChange(event) {
  const value = event.target.checked
  emit('update:modelValue', value)
  emit('change', value)
}
</script>
