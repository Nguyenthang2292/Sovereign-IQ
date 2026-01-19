<template>
  <div class="glass-panel">
    <div class="panel-title">Chart Parameters</div>

    <div class="param-grid">
      <div class="param-group">
        <label for="symbol">Symbol</label>
        <Input id="symbol" v-model="localSymbol" type="text" placeholder="BTC/USDT" icon="💵" :full-width="true"
          @keyup.enter="handleLoad" />
      </div>

      <div class="param-group">
        <label for="timeframe">Timeframe</label>
        <div class="relative">
          <span
            class="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 z-10 pointer-events-none">⏰</span>
          <CustomDropdown id="timeframe" v-model="localTimeframe" :options="timeframeOptions"
            placeholder="Select timeframe" :has-left-icon="true" />
        </div>
      </div>

      <div class="param-group">
        <label for="limit">Number of Candles</label>
        <Input id="limit" v-model="localLimit" type="number" min="100" max="5000" step="100" placeholder="Enter limit"
          icon="📊" :full-width="true" />
      </div>

      <div class="param-group button-group">
        <Button @click="handleLoad" :disabled="loading || !symbol" :loading="loading" loading-text="Loading..."
          variant="primary" :full-width="true">
          <span v-if="!loading">📊</span>
          <span>Load Data</span>
        </Button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import CustomDropdown from '@shared/components/CustomDropdown.vue'
import Input from '@shared/components/Input.vue'
import Button from '@shared/components/Button.vue'
import { computed } from 'vue'

interface Props {
  symbol: string
  timeframe: string
  limit: number
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  loading: false
})

const emit = defineEmits<{
  (e: 'update:symbol', value: string): void
  (e: 'update:timeframe', value: string): void
  (e: 'update:limit', value: number): void
  (e: 'load-data'): void
}>()

const localSymbol = computed({
  get: () => props.symbol,
  set: (value) => emit('update:symbol', value)
})

const localTimeframe = computed({
  get: () => props.timeframe,
  set: (value) => emit('update:timeframe', value)
})

const localLimit = computed({
  get: () => props.limit,
  set: (value: any) => {
    // Allow empty string (user is clearing the field)
    if (value === '') {
      return
    }
    const numValue = Number(value)
    if (!isNaN(numValue) && numValue > 0) {
      emit('update:limit', numValue)
    }
  }
})

const handleLoad = () => {
  emit('load-data')
}

const timeframeOptions = computed(() => [
  { value: '1m', label: '1 Minute' },
  { value: '5m', label: '5 Minutes' },
  { value: '15m', label: '15 Minutes' },
  { value: '30m', label: '30 Minutes' },
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
  { value: '1d', label: '1 Day' }
])
</script>

<style scoped>
.glass-panel {
  position: relative;
  z-index: 20;
  background: rgba(20, 20, 30, 0.5);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  padding: 25px;
  margin-bottom: 20px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
}

.panel-title {
  font-size: 1.3em;
  font-weight: 600;
  color: #667eea;
  margin-bottom: 20px;
  border-bottom: 2px solid #667eea;
  padding-bottom: 10px;
}

.param-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  align-items: end;
}

.param-group {
  display: flex;
  flex-direction: column;
}

.param-group label {
  color: #8892b0;
  font-size: 0.9em;
  margin-bottom: 8px;
  font-weight: 500;
}

.button-group {
  justify-content: flex-end;
}
</style>
