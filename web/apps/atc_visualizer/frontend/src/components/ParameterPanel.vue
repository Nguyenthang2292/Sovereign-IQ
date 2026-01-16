<template>
  <div class="glass-panel">
    <div class="panel-title">Chart Parameters</div>

    <div class="param-grid">
      <div class="param-group">
        <label for="symbol">Symbol</label>
        <Input
          id="symbol"
          v-model="localSymbol"
          type="text"
          placeholder="BTC/USDT"
          @keyup.enter="handleLoad"
          icon="💵"
          fullWidth
        />
      </div>

      <div class="param-group">
        <label for="timeframe">Timeframe</label>
        <CustomDropdown
          id="timeframe"
          v-model="localTimeframe"
          :options="timeframeOptions"
          placeholder="Select timeframe"
        />
      </div>

      <div class="param-group">
        <label for="limit">Number of Candles</label>
        <Input
          id="limit"
          v-model.number="localLimit"
          type="number"
          min="100"
          max="5000"
          step="100"
          icon="📊"
          placeholder="Enter limit"
          fullWidth
        />
      </div>

      <div class="param-group button-group">
        <Button
          @click="handleLoad"
          :disabled="loading"
          :loading="loading"
          loadingText="Loading..."
          variant="primary"
          fullWidth
        >
          Load Data
        </Button>
      </div>
    </div>
  </div>
</template>

<script setup>
import Button from '@shared/components/Button.vue'
import CustomDropdown from '@shared/components/CustomDropdown.vue'
import Input from '@shared/components/Input.vue'
import { computed } from 'vue'

const props = defineProps({
  symbol: {
    type: String,
    required: true
  },
  timeframe: {
    type: String,
    required: true
  },
  limit: {
    type: Number,
    required: true
  },
  loading: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['update:symbol', 'update:timeframe', 'update:limit', 'load-data'])

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
  set: (value) => emit('update:limit', value)
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
