<template>
  <div class="parameter-panel">
    <div class="panel-title">Chart Parameters</div>

    <div class="param-grid">
      <div class="param-group">
        <label for="symbol">Symbol</label>
        <input
          id="symbol"
          v-model="localSymbol"
          type="text"
          placeholder="BTC/USDT"
          @keyup.enter="handleLoad"
        />
      </div>

      <div class="param-group">
        <label for="timeframe">Timeframe</label>
        <select id="timeframe" v-model="localTimeframe">
          <option value="1m">1 Minute</option>
          <option value="5m">5 Minutes</option>
          <option value="15m">15 Minutes</option>
          <option value="30m">30 Minutes</option>
          <option value="1h">1 Hour</option>
          <option value="4h">4 Hours</option>
          <option value="1d">1 Day</option>
        </select>
      </div>

      <div class="param-group">
        <label for="limit">Number of Candles</label>
        <input
          id="limit"
          v-model.number="localLimit"
          type="number"
          min="100"
          max="5000"
          step="100"
        />
      </div>

      <div class="param-group button-group">
        <button
          @click="handleLoad"
          :disabled="loading"
          class="load-button"
        >
          <span v-if="!loading">Load Data</span>
          <span v-else>Loading...</span>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
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
</script>

<style scoped>
.parameter-panel {
  background-color: #16213e;
  border-radius: 10px;
  padding: 25px;
  margin-bottom: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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

.param-group input,
.param-group select {
  background-color: #0f3460;
  border: 1px solid #1a1a2e;
  border-radius: 6px;
  padding: 12px 15px;
  color: #e0e0e0;
  font-size: 1em;
  transition: all 0.3s ease;
}

.param-group input:focus,
.param-group select:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.param-group button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  border-radius: 6px;
  padding: 12px 25px;
  color: white;
  font-size: 1em;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
}

.param-group button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
}

.param-group button:active:not(:disabled) {
  transform: translateY(0);
}

.param-group button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.button-group {
  justify-content: flex-end;
}
</style>
