<template>
  <div class="signal-legend">
    <div class="legend-section">
      <h3>Moving Averages</h3>
      <div class="legend-grid">
        <label
          v-for="(visible, maType) in visibleMas"
          :key="maType"
          class="legend-item"
        >
          <input
            type="checkbox"
            v-model="visibleMas[maType]"
            :style="{ accentColor: getMAColor(maType) }"
          />
          <span :style="{ color: getMAColor(maType) }">{{ maType }}</span>
        </label>
      </div>
    </div>

    <div class="legend-section">
      <h3>Signals</h3>
      <div class="legend-grid">
        <label
          v-for="(visible, signalType) in showSignals"
          :key="signalType"
          class="legend-item"
        >
          <input
            type="checkbox"
            v-model="showSignals[signalType]"
            :style="{ accentColor: getSignalColor(signalType) }"
          />
          <span :style="{ color: getSignalColor(signalType) }">{{ signalType }}</span>
        </label>
      </div>
    </div>

    <div class="legend-section">
      <h3>Signal Markers</h3>
      <div class="legend-items-inline">
        <div class="legend-item-inline">
          <span class="arrow-up">↑</span>
          <span>Buy Signal (1)</span>
        </div>
        <div class="legend-item-inline">
          <span class="arrow-down">↓</span>
          <span>Sell Signal (-1)</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  visibleMas: {
    type: Object,
    required: true
  },
  showSignals: {
    type: Object,
    required: true
  }
})

const getMAColor = (maType) => {
  const colors = {
    EMA: '#00E396',
    HMA: '#FEB019',
    WMA: '#775DD0',
    DEMA: '#008FFB',
    LSMA: '#FF4560',
    KAMA: '#775DD0'
  }
  return colors[maType] || '#888'
}

const getSignalColor = (signalType) => {
  if (signalType === 'Average_Signal') return '#FFFFFF'
  const colors = {
    EMA_Signal: '#00E396',
    HMA_Signal: '#FEB019',
    WMA_Signal: '#775DD0',
    DEMA_Signal: '#008FFB',
    LSMA_Signal: '#FF4560',
    KAMA_Signal: '#775DD0'
  }
  return colors[signalType] || '#888'
}
</script>

<style scoped>
.signal-legend {
  background-color: #16213e;
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.legend-section {
  margin-bottom: 20px;
}

.legend-section:last-child {
  margin-bottom: 0;
}

.legend-section h3 {
  font-size: 1.1em;
  font-weight: 600;
  color: #667eea;
  margin-bottom: 12px;
}

.legend-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 10px;
}

.legend-items-inline {
  display: flex;
  gap: 30px;
  flex-wrap: wrap;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 5px;
  border-radius: 4px;
  transition: background-color 0.2s ease;
}

.legend-item:hover {
  background-color: rgba(102, 126, 234, 0.1);
}

.legend-item input[type="checkbox"] {
  width: 18px;
  height: 18px;
  cursor: pointer;
}

.legend-item span {
  font-size: 0.9em;
  font-weight: 500;
}

.legend-item-inline {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.9em;
}

.arrow-up {
  color: #00E396;
  font-size: 1.5em;
  font-weight: bold;
}

.arrow-down {
  color: #FF4560;
  font-size: 1.5em;
  font-weight: bold;
}
</style>
