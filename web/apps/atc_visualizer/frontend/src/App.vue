<template>
  <div class="app">
    <div class="container">
      <header class="app-header">
        <h1>ATC Visualizer</h1>
        <p>Adaptive Trend Classification - Chart Visualization</p>
      </header>

      <ParameterPanel
        v-model:symbol="symbol"
        v-model:timeframe="timeframe"
        v-model:limit="limit"
        :loading="loading"
        @load-data="loadData"
      />

      <div v-if="error" class="error-message">
        {{ error }}
      </div>

      <SignalLegend
        v-model:visible-mas="visibleMas"
        v-model:show-signals="showSignals"
      />

      <ChartView
        v-if="dataLoaded"
        :ohlcv="ohlcvData"
        :moving-averages="movingAveragesData"
        :signals="signalsData"
        :visible-mas="visibleMas"
        :show-signals="showSignals"
        :symbol="symbol"
        :timeframe="timeframe"
      />

      <div v-if="!dataLoaded && !loading" class="empty-state">
        <p>Enter symbol and timeframe to load data</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import ParameterPanel from './components/ParameterPanel.vue'
import SignalLegend from './components/SignalLegend.vue'
import ChartView from './components/ChartView.vue'

let symbol = ref('BTC/USDT')
let timeframe = ref('15m')
let limit = ref(1500)

const loading = ref(false)
const error = ref(null)
const dataLoaded = ref(false)

const ohlcvData = ref(null)
const movingAveragesData = ref(null)
const signalsData = ref(null)

let visibleMas = reactive({
  EMA: true,
  HMA: true,
  WMA: false,
  DEMA: false,
  LSMA: false,
  KAMA: false
})

let showSignals = reactive({
  Average_Signal: true,
  EMA_Signal: false,
  HMA_Signal: false,
  WMA_Signal: false,
  DEMA_Signal: false,
  LSMA_Signal: false,
  KAMA_Signal: false
})

const loadData = async () => {
  loading.value = true
  error.value = null
  dataLoaded.value = false

  try {
    const params = new URLSearchParams({
      symbol: symbol.value,
      timeframe: timeframe.value,
      limit: limit.value
    })

    const [ohlcvRes, signalsRes, masRes] = await Promise.all([
      fetch(`/api/ohlcv?${params}`),
      fetch(`/api/atc-signals?${params}`),
      fetch(`/api/moving-averages?${params}`)
    ])

    const ohlcvDataResult = await ohlcvRes.json()
    const signalsDataResult = await signalsRes.json()
    const masDataResult = await masRes.json()

    if (!ohlcvDataResult.success) {
      throw new Error(ohlcvDataResult.error || 'Failed to load OHLCV data')
    }

    if (!signalsDataResult.success) {
      throw new Error(signalsDataResult.error || 'Failed to load ATC signals')
    }

    // Correct data extraction based on backend response structure
    ohlcvData.value = signalsDataResult.data.ohlcv || ohlcvDataResult.data.data
    signalsData.value = signalsDataResult.data.signals
    movingAveragesData.value = masDataResult.success ? masDataResult.data : null

    dataLoaded.value = true
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.app-header {
  text-align: center;
  margin-bottom: 30px;
  padding: 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.app-header h1 {
  color: white;
  font-size: 2.5em;
  margin-bottom: 10px;
}

.app-header p {
  color: rgba(255, 255, 255, 0.9);
  font-size: 1.1em;
}

.error-message {
  background-color: #e74c3c;
  color: white;
  padding: 15px;
  border-radius: 8px;
  margin: 20px 0;
  text-align: center;
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
  background-color: #16213e;
  border-radius: 10px;
  margin: 20px 0;
}

.empty-state p {
  color: #8892b0;
  font-size: 1.2em;
}
</style>
