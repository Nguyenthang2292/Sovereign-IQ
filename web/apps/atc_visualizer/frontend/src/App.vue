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

      <LightweightChartView
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
import { ref, reactive, shallowRef } from 'vue'
import ParameterPanel from './components/ParameterPanel.vue'
import SignalLegend from './components/SignalLegend.vue'
import LightweightChartView from './components/LightweightChartView.vue'

let symbol = ref('BTC/USDT')
let timeframe = ref('15m')
let limit = ref(1500)

const loading = ref(false)
const error = ref(null)
const dataLoaded = ref(false)

// Use shallowRef for large datasets to improve performance
const ohlcvData = shallowRef(null)
const movingAveragesData = shallowRef(null)
const signalsData = shallowRef(null)

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
      fetch(`/api/ohlcv?${params}`).catch(err => {
        console.error('Error fetching OHLCV:', err)
        return { ok: false, status: 500, json: async () => ({ success: false, error: `Network error: ${err.message}` }) }
      }),
      fetch(`/api/atc-signals?${params}`).catch(err => {
        console.error('Error fetching ATC signals:', err)
        return { ok: false, status: 500, json: async () => ({ success: false, error: `Network error: ${err.message}` }) }
      }),
      fetch(`/api/moving-averages?${params}`).catch(err => {
        console.error('Error fetching moving averages:', err)
        return { ok: false, status: 500, json: async () => ({ success: false, error: `Network error: ${err.message}` }) }
      })
    ])

    // Check if responses are OK
    if (!ohlcvRes.ok) {
      const errorData = await ohlcvRes.json().catch(() => ({ error: `HTTP ${ohlcvRes.status}: Failed to load OHLCV data` }))
      throw new Error(errorData.error || errorData.detail || `HTTP ${ohlcvRes.status}: Failed to load OHLCV data`)
    }

    if (!signalsRes.ok) {
      const errorData = await signalsRes.json().catch(() => ({ error: `HTTP ${signalsRes.status}: Failed to load ATC signals` }))
      throw new Error(errorData.error || errorData.detail || `HTTP ${signalsRes.status}: Failed to load ATC signals`)
    }

    if (!masRes.ok) {
      const errorData = await masRes.json().catch(() => ({ error: `HTTP ${masRes.status}: Failed to load moving averages` }))
      throw new Error(errorData.error || errorData.detail || `HTTP ${masRes.status}: Failed to load moving averages`)
    }

    const ohlcvDataResult = await ohlcvRes.json()
    const signalsDataResult = await signalsRes.json()
    const masDataResult = await masRes.json()

    if (!ohlcvDataResult.success) {
      throw new Error(ohlcvDataResult.error || ohlcvDataResult.detail || 'Failed to load OHLCV data')
    }

    if (!signalsDataResult.success) {
      throw new Error(signalsDataResult.error || signalsDataResult.detail || 'Failed to load ATC signals')
    }

    // Correct data extraction based on backend response structure
    ohlcvData.value = signalsDataResult.data.ohlcv || ohlcvDataResult.data.data
    signalsData.value = signalsDataResult.data.signals
    movingAveragesData.value = masDataResult.success ? masDataResult.data : null

    // Debug log for signals
    console.log('=== ATC Signals Debug ===')
    console.log('Signal keys:', Object.keys(signalsData.value || {}))
    if (signalsData.value && signalsData.value.Average_Signal) {
      console.log('Average_Signal sample:', signalsData.value.Average_Signal.slice(0, 5))
      const strongSignals = signalsData.value.Average_Signal.filter(p => Math.abs(p.y) >= 0.5)
      console.log(`Average_Signal points with |y| >= 0.5: ${strongSignals.length} / ${signalsData.value.Average_Signal.length}`)
      if (strongSignals.length > 0) {
        console.log('Sample strong signals:', strongSignals.slice(0, 3))
      }
    } else {
      console.warn('Average_Signal not found in response!')
    }

    dataLoaded.value = true
  } catch (err) {
    console.error('Error loading data:', err)
    error.value = err.message || 'An unexpected error occurred. Please check if the backend server is running on port 8002.'
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
