<template>
  <div class="chart-view">
    <div class="chart-header">
      <h2>{{ symbol }} - {{ timeframe }}</h2>
      <div class="legend" ref="legendRef"></div>
    </div>

    <div class="chart-container" ref="chartContainer">
      <div v-if="!ohlcv || ohlcv.length === 0" class="no-data">
        No OHLCV data available for the selected parameters.
      </div>
    </div>
  </div>
</template>

<script setup>
import { ColorType, createChart, CandlestickSeries, LineSeries, LineStyle } from 'lightweight-charts'
import { onMounted, onUnmounted, ref, watch } from 'vue'

const props = defineProps({
  ohlcv: {
    type: Array,
    required: true
  },
  movingAverages: {
    type: Object,
    default: null
  },
  signals: {
    type: Object,
    required: true
  },
  visibleMas: {
    type: Object,
    required: true
  },
  showSignals: {
    type: Object,
    required: true
  },
  symbol: {
    type: String,
    required: true
  },
  timeframe: {
    type: String,
    required: true
  }
})

const chartContainer = ref(null)
const legendRef = ref(null)

// Sử dụng biến bình thường, không dùng ref cho các đối tượng API của library
// để tránh Vue can thiệp vào Prototype của chúng
let chartInstance = null
let candleSeriesInstance = null
const maSeriesMap = new Map()

const MA_COLORS = {
  EMA: '#00E396',
  HMA: '#FEB019',
  WMA: '#775DD0',
  DEMA: '#008FFB',
  LSMA: '#FF4560',
  KAMA: '#775DD0'
}

const transformOHLCV = (data) => {
  if (!data || !Array.isArray(data)) return [];
  const transformed = data.map(d => ({
    time: Math.floor(d.x / 1000),
    open: d.y[0],
    high: d.y[1],
    low: d.y[2],
    close: d.y[3]
  })).sort((a, b) => a.time - b.time);
  // Loại bỏ các nến trùng timestamp
  return transformed.filter((item, index, self) => index === 0 || item.time > self[index - 1].time);
}

const transformLineData = (data) => {
  if (!data || !Array.isArray(data)) return [];
  const transformed = data.map(d => ({
    time: Math.floor(d.x / 1000),
    value: d.y
  })).sort((a, b) => a.time - b.time);
  return transformed.filter((item, index, self) => index === 0 || item.time > self[index - 1].time);
}

const initChart = () => {
  if (!chartContainer.value) return

  try {
    const container = chartContainer.value
    
    // Đảm bảo container có kích thước
    if (container.clientWidth === 0 || container.clientHeight === 0) {
      // Đợi container render xong
      setTimeout(initChart, 100)
      return
    }
    
    // Khởi tạo chart
    chartInstance = createChart(container, {
      width: container.clientWidth || 800,
      height: container.clientHeight || 600,
      layout: {
        background: { type: ColorType.Solid, color: '#16213e' },
        textColor: '#8892b0',
      },
      grid: {
        vertLines: { color: '#2B2B43' },
        horzLines: { color: '#2B2B43' },
      },
      timeScale: {
        borderColor: '#485c7b',
        timeVisible: true,
      },
    })

    // Kiểm tra chartInstance đã được khởi tạo đúng
    if (!chartInstance || typeof chartInstance.addSeries !== 'function') {
      console.error('Chart instance is not properly initialized or addSeries is not available')
      throw new Error('Failed to initialize chart: addSeries method not found')
    }

    // Khởi tạo Candlestick Series
    candleSeriesInstance = chartInstance.addSeries(CandlestickSeries, {
      upColor: '#00E396',
      downColor: '#FF4560',
      borderVisible: false,
      wickUpColor: '#00E396',
      wickDownColor: '#FF4560',
    })

    // Resize handling
    const resizeObserver = new ResizeObserver(entries => {
      if (chartInstance && entries[0]) {
        const { width, height } = entries[0].contentRect
        chartInstance.applyOptions({ width, height })
      }
    })
    resizeObserver.observe(container)

    updateChartData()
  } catch (err) {
    console.error('Error during chart initialization:', err)
  }
}

const updateChartData = () => {
  if (!candleSeriesInstance || !props.ohlcv) return
  candleSeriesInstance.setData(transformOHLCV(props.ohlcv))
  updateMovingAverages()
  updateMarkers()
}

const updateMovingAverages = () => {
  if (!chartInstance) return
  
  maSeriesMap.forEach(series => {
    try {
      chartInstance.removeSeries(series)
    } catch (e) {}
  })
  maSeriesMap.clear()

  if (!props.movingAverages) return

  const maTypes = ['EMA', 'HMA', 'WMA', 'DEMA', 'LSMA', 'KAMA']
  maTypes.forEach(maType => {
    if (!props.visibleMas[maType]) return
    const color = MA_COLORS[maType]
    const baseKey = `${maType}_MA`
    if (props.movingAverages[baseKey]) {
      addLineSeries(baseKey, props.movingAverages[baseKey], color, 2)
    }
    for (let i = 1; i <= 4; i++) {
      const posKey = `${maType}_MA${i}`
      const negKey = `${maType}_MA_${i}`
      if (props.movingAverages[posKey]) addLineSeries(posKey, props.movingAverages[posKey], color, 1, LineStyle.Dotted)
      if (props.movingAverages[negKey]) addLineSeries(negKey, props.movingAverages[negKey], color, 1, LineStyle.Dotted)
    }
  })
}

const addLineSeries = (id, data, color, lineWidth, lineStyle) => {
  if (!chartInstance) return
  const series = chartInstance.addSeries(LineSeries, {
    color,
    lineWidth,
    lineStyle: lineStyle || LineStyle.Solid,
    crosshairMarkerVisible: false,
    lastValueVisible: false,
    priceLineVisible: false
  })
  series.setData(transformLineData(data))
  maSeriesMap.set(id, series)
}

const updateMarkers = () => {
  if (!candleSeriesInstance || !props.signals) return
  console.log('Signals available:', Object.keys(props.signals))
  console.log('Show signals config:', props.showSignals)
  try {
    const markers = []
    Object.entries(props.showSignals).forEach(([signalType, show]) => {
      if (show && props.signals[signalType]) {
        console.log(`Processing signal: ${signalType}, points: ${props.signals[signalType].length}`)
        props.signals[signalType].forEach(point => {
          const time = Math.floor(point.x / 1000)
          // Lower threshold for Average_Signal since it's an average of multiple signals
          const threshold = signalType === 'Average_Signal' ? 0.2 : 0.5
          if (point.y >= threshold) {
            markers.push({ time, position: 'belowBar', color: '#00E396', shape: 'arrowUp', text: 'B' })
          } else if (point.y <= -threshold) {
            markers.push({ time, position: 'aboveBar', color: '#FF4560', shape: 'arrowDown', text: 'S' })
          }
        })
      } else {
        console.log(`Signal ${signalType} not shown - show: ${show}, exists: ${!!props.signals[signalType]}`)
      }
    })
    console.log(`Total markers created: ${markers.length}`)
    markers.sort((a, b) => a.time - b.time)
    const uniqueMarkers = markers.filter((item, index, self) => index === 0 || item.time !== self[index - 1].time)
    console.log(`Unique markers: ${uniqueMarkers.length}`)
    if (typeof candleSeriesInstance.setMarkers === 'function') {
      candleSeriesInstance.setMarkers(uniqueMarkers)
    }
  } catch (e) {
    console.warn('Markers not supported in this version:', e)
  }
}

watch(() => props.ohlcv, updateChartData, { deep: false })
watch(() => props.movingAverages, updateMovingAverages, { deep: false })
watch(() => [props.visibleMas, props.showSignals], () => {
  updateMovingAverages()
  updateMarkers()
}, { deep: true })

onMounted(initChart)
onUnmounted(() => {
  if (chartInstance) {
    chartInstance.remove()
    chartInstance = null
  }
})
</script>

<style scoped>
.chart-view {
  background-color: #16213e;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
}
.chart-header {
  margin-bottom: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.chart-header h2 { color: #667eea; font-size: 1.5em; font-weight: 600; margin: 0; }
.legend { color: #e0e0e0; font-family: monospace; }
.chart-container { position: relative; flex: 1; min-height: 600px; width: 100%; }
.no-data {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 400px;
  color: #8892b0;
  font-size: 1.2em;
  background-color: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
}
</style>
