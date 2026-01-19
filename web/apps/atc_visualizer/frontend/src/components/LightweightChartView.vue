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
      <!-- SVG Overlay for Signal Markers -->
      <svg ref="markerOverlay" class="marker-overlay" :width="chartWidth" :height="chartHeight">
        <g v-for="marker in renderedMarkers" :key="marker.id">
          <!-- Buy Signal (Arrow Up) -->
          <path v-if="marker.position === 'belowBar'" :d="getArrowUpPath(marker.x, marker.y)" :fill="marker.color"
            class="signal-marker" />
          <!-- Sell Signal (Arrow Down) -->
          <path v-else :d="getArrowDownPath(marker.x, marker.y)" :fill="marker.color" class="signal-marker" />
        </g>
      </svg>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  ColorType,
  createChart,
  CandlestickSeries,
  LineSeries,
  LineStyle,
  IChartApi,
  ISeriesApi,
  ITimeScaleApi,
  Time,
  CandlestickData,
  LineData
} from 'lightweight-charts'
import { onMounted, onUnmounted, ref, watch } from 'vue'

interface OHLCVData {
  x: number
  y: number[]
}

interface Point {
  x: number
  y: number
}

interface MarkerData {
  id: string
  time: number
  position: 'belowBar' | 'aboveBar'
  color: string
  value: number
  x: number
  y: number
  price?: number
}

interface Props {
  ohlcv: OHLCVData[]
  movingAverages?: Record<string, Point[]> | null
  signals: Record<string, Point[]>
  visibleMas: Record<string, boolean>
  showSignals: Record<string, boolean>
  symbol: string
  timeframe: string
}

const props = defineProps<Props>()

const chartContainer = ref<HTMLElement | null>(null)
const legendRef = ref<HTMLElement | null>(null)
const markerOverlay = ref<SVGElement | null>(null)

// SVG Overlay state
const chartWidth = ref(800)
const chartHeight = ref(600)
const renderedMarkers = ref<MarkerData[]>([])

// Sử dụng biến bình thường, không dùng ref cho các đối tượng API của library
// để tránh Vue can thiệp vào Prototype của chúng
let chartInstance: IChartApi | null = null
let candleSeriesInstance: ISeriesApi<"Candlestick"> | null = null
const maSeriesMap = new Map<string, ISeriesApi<"Line">>()

// Time and price scale for coordinate conversion
let timeScale: ITimeScaleApi<Time> | null = null
let priceScale: any = null

const MA_COLORS: Record<string, string> = {
  EMA: '#00E396',
  HMA: '#FEB019',
  WMA: '#775DD0',
  DEMA: '#008FFB',
  LSMA: '#FF4560',
  KAMA: '#775DD0'
}

const transformOHLCV = (data: OHLCVData[]): CandlestickData<Time>[] => {
  if (!data || !Array.isArray(data)) return [];
  const transformed = data.map(d => ({
    time: Math.floor(d.x / 1000) as Time,
    open: d.y[0],
    high: d.y[1],
    low: d.y[2],
    close: d.y[3]
  })).sort((a, b) => (a.time as number) - (b.time as number));
  // Loại bỏ các nến trùng timestamp
  return transformed.filter((item, index, self) => index === 0 || (item.time as number) > (self[index - 1].time as number));
}

const transformLineData = (data: Point[]): LineData<Time>[] => {
  if (!data || !Array.isArray(data)) return [];
  const transformed = data.map(d => ({
    time: Math.floor(d.x / 1000) as Time,
    value: d.y
  })).sort((a, b) => (a.time as number) - (b.time as number));
  return transformed.filter((item, index, self) => index === 0 || (item.time as number) > (self[index - 1].time as number));
}

// ===== Coordinate Conversion Utilities =====

/**
 * Convert timestamp to X pixel coordinate
 */
const timeToPixel = (timestamp: number) => {
  if (!timeScale) return 0
  try {
    // Try timeToCoordinate first (v5 API)
    // @ts-ignore
    if (typeof timeScale.timeToCoordinate === 'function') {
      // @ts-ignore
      return timeScale.timeToCoordinate(timestamp as Time) || 0
    }
    // Fallback to logicalToCoordinate
    // @ts-ignore
    if (typeof timeScale.logicalToCoordinate === 'function') {
      // @ts-ignore
      const logical = timeScale.timeToLogical(timestamp as Time)
      // @ts-ignore
      return timeScale.logicalToCoordinate(logical) || 0
    }
    return 0
  } catch (e) {
    console.error('Error in timeToPixel:', e)
    return 0
  }
}

/**
 * Convert price to Y pixel coordinate
 * Fully manual calculation using OHLCV data
 */
const priceToPixel = (price: number) => {
  if (!props.ohlcv || props.ohlcv.length === 0 || !price || price === 0) {
    return 0
  }

  try {
    // Find min and max prices from visible OHLCV data
    let minPrice = Infinity
    let maxPrice = -Infinity

    props.ohlcv.forEach(candle => {
      const low = candle.y[2]
      const high = candle.y[1]
      if (low < minPrice) minPrice = low
      if (high > maxPrice) maxPrice = high
    })

    if (minPrice === Infinity || maxPrice === -Infinity) {
      return 0
    }

    const priceSpan = maxPrice - minPrice

    if (priceSpan === 0) {
      return chartHeight.value / 2
    }

    // Calculate Y position (inverted because canvas Y increases downward)
    // Top of chart = maxPrice, Bottom = minPrice
    const normalizedPrice = (price - minPrice) / priceSpan
    const y = chartHeight.value * (1 - normalizedPrice)

    return y

  } catch (e) {
    console.error('Error in priceToPixel:', e)
    return 0
  }
}

/**
 * Get price at marker time for vertical positioning
 */
const getPriceAtTime = (timestamp: number, position: 'belowBar' | 'aboveBar') => {
  if (!props.ohlcv || props.ohlcv.length === 0) {
    console.warn('OHLCV data not available')
    return 0
  }

  // Try exact match first
  let candle = props.ohlcv.find(d => Math.floor(d.x / 1000) === timestamp)

  // If no exact match, find nearest candle
  if (!candle) {
    const nearest = props.ohlcv.reduce((prev, curr) => {
      const prevDiff = Math.abs(Math.floor(prev.x / 1000) - timestamp)
      const currDiff = Math.abs(Math.floor(curr.x / 1000) - timestamp)
      return currDiff < prevDiff ? curr : prev
    })
    candle = nearest

    if (Math.abs(Math.floor(candle.x / 1000) - timestamp) > 300) {
      // More than 5 minutes difference, skip
      return 0
    }
  }

  // Position below bar (at low) for buy signals, above bar (at high) for sell
  const price = position === 'belowBar' ? candle.y[2] : candle.y[1]
  return price
}

// ===== SVG Path Generators =====

/**
 * Generate SVG path for upward arrow (Buy signal)
 * Positioned below the candle low
 */
const getArrowUpPath = (x: number, y: number) => {
  const width = 10
  const height = 12
  const gap = 3 // Small gap from candle

  // Arrow points up towards the candle
  return `
    M ${x},${y + gap}
    L ${x - width / 2},${y + gap + height}
    L ${x - width / 3},${y + gap + height}
    L ${x - width / 3},${y + gap + height + 4}
    L ${x + width / 3},${y + gap + height + 4}
    L ${x + width / 3},${y + gap + height}
    L ${x + width / 2},${y + gap + height}
    Z
  `
}

/**
 * Generate SVG path for downward arrow (Sell signal)
 * Positioned above the candle high
 */
const getArrowDownPath = (x: number, y: number) => {
  const width = 10
  const height = 12
  const gap = 3

  // Arrow points down towards the candle
  return `
    M ${x},${y - gap}
    L ${x - width / 2},${y - gap - height}
    L ${x - width / 3},${y - gap - height}
    L ${x - width / 3},${y - gap - height - 4}
    L ${x + width / 3},${y - gap - height - 4}
    L ${x + width / 3},${y - gap - height}
    L ${x + width / 2},${y - gap - height}
    Z
  `
}

const initChart = () => {
  if (!chartContainer.value) return

  try {
    const container = chartContainer.value

    // Đảm bảo container có kích thước
    if (container.clientWidth === 0 || container.clientHeight === 0) {
      setTimeout(initChart, 100)
      return
    }

    // Cleanup old instance
    if (chartInstance) {
      chartInstance.remove()
      chartInstance = null
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

    if (!chartInstance) throw new Error('Failed to create chart instance')

    console.log('Chart created. Validating APIs:', {
      addSeries: typeof chartInstance.addSeries,
      CandlestickSeries: !!CandlestickSeries
    })

    // Robust Series Creation
    // @ts-ignore
    if (typeof chartInstance.addCandlestickSeries === 'function') {
      // @ts-ignore
      candleSeriesInstance = chartInstance.addCandlestickSeries({
        upColor: '#00E396',
        downColor: '#FF4560',
        borderVisible: false,
        wickUpColor: '#00E396',
        wickDownColor: '#FF4560',
      })
    } else if (CandlestickSeries) {
      // Fallback to Constructor-based API (v3/v4 style)
      // @ts-ignore
      candleSeriesInstance = chartInstance.addSeries(CandlestickSeries, {
        upColor: '#00E396',
        downColor: '#FF4560',
        borderVisible: false,
        wickUpColor: '#00E396',
        wickDownColor: '#FF4560',
      })
    } else {
      throw new Error('Example: Cannot create series - addCandlestickSeries missing and CandlestickSeries constructor not imported')
    }

    console.log('Candlestick series created.')

    // Capture scales for coordinate conversion
    timeScale = chartInstance.timeScale()
    priceScale = candleSeriesInstance.priceScale()

    // Subscribe to chart changes for marker updates
    chartInstance.timeScale().subscribeVisibleTimeRangeChange(() => {
      updateMarkers()
    })

    // Resize handling
    const resizeObserver = new ResizeObserver(entries => {
      if (chartInstance && entries[0]) {
        const { width, height } = entries[0].contentRect
        chartWidth.value = width
        chartHeight.value = height
        chartInstance.applyOptions({ width, height })
        updateMarkers()
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
      if (chartInstance) chartInstance.removeSeries(series)
    } catch (e) { }
  })
  maSeriesMap.clear()

  if (!props.movingAverages) return

  const maTypes = ['EMA', 'HMA', 'WMA', 'DEMA', 'LSMA', 'KAMA']
  maTypes.forEach(maType => {
    if (!props.visibleMas[maType]) return
    const color = MA_COLORS[maType]
    const baseKey = `${maType}_MA`
    if (props.movingAverages![baseKey]) {
      addLineSeries(baseKey, props.movingAverages![baseKey], color, 2)
    }
    for (let i = 1; i <= 4; i++) {
      const posKey = `${maType}_MA${i}`
      const negKey = `${maType}_MA_${i}`
      if (props.movingAverages![posKey]) addLineSeries(posKey, props.movingAverages![posKey], color, 1, LineStyle.Dotted)
      if (props.movingAverages![negKey]) addLineSeries(negKey, props.movingAverages![negKey], color, 1, LineStyle.Dotted)
    }
  })
}

const addLineSeries = (id: string, data: Point[], color: string, lineWidth: number, lineStyle?: LineStyle) => {
  if (!chartInstance) return

  let series: ISeriesApi<"Line">;
  // @ts-ignore
  if (typeof chartInstance.addLineSeries === 'function') {
    // @ts-ignore
    series = chartInstance.addLineSeries({
      color,
      lineWidth: lineWidth as any,
      // @ts-ignore
      lineStyle: lineStyle || LineStyle.Solid,
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false
    })
  } else if (LineSeries) {
    // @ts-ignore
    series = chartInstance.addSeries(LineSeries, {
      color,
      lineWidth,
      lineStyle: lineStyle || LineStyle.Solid,
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false
    })
  } else {
    console.error('Cannot create line series - LineSeries constructor missing')
    return;
  }

  series.setData(transformLineData(data))
  maSeriesMap.set(id, series)
}

const updateMarkers = () => {
  if (!candleSeriesInstance || !props.signals || !chartInstance) return
  if (!timeScale || !priceScale) {
    console.warn('Scales not ready, skipping marker update')
    return
  }

  console.log('=== Rendering Custom Marker Overlay ===')

  try {
    const markers: MarkerData[] = []
    Object.entries(props.showSignals).forEach(([signalType, show]) => {
      if (show && props.signals[signalType]) {
        console.log(`Processing ${signalType}: ${props.signals[signalType].length} points`)
        props.signals[signalType].forEach((point, index) => {
          const time = Math.floor(point.x / 1000)
          const threshold = signalType === 'Average_Signal' ? 0.2 : 0.5

          if (point.y >= threshold) {
            markers.push({
              id: `${signalType}-${time}-${index}`,
              time,
              position: 'belowBar',
              color: '#00E396',
              value: point.y
            })
          } else if (point.y <= -threshold) {
            markers.push({
              id: `${signalType}-${time}-${index}`,
              time,
              position: 'aboveBar',
              color: '#FF4560',
              value: point.y
            })
          }
        })
      }
    })

    console.log(`Total markers created: ${markers.length}`)

    // Convert markers to pixel coordinates
    const markersWithCoords = markers.map(marker => {
      const price = getPriceAtTime(marker.time, marker.position)
      const x = timeToPixel(marker.time)
      const y = priceToPixel(price)

      return {
        ...marker,
        x,
        y,
        price // for debugging
      }
    })

    // Debug: Log first few markers with coordinates
    console.log('First 3 markers with coords:', markersWithCoords.slice(0, 3).map(m => ({
      time: m.time,
      price: m.price,
      x: m.x,
      y: m.y,
      position: m.position
    })))

    // Filter valid markers (allow negative X for off-screen markers)
    renderedMarkers.value = markersWithCoords.filter(m => {
      // Only require valid Y coordinate and reasonable X (can be negative or beyond width)
      return m.y! > 0 && m.y! < chartHeight.value && !isNaN(m.x!) && !isNaN(m.y!)
    })

    console.log(`✅ Rendered ${renderedMarkers.value.length} markers on SVG overlay`)
    if (renderedMarkers.value.length > 0) {
      console.log('Sample coordinates:', renderedMarkers.value.slice(0, 3).map(m => `(${Math.round(m.x!)}, ${Math.round(m.y!)})`))
    }

  } catch (e) {
    console.error('Error rendering markers:', e)
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

.chart-header h2 {
  color: #667eea;
  font-size: 1.5em;
  font-weight: 600;
  margin: 0;
}

.legend {
  color: #e0e0e0;
  font-family: monospace;
}

.chart-container {
  position: relative;
  flex: 1;
  min-height: 600px;
  width: 100%;
}

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

.marker-overlay {
  position: absolute;
  top: 0;
  left: 0;
  pointer-events: none;
  z-index: 10;
}

.signal-marker {
  opacity: 0.85;
  transition: opacity 0.2s;
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
}
</style>
