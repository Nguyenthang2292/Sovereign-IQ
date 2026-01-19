<template>
  <div class="chart-view">
    <div class="chart-header">
      <h2>{{ symbol }} - {{ timeframe }}</h2>
    </div>

    <div class="chart-container">
      <div v-if="!ohlcv || ohlcv.length === 0" class="no-data">
        No OHLCV data available for the selected parameters.
      </div>
      <apexchart v-else ref="chart" type="candlestick" :options="chartOptions" :series="chartSeries" :height="800" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface OHLCVData {
  x: number
  y: number[]
}

interface Point {
  x: number
  y: number
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

const chart = ref(null)

const getMAColor = (maType: string): string => {
  const colors: Record<string, string> = {
    EMA: '#00E396',
    HMA: '#FEB019',
    WMA: '#775DD0',
    DEMA: '#008FFB',
    LSMA: '#FF4560',
    KAMA: '#775DD0'
  }
  return colors[maType] || '#888'
}

const getSignalColor = (signalType: string): string => {
  if (signalType === 'Average_Signal') return '#FFFFFF'
  const colors: Record<string, string> = {
    EMA_Signal: '#00E396',
    HMA_Signal: '#FEB019',
    WMA_Signal: '#775DD0',
    DEMA_Signal: '#008FFB',
    LSMA_Signal: '#FF4560',
    KAMA_Signal: '#775DD0'
  }
  return colors[signalType] || '#888'
}

const getSignalMarkers = (signalData: Point[]) => {
  if (!signalData || !props.ohlcv) return []

  const markers: any[] = []
  // Create a map for quick price lookup by timestamp
  const priceMap = new Map<number, number[]>()
  props.ohlcv.forEach(d => {
    priceMap.set(d.x, d.y) // d.y is [O, H, L, C]
  })

  signalData.forEach(point => {
    const prices = priceMap.get(point.x)
    if (!prices) return

    if (point.y >= 0.5) {
      // Buy signal - place below Low
      markers.push({
        x: point.x,
        y: prices[2], // Low
        marker: {
          size: 8,
          shape: 'triangle-up',
          fillColor: '#00E396',
          strokeColor: '#00E396',
          radius: 2,
        }
      })
    } else if (point.y <= -0.5) {
      // Sell signal - place above High
      markers.push({
        x: point.x,
        y: prices[1], // High
        marker: {
          size: 8,
          shape: 'triangle-down',
          fillColor: '#FF4560',
          strokeColor: '#FF4560',
          radius: 2,
        }
      })
    }
  })

  return markers
}

const chartSeries = computed(() => {
  const series: any[] = []

  series.push({
    name: 'Candlestick',
    type: 'candlestick',
    data: props.ohlcv
  })

  if (props.movingAverages) {
    const maTypes = ['EMA', 'HMA', 'WMA', 'DEMA', 'LSMA', 'KAMA']

    maTypes.forEach(maType => {
      if (props.visibleMas[maType]) {
        const baseKey = `${maType}_MA`

        if (props.movingAverages![baseKey]) {
          series.push({
            name: baseKey,
            type: 'line',
            data: props.movingAverages![baseKey],
            tooltip: {
              enabled: false // Tắt tooltip cho MA để tăng tốc
            }
          })
        }

        for (let i = 1; i <= 4; i++) {
          const posKey = `${maType}_MA${i}`
          const negKey = `${maType}_MA_${i}`

          if (props.movingAverages![posKey]) {
            series.push({
              name: posKey,
              type: 'line',
              data: props.movingAverages![posKey],
              tooltip: {
                enabled: false
              }
            })
          }

          if (props.movingAverages![negKey]) {
            series.push({
              name: negKey,
              type: 'line',
              data: props.movingAverages![negKey],
              tooltip: {
                enabled: false
              }
            })
          }
        }
      }
    })
  }

  if (props.signals) {
    Object.entries(props.showSignals).forEach(([signalType, show]) => {
      if (show && props.signals[signalType]) {
        series.push({
          name: signalType,
          type: 'scatter',
          data: getSignalMarkers(props.signals[signalType])
        })
      }
    })
  }

  return series
})

const chartOptions = computed(() => {
  const seriesNames = chartSeries.value.map(s => s.name)

  return {
    chart: {
      type: 'candlestick',
      height: 800,
      background: '#1a1a2e',
      toolbar: {
        show: true,
        tools: {
          zoom: true,
          zoomin: true,
          zoomout: true,
          pan: true,
          reset: true
        },
        autoSelected: 'zoom'
      },
      animations: {
        enabled: false
      }
    },
    theme: {
      mode: 'dark'
    },
    title: {
      text: `${props.symbol} - ${props.timeframe}`,
      align: 'left',
      style: {
        fontSize: '20px',
        fontWeight: 'bold',
        color: '#e0e0e0'
      }
    },
    plotOptions: {
      candlestick: {
        colors: {
          upward: '#00E396',
          downward: '#FF4560'
        },
        wick: {
          useFillColor: true
        }
      }
    },
    xaxis: {
      type: 'datetime',
      axisBorder: {
        show: true,
        color: '#475467'
      },
      axisTicks: {
        show: true,
        color: '#475467'
      },
      labels: {
        datetimeFormatter: {
          year: 'yyyy',
          month: 'MMM \'yy',
          day: 'dd MMM',
          hour: 'HH:mm'
        },
        style: {
          colors: '#8892b0'
        }
      },
      tooltip: {
        enabled: true,
        theme: 'dark'
      }
    },
    yaxis: {
      labels: {
        style: {
          colors: '#8892b0'
        }
      }
    },
    grid: {
      borderColor: '#475467',
      row: {
        colors: ['#1a1a2e', '#1a1a2e'],
        opacity: 1
      }
    },
    legend: {
      show: true,
      position: 'top',
      horizontalAlign: 'right',
      labels: {
        colors: '#e0e0e0'
      },
      markers: {
        size: 6,
        strokeColor: '#1a1a2e',
        strokeWidth: 2
      },
      onItemHover: {
        highlightDataSeries: false
      }
    },
    stroke: {
      width: 1.5, // Giảm độ dày để vẽ nhanh hơn
      curve: 'straight' // 'straight' nhanh hơn 'smooth' rất nhiều
    },
    colors: [
      '#00E396', // Candlestick
      // Màu cho các MA sẽ được ApexCharts tự động lặp lại hoặc lấy từ theme
    ],
    dataLabels: {
      enabled: false
    },
    tooltip: {
      theme: 'dark',
      shared: true, // Gộp tooltip lại
      followCursor: false, // Giảm lag khi di chuột
      x: {
        format: 'dd MMM HH:mm'
      }
    },
    markers: {
      size: 0, // Tắt markers mặc định cho các đường line
      hover: {
        size: 0
      }
    }
  }
})

// Xóa bỏ các watch thủ công vì vue3-apexcharts đã tự động watch :series prop
</script>

<style scoped>
.chart-view {
  background-color: #16213e;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.chart-header {
  margin-bottom: 20px;
}

.chart-header h2 {
  color: #667eea;
  font-size: 1.5em;
  font-weight: 600;
}

.chart-container {
  position: relative;
  min-height: 800px;
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

:deep(.apexcharts-tooltip) {
  background: #16213e !important;
  border-color: #667eea !important;
  color: #e0e0e0 !important;
}

:deep(.apexcharts-text) {
  fill: #8892b0 !important;
}

:deep(.apexcharts-gridline) {
  stroke: #475467 !important;
}

:deep(.apexcharts-axis-line) {
  stroke: #475467 !important;
}

:deep(.apexcharts-legend-text) {
  color: #e0e0e0 !important;
}

:deep(.apexcharts-toolbar) {
  background-color: #1a1a2e !important;
  border: 1px solid #475467 !important;
}

:deep(.apexcharts-menu-icon) {
  background-color: transparent !important;
  color: #e0e0e0 !important;
}

:deep(.apexcharts-menu) {
  background: #1a1a2e !important;
  border: 1px solid #475467 !important;
}

:deep(.apexcharts-menu-item) {
  background: transparent !important;
  color: #e0e0e0 !important;
}

:deep(.apexcharts-menu-item:hover) {
  background: rgba(102, 126, 234, 0.1) !important;
}
</style>
