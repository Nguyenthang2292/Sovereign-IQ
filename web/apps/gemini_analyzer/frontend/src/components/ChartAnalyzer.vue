<template>
  <div class="max-w-7xl mx-auto p-4 md:p-6">
    <!-- Header -->
    <div class="mb-6 md:mb-8">
      <h1 class="text-3xl md:text-4xl font-bold text-white mb-2 flex items-center gap-3">
        <span class="text-4xl md:text-5xl">📊</span>
        <span>{{ $t('chartAnalyzer.title') }}</span>
      </h1>
      <p class="text-gray-300 md:text-gray-400 text-sm md:text-base">{{ $t('chartAnalyzer.subtitle') }}</p>
    </div>

    <!-- Mode Toggle -->
    <div class="glass-panel rounded-xl p-4 mb-6">
      <div class="flex gap-4">
        <button
          data-testid="mode-single-button"
          @click="mode = 'single'"
          :class="[
            'flex-1 px-4 md:px-6 py-3 rounded-lg font-semibold transition-all duration-300',
            mode === 'single'
              ? 'btn-gradient text-white hover:shadow-glow-purple hover:scale-[1.02] active:scale-[0.98]'
              : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50 border border-gray-600/50'
          ]"
        >
          {{ $t('common.singleTimeframe') }}
        </button>
        <button
          data-testid="mode-multi-button"
          @click="mode = 'multi'"
          :class="[
            'flex-1 px-4 md:px-6 py-3 rounded-lg font-semibold transition-all duration-300',
            mode === 'multi'
              ? 'btn-gradient text-white hover:shadow-glow-purple hover:scale-[1.02] active:scale-[0.98]'
              : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50 border border-gray-600/50'
          ]"
        >
          {{ $t('common.multiTimeframe') }}
        </button>
      </div>
    </div>

    <!-- Form -->
    <div class="glass-panel rounded-xl p-4 md:p-6 mb-6">
      <h2 class="text-xl md:text-2xl font-bold text-white mb-4 md:mb-6">{{ $t('chartAnalyzer.configTitle') }}</h2>
      
      <div class="space-y-6">
        <!-- Symbol Input -->
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
            <span>💰</span>
            <span>{{ $t('common.symbol') }} <span class="text-red-400">{{ $t('common.required') }}</span></span>
          </label>
          <div class="relative">
            <input
              data-testid="symbol-input"
              v-model="form.symbol"
              type="text"
              :placeholder="$t('chartAnalyzer.fields.symbolPlaceholder')"
              class="w-full px-4 py-3 pl-10 bg-gray-700/50 border border-gray-600/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 backdrop-blur-sm"
            />
            <span class="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">💵</span>
          </div>
        </div>

        <!-- Timeframe Input (Single Mode) -->
        <div v-if="mode === 'single'">
          <label class="block text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
            <span>🕐</span>
            <span>{{ $t('common.timeframe') }} <span class="text-red-400">{{ $t('common.required') }}</span></span>
          </label>
          <div class="relative">
            <span class="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 z-10 pointer-events-none">⏰</span>
            <CustomDropdown
              data-testid="timeframe-select"
              v-model="form.timeframe"
              :options="['15m', '30m', '1h', '4h', '1d', '1w']"
              :placeholder="$t('common.selectTimeframe')"
              :has-left-icon="true"
            />
          </div>
        </div>

        <!-- Timeframes Input (Multi Mode) -->
        <div v-if="mode === 'multi'">
          <label class="block text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
            <span>🕐</span>
            <span>{{ $t('common.timeframes') }} <span class="text-red-400">{{ $t('common.required') }}</span> {{ $t('common.commaSeparated') }}</span>
          </label>
          <div class="relative">
            <input
              data-testid="timeframes-input"
              v-model="form.timeframes"
              type="text"
              :placeholder="$t('chartAnalyzer.fields.example')"
              class="w-full px-4 py-3 pl-10 bg-gray-700/50 border border-gray-600/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 backdrop-blur-sm"
            />
            <span class="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">⏰</span>
          </div>
          <p class="mt-2 text-sm text-gray-400">{{ $t('common.example') }}: {{ $t('chartAnalyzer.fields.example') }}</p>
        </div>

        <!-- Indicators Configuration -->
        <div class="border-t border-gray-700 pt-6">
          <h3 class="text-lg font-semibold text-white mb-4">{{ $t('chartAnalyzer.fields.indicators') }}</h3>
          
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- MA Periods -->
            <div>
              <label class="block text-sm font-medium text-gray-300 mb-2">
                {{ $t('chartAnalyzer.fields.maPeriods') }}
              </label>
              <input
                v-model="form.indicators.maPeriods"
                type="text"
                :placeholder="$t('chartAnalyzer.fields.maPeriodsPlaceholder')"
                class="w-full px-4 py-2 bg-gray-700/50 border border-gray-600/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 backdrop-blur-sm"
              />
            </div>

            <!-- RSI Period -->
            <div>
              <label class="block text-sm font-medium text-gray-300 mb-2">
                {{ $t('chartAnalyzer.fields.rsiPeriod') }}
              </label>
              <input
                v-model.number="form.indicators.rsiPeriod"
                type="number"
                min="1"
                class="w-full px-4 py-2 bg-gray-700/50 border border-gray-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 backdrop-blur-sm"
              />
            </div>

            <!-- MACD -->
            <div class="flex items-center gap-3">
              <input
                v-model="form.indicators.enableMacd"
                type="checkbox"
                id="enable-macd"
                class="w-5 h-5 text-purple-600 bg-gray-700/50 border-gray-600/50 rounded focus:ring-purple-500"
              />
              <label for="enable-macd" class="text-sm font-medium text-gray-300">
                {{ $t('chartAnalyzer.fields.enableMacd') }}
              </label>
            </div>

            <!-- Bollinger Bands -->
            <div class="flex items-center gap-3">
              <input
                v-model="form.indicators.enableBb"
                type="checkbox"
                id="enable-bb"
                class="w-5 h-5 text-purple-600 bg-gray-700/50 border-gray-600/50 rounded focus:ring-purple-500"
              />
              <label for="enable-bb" class="text-sm font-medium text-gray-300">
                {{ $t('chartAnalyzer.fields.enableBb') }}
              </label>
            </div>
          </div>
        </div>

        <!-- Advanced Options -->
        <div class="border-t border-gray-700 pt-6">
          <details class="cursor-pointer">
            <summary class="text-lg font-semibold text-white mb-4">{{ $t('chartAnalyzer.fields.advancedOptions') }}</summary>
            
            <div class="mt-4 space-y-4">
              <div>
                <label class="block text-sm font-medium text-gray-300 mb-2">
                  {{ $t('chartAnalyzer.fields.promptType') }}
                </label>
                <CustomDropdown
                  v-model="form.promptType"
                  :options="promptTypeOptions"
                  option-label="label"
                  option-value="value"
                  :placeholder="$t('chartAnalyzer.fields.promptType')"
                />
              </div>

              <div v-if="form.promptType === 'custom'">
                <label class="block text-sm font-medium text-gray-300 mb-2">
                  {{ $t('chartAnalyzer.fields.customPrompt') }}
                </label>
                <textarea
                  v-model="form.customPrompt"
                  rows="4"
                  class="w-full px-4 py-2 bg-gray-700/50 border border-gray-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 backdrop-blur-sm"
                  :placeholder="$t('chartAnalyzer.fields.customPromptPlaceholder')"
                ></textarea>
              </div>

              <div>
                <label class="block text-sm font-medium text-gray-300 mb-2">
                  {{ $t('chartAnalyzer.fields.limit') }}
                </label>
                <input
                  v-model.number="form.limit"
                  type="number"
                  min="1"
                  class="w-full px-4 py-2 bg-gray-700/50 border border-gray-600/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 backdrop-blur-sm"
                />
              </div>
            </div>
          </details>
        </div>

        <!-- Submit Button -->
        <div class="pt-4">
          <button
            @click="handleAnalyze"
            :disabled="loading || !isFormValid"
            :class="[
              'w-full px-6 py-4 rounded-lg font-semibold text-white transition-all duration-300 flex items-center justify-center gap-2',
              loading || !isFormValid
                ? 'bg-gray-600/50 cursor-not-allowed border border-gray-600/50'
                : 'btn-gradient hover:shadow-glow-purple hover:scale-[1.02] active:scale-[0.98]'
            ]"
          >
            <span v-if="loading" class="flex items-center justify-center gap-2">
              <svg class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              {{ $t('chartAnalyzer.analyzing') }}
            </span>
            <span v-else>🔍 {{ $t('chartAnalyzer.startAnalyze') }}</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Progress Indicator -->
    <div v-if="loading" class="glass-panel rounded-xl p-4 md:p-6 mb-6">
      <div class="flex items-center gap-4 mb-4">
        <svg class="animate-spin h-8 w-8 text-purple-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <div class="flex-1">
          <h3 class="text-lg font-semibold text-white mb-1">{{ $t('chartAnalyzer.analyzingProgress') }}</h3>
          <p class="text-gray-400 text-sm">{{ $t('chartAnalyzer.analyzingDescription') }}</p>
        </div>
      </div>
      <div class="w-full bg-gray-700/50 rounded-full h-2 mb-4">
        <div class="bg-purple-600 h-2 rounded-full animate-pulse" style="width: 100%"></div>
      </div>
    </div>

    <!-- Logs Section (show when there are logs, even after completion) -->
    <div v-if="logs.length > 0" class="glass-panel rounded-xl p-4 md:p-6 mb-6">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold text-white">{{ $t('chartAnalyzer.logs') }}</h3>
        <button
          v-if="!loading"
          @click="logs = []"
          class="px-3 py-1 text-sm text-gray-400 hover:text-white bg-gray-700/50 hover:bg-gray-600/50 rounded-lg transition-colors backdrop-blur-sm border border-gray-600/50"
        >
          {{ $t('common.clear') }}
        </button>
      </div>
      <LogViewer :logs="logs" />
    </div>

    <!-- Error Message -->
    <div v-if="error" class="glass-panel border border-red-500/50 rounded-lg p-4 mb-6 bg-red-900/20">
      <p class="text-red-400 flex items-center gap-2">
        <span>⚠️</span>
        <span>{{ error }}</span>
      </p>
    </div>

    <!-- Results -->
    <div v-if="result" class="glass-panel rounded-xl p-4 md:p-6">
      <h2 class="text-2xl font-bold text-white mb-6">{{ $t('chartAnalyzer.resultsTitle') }}</h2>

      <!-- Single Timeframe Results -->
      <div v-if="mode === 'single' && result.analysis">
        <!-- Chart Image -->
        <div v-if="result.chart_url" class="mb-6">
          <img
            :src="result.chart_url"
            :alt="`Chart ${result.symbol} ${result.timeframe}`"
            class="w-full rounded-lg border border-gray-700"
          />
        </div>

        <!-- Signal Badge -->
        <div class="mb-6">
          <div class="flex items-center gap-4">
            <span
              :class="[
                'px-6 py-3 rounded-lg font-bold text-lg',
                result.signal === 'LONG'
                  ? 'bg-green-500/20 text-green-400 border-2 border-green-500'
                  : result.signal === 'SHORT'
                  ? 'bg-red-500/20 text-red-400 border-2 border-red-500'
                  : 'bg-gray-500/20 text-gray-400 border-2 border-gray-500'
              ]"
            >
              {{ result.signal || 'NONE' }}
            </span>
            <span class="text-gray-300">
              {{ $t('chartAnalyzer.confidence') }}: {{ ((result.confidence ?? 0) * 100).toFixed(0) }}%
            </span>
          </div>
        </div>

        <!-- Analysis Text -->
        <div class="glass-panel rounded-lg p-4 md:p-6 bg-gray-900/50">
          <h3 class="text-lg font-semibold text-white mb-4">{{ $t('chartAnalyzer.detailedAnalysis') }}</h3>
          <div class="prose prose-invert max-w-none text-gray-300 whitespace-pre-wrap">
            {{ result.analysis }}
          </div>
        </div>
      </div>

      <!-- Multi-Timeframe Results -->
      <div v-if="mode === 'multi' && result.aggregated">
        <!-- Aggregated Signal -->
        <div class="mb-6 p-4 md:p-6 bg-gradient-to-br from-purple-900/30 to-blue-900/30 rounded-lg border border-purple-500/50 backdrop-blur-sm">
          <h3 class="text-lg font-semibold text-white mb-4">{{ $t('chartAnalyzer.aggregatedResult') }}</h3>
          <div class="flex items-center gap-4">
            <span
              :class="[
                'px-6 py-3 rounded-lg font-bold text-xl',
                result.aggregated.signal === 'LONG'
                  ? 'bg-green-500/20 text-green-400 border-2 border-green-500'
                  : result.aggregated.signal === 'SHORT'
                  ? 'bg-red-500/20 text-red-400 border-2 border-red-500'
                  : 'bg-gray-500/20 text-gray-400 border-2 border-gray-500'
              ]"
            >
              {{ result.aggregated.signal }}
            </span>
            <span class="text-gray-300 text-lg">
              {{ $t('chartAnalyzer.confidence') }}: {{ (result.aggregated.confidence * 100).toFixed(0) }}%
            </span>
          </div>
        </div>

        <!-- Timeframe Breakdown -->
        <div class="space-y-4">
          <h3 class="text-lg font-semibold text-white">{{ $t('chartAnalyzer.timeframeBreakdown') }}</h3>
          <div
            v-for="(tfResult, tf) in result.timeframes_results"
            :key="tf"
            class="glass-panel rounded-lg p-4 bg-gray-900/50"
          >
            <div class="flex items-center justify-between mb-3">
              <h4 class="text-md font-semibold text-purple-400">{{ tf }}</h4>
              <span
                :class="[
                  'px-4 py-1 rounded-full text-sm font-semibold',
                  tfResult.signal === 'LONG'
                    ? 'bg-green-500/20 text-green-400 border border-green-500/50'
                    : tfResult.signal === 'SHORT'
                    ? 'bg-red-500/20 text-red-400 border border-red-500/50'
                    : 'bg-gray-500/20 text-gray-400 border border-gray-500/50'
                ]"
              >
                {{ tfResult.signal }} ({{ (tfResult.confidence * 100).toFixed(0) }}%)
              </span>
            </div>
            <div v-if="tfResult.chart_url" class="mt-3">
              <img
                :src="tfResult.chart_url"
                :alt="`Chart ${result.symbol} ${tf}`"
                class="w-full rounded-lg border border-gray-700"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { chartAnalyzerAPI } from '../services/api'
import LogPoller from '../services/logPoller'
import LogViewer from './LogViewer.vue'
import CustomDropdown from './CustomDropdown.vue'
const { t } = useI18n()

const route = useRoute()

// Computed options for prompt type dropdown
const promptTypeOptions = computed(() => [
  { value: 'detailed', label: t('chartAnalyzer.promptTypes.detailed') },
  { value: 'simple', label: t('chartAnalyzer.promptTypes.simple') },
  { value: 'custom', label: t('chartAnalyzer.promptTypes.custom') }
])

// State
const mode = ref('single')
const loading = ref(false)
const error = ref(null)
const result = ref(null)
const logs = ref([])
const logPoller = ref(null)

// Form data
const form = ref({
  symbol: '',
  timeframe: '1h',
  timeframes: '15m, 1h, 4h, 1d',
  indicators: {
    maPeriods: '20, 50, 200',
    rsiPeriod: 14,
    enableMacd: true,
    enableBb: false,
  },
  promptType: 'detailed',
  customPrompt: '',
  limit: 500,
})

// Load symbol from query parameter
onMounted(() => {
  if (route.query.symbol) {
    form.value.symbol = route.query.symbol
  }
})

// Computed
const isFormValid = computed(() => {
  if (!form.value.symbol) return false
  if (mode.value === 'single' && !form.value.timeframe) return false
  if (mode.value === 'multi' && !form.value.timeframes) return false
  return true
})

// Methods
async function handleAnalyze() {
  if (!isFormValid.value) return

  loading.value = true
  error.value = null
  result.value = null
  logs.value = []

  // Stop existing poller if any
  if (logPoller.value) {
    logPoller.value.stopPolling()
    logPoller.value = null
  }

  try {
    // Parse indicators
    const indicators = {
      ma_periods: form.value.indicators.maPeriods
        ? form.value.indicators.maPeriods.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n))
        : [20, 50, 200],
      rsi_period: form.value.indicators.rsiPeriod || 14,
      enable_macd: form.value.indicators.enableMacd,
      enable_bb: form.value.indicators.enableBb,
    }

    const config = {
      indicators,
      promptType: form.value.promptType,
      customPrompt: form.value.customPrompt || null,
      limit: form.value.limit,
    }

    let response

    if (mode.value === 'single') {
      response = await chartAnalyzerAPI.analyzeSingle(
        form.value.symbol,
        form.value.timeframe,
        config
      )
    } else {
      // Parse timeframes
      const timeframes = form.value.timeframes
        .split(',')
        .map(s => s.trim())
        .filter(s => s)

      response = await chartAnalyzerAPI.analyzeMulti(
        form.value.symbol,
        timeframes,
        config
      )
    }

    if (response.data?.session_id) {
      // Start polling logs and status
      startLogPolling(response.data.session_id)
    } else {
      // Fallback: if no session_id, treat as old API response
      result.value = response.data || response
      loading.value = false
    }
  } catch (err) {
    error.value = err.message || t('chartAnalyzer.errors.analyzeError')
    result.value = null
    console.error('Analysis error:', err)
    loading.value = false
    if (logPoller.value) {
      logPoller.value.stopPolling()
      logPoller.value = null
    }
  }
}

function startLogPolling(sessionId) {
  // Create log poller
  logPoller.value = new LogPoller(
    sessionId,
    'analyze',
    // onLogUpdate
    (newLogLines, allLogs) => {
      logs.value = [...allLogs]
      // Auto-scroll to bottom
      // Use a template ref for the log container for robust access and avoid setTimeout after unmount.
      nextTick(() => {
        if (logContainerRef.value) {
          logContainerRef.value.scrollTop = logContainerRef.value.scrollHeight
        }
      })
    },
    // onStatusUpdate
    (status, statusResponse) => {
      // Status updated, can show progress if needed
    },
    // onComplete
    (resultData, errorMsg) => {
      loading.value = false
      
      if (errorMsg) {
        error.value = errorMsg
        result.value = null
      } else if (resultData) {
        result.value = resultData
        error.value = null
      } else {
        // No error and no result - might be normal completion
        error.value = null
        // Keep existing result if any
      }
      
      // Stop polling
      if (logPoller.value) {
        logPoller.value.stopPolling()
        logPoller.value = null
      }
    }
  )

  // Start polling
  logPoller.value.startPolling()
}

onUnmounted(() => {
  if (logPoller.value) {
    logPoller.value.stopPolling()
    logPoller.value = null
  }
})
</script>

