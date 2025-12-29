<template>
  <div class="max-w-7xl mx-auto p-4 md:p-6">
    <!-- Header -->
    <div class="mb-6 md:mb-8">
      <h1 class="text-3xl md:text-4xl font-bold text-white mb-2 flex items-center gap-3">
        <span class="text-4xl md:text-5xl">🔍</span>
        <span>{{ $t('batchScanner.title') }}</span>
      </h1>
      <p class="text-gray-300 md:text-gray-400 text-sm md:text-base">{{ $t('batchScanner.subtitle') }}</p>
    </div>

    <!-- Mode Toggle -->
    <div class="glass-panel rounded-xl p-4 mb-6">
      <div class="flex gap-4">
        <button
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
      <h2 class="text-xl md:text-2xl font-bold text-white mb-4 md:mb-6">{{ $t('batchScanner.configTitle') }}</h2>
      
      <!-- 4 Input Fields Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
        <!-- 1. Timeframe -->
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
            <span>🕐</span>
            <span>{{ $t('common.timeframe') }} <span class="text-red-400">{{ $t('common.required') }}</span></span>
          </label>
          <div class="relative" v-if="mode === 'single'">
            <span class="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 z-10 pointer-events-none">⏰</span>
            <CustomDropdown
              v-model="form.timeframe"
              :options="['15m', '30m', '1h', '4h', '1d', '1w']"
              :placeholder="$t('common.selectTimeframe')"
              :has-left-icon="true"
            />
          </div>
          <div class="relative" v-else>
            <input
              v-model="form.timeframes"
              type="text"
              :placeholder="$t('batchScanner.fields.example')"
              class="w-full px-4 py-3 pl-10 bg-gray-700/50 border border-gray-600/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 backdrop-blur-sm"
            />
            <span class="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">⏰</span>
          </div>
          <p v-if="mode === 'multi'" class="mt-2 text-xs text-gray-400">{{ $t('common.example') }}: {{ $t('batchScanner.fields.example') }}</p>
        </div>

        <!-- 2. Max Symbols -->
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
            <span>📊</span>
            <span>{{ $t('batchScanner.fields.maxSymbols') }}</span>
          </label>
          <div class="relative">
            <input
              v-model.number="form.maxSymbols"
              type="number"
              min="1"
              max="1000"
              :placeholder="$t('batchScanner.fields.maxSymbolsPlaceholder')"
              :class="[
                'w-full px-4 py-3 pl-10 bg-gray-700/50 border rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 backdrop-blur-sm',
                validationErrors.maxSymbols
                  ? 'border-red-500 focus:ring-red-500 focus:border-red-500'
                  : 'border-gray-600/50 focus:ring-purple-500 focus:border-purple-500'
              ]"
            />
            <span class="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">🔢</span>
          </div>
          <p v-if="validationErrors.maxSymbols" class="mt-1 text-xs text-red-400">
            {{ validationErrors.maxSymbols }}
          </p>
          <p v-else class="mt-1 text-xs text-gray-400">
            {{ $t('batchScanner.validation.maxSymbolsHelper') }}
          </p>        </div>

        <!-- 3. Number of Candles per Symbol -->
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
            <span>📈</span>
            <span>{{ $t('batchScanner.fields.limit') }}</span>
          </label>
          <div class="relative">
            <input
              v-model.number="form.limit"
              type="number"
              min="1"
              max="5000"
              :class="[
                'w-full px-4 py-3 pl-10 bg-gray-700/50 border rounded-lg text-white focus:outline-none focus:ring-2 backdrop-blur-sm',
                validationErrors.limit
                  ? 'border-red-500 focus:ring-red-500 focus:border-red-500'
                  : 'border-gray-600/50 focus:ring-purple-500 focus:border-purple-500'
              ]"
            />
            <span class="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">📊</span>
          </div>
          <p v-if="validationErrors.limit" class="mt-1 text-xs text-red-400">
            {{ validationErrors.limit }}
          </p>
          <p v-else class="mt-1 text-xs text-gray-400">
            {{ $t('batchScanner.validation.limitHelper') }}
          </p>        </div>

        <!-- 4. Cooldown -->
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
            <span>⏱️</span>
            <span>{{ $t('batchScanner.fields.cooldown') }}</span>
          </label>
          <div class="relative">
            <input
              v-model.number="form.cooldown"
              type="number"
              min="0"
              max="60"
              step="0.1"
              :class="[
                'w-full px-4 py-3 pl-10 bg-gray-700/50 border rounded-lg text-white focus:outline-none focus:ring-2 backdrop-blur-sm',
                validationErrors.cooldown
                  ? 'border-red-500 focus:ring-red-500 focus:border-red-500'
                  : 'border-gray-600/50 focus:ring-purple-500 focus:border-purple-500'
              ]"
            />
            <span class="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">⏳</span>
          </div>
          <p v-if="validationErrors.cooldown" class="mt-1 text-xs text-red-400">
            {{ validationErrors.cooldown }}
          </p>
          <p v-else class="mt-1 text-xs text-gray-400">
            {{ $t('batchScanner.validation.cooldownHelper') }}
          </p>
        </div>
      </div>

      <!-- Submit Button -->
      <div class="pt-4 md:pt-6">
          <button
            v-if="!loading"
            @click="handleScan"
            :disabled="!isFormValid"
            :class="[
              'w-full px-6 py-4 rounded-lg font-semibold text-white transition-all duration-300 flex items-center justify-center gap-2',
              !isFormValid
                ? 'bg-gray-600/50 cursor-not-allowed border border-gray-600/50'
                : 'btn-gradient hover:shadow-glow-purple hover:scale-[1.02] active:scale-[0.98]'
            ]"
          >
            🚀 {{ $t('batchScanner.startScan') }}
          </button>
          <button
            v-else
            @click="handleCancel"
            class="w-full px-6 py-4 rounded-lg font-semibold text-white transition-all duration-300 flex items-center justify-center gap-2 bg-gradient-to-r from-red-600 to-red-500 hover:from-red-700 hover:to-red-600 hover:shadow-lg hover:scale-[1.02] active:scale-[0.98]"
          >
            <span>❌</span>
            <span>{{ $t('batchScanner.cancelScan') }}</span>
          </button>
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
          <h3 class="text-lg font-semibold text-white mb-1">{{ $t('batchScanner.scanningProgress') }}</h3>
          <p class="text-gray-400 text-sm">{{ $t('batchScanner.scanningDescription') }}</p>
        </div>
      </div>
      <div class="w-full bg-gray-700 rounded-full h-2 mb-4">
        <div class="bg-purple-600 h-2 rounded-full animate-pulse" style="width: 100%"></div>
      </div>
    </div>

    <!-- Logs Section (show when there are logs, even after completion) -->
    <div v-if="logs.length > 0" class="glass-panel rounded-xl p-4 md:p-6 mb-6">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold text-white">{{ $t('batchScanner.logs') }}</h3>
        <button
          v-if="!loading"
          @click="logs = []"
          class="px-3 py-1 text-sm text-gray-400 hover:text-white bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
        >
          {{ $t('common.clear') }}
        </button>
      </div>
      <LogViewer ref="logContainerRef" :logs="logs" />
    </div>

    <!-- Results -->
    <div v-if="result && !loading" class="mt-6">
      <ResultsTable 
        :results="result" 
        @symbol-click="handleSymbolClick"
      />
    </div>

    <!-- Error Message -->
    <div v-if="error" class="glass-panel border border-red-500/50 rounded-lg p-4 mb-6 bg-red-900/20">
      <p class="text-red-400 flex items-center gap-2">
        <span>⚠️</span>
        <span>{{ error }}</span>
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onUnmounted, watch, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import { batchScannerAPI } from '../services/api'
import LogPoller from '../services/logPoller'
import ResultsTable from './ResultsTable.vue'
import LogViewer from './LogViewer.vue'
import CustomDropdown from './CustomDropdown.vue'

const { t } = useI18n()

// Emit
const emit = defineEmits(['symbol-click'])

// State
const mode = ref('multi')
const loading = ref(false)
const error = ref(null)
const result = ref(null)
const logs = ref([])
const logPoller = ref(null)
const sessionId = ref(null)
const logContainerRef = ref(null)

// Form data
const form = ref({
  timeframe: '1h',
  timeframes: '15m, 1h, 4h, 1d',
  maxSymbols: null,
  limit: 500,
  cooldown: 2.5,
})

// Validation errors
const validationErrors = ref({
  maxSymbols: null,
  limit: null,
  cooldown: null,
})

// Validators
function validateMaxSymbols() {
  const value = form.value.maxSymbols
  if (value === null || value === undefined || value === '') {
    validationErrors.value.maxSymbols = null
    return
  }
  const num = Number(value)
  if (isNaN(num) || num <= 0) {
    validationErrors.value.maxSymbols = t('batchScanner.validation.positiveNumber')
  } else if (num > 1000) {
    validationErrors.value.maxSymbols = t('batchScanner.validation.maxValue', { max: 1000 })
  } else {
    validationErrors.value.maxSymbols = null
  }
}

function validateLimit() {
  const value = form.value.limit
  if (value === null || value === undefined || value === '') {
    validationErrors.value.limit = t('batchScanner.validation.required')
    return
  }
  const num = Number(value)
  if (isNaN(num) || num <= 0) {
    validationErrors.value.limit = t('batchScanner.validation.positiveNumber')
  } else if (num > 5000) {
    validationErrors.value.limit = t('batchScanner.validation.maxValue', { max: 5000 })
  } else {
    validationErrors.value.limit = null
  }
}

function validateCooldown() {
  const value = form.value.cooldown
  if (value === null || value === undefined || value === '') {
    validationErrors.value.cooldown = null
    return
  }
  const num = Number(value)
  if (isNaN(num) || num < 0) {
    validationErrors.value.cooldown = t('batchScanner.validation.nonNegative')
  } else if (num > 60) {
    validationErrors.value.cooldown = t('batchScanner.validation.maxSeconds', { max: 60 })
  } else {
    validationErrors.value.cooldown = null
  }
}

// Watch form fields for validation
watch(() => form.value.maxSymbols, validateMaxSymbols, { immediate: true })
watch(() => form.value.limit, validateLimit, { immediate: true })
watch(() => form.value.cooldown, validateCooldown, { immediate: true })

// Computed
const isFormValid = computed(() => {
  if (mode.value === 'single' && !form.value.timeframe) return false
  if (mode.value === 'multi' && !form.value.timeframes) return false
  // Check if any validation errors exist
  if (validationErrors.value.maxSymbols) return false
  if (validationErrors.value.limit) return false
  if (validationErrors.value.cooldown) return false
  return true
})

// Methods
async function handleScan() {
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
    const config = {
      timeframe: mode.value === 'single' ? form.value.timeframe : null,
      timeframes: mode.value === 'multi'
        ? form.value.timeframes.split(',').map(s => s.trim()).filter(s => s)
        : null,
      maxSymbols: form.value.maxSymbols || null,
      limit: form.value.limit,
      cooldown: form.value.cooldown,
    }

    // Start scan and get session_id
    const response = await batchScannerAPI.scanMarket(config)
    
    if (response.data?.session_id) {
      sessionId.value = response.data.session_id
      // Start polling logs and status
      startLogPolling(response.data.session_id)
    } else {
      // Fallback: if no session_id, treat as old API response
      result.value = response.data || response
      loading.value = false
    }
  } catch (err) {
    error.value = err.message || t('batchScanner.errors.scanError')
    console.error('Scan error:', err)
    loading.value = false
    sessionId.value = null
    if (logPoller.value) {
      logPoller.value.stopPolling()
      logPoller.value = null
    }
  }
}

async function handleCancel() {
  if (!sessionId.value) return

  try {
    await batchScannerAPI.cancelBatchScan(sessionId.value)
    
    // Stop polling
    if (logPoller.value) {
      logPoller.value.stopPolling()
      logPoller.value = null
    }
    
    // Update UI
    loading.value = false
    error.value = null
    
    // Add cancel message to logs
    logs.value.push(`⚠️ ${t('batchScanner.cancelledByUser')}`)
    
    // Clear session ID
    sessionId.value = null
  } catch (err) {
    error.value = err.message || t('batchScanner.errors.cancelError')
    console.error('Cancel error:', err)
  }
}

function startLogPolling(sessionId) {
  // Create log poller
  logPoller.value = new LogPoller(
    sessionId,
    'scan',
    // onLogUpdate
    (newLogLines, allLogs) => {
      logs.value = [...allLogs]
      // Auto-scroll to bottom using Vue ref
      nextTick(() => {
        if (logContainerRef.value?.scrollContainer) {
          const container = logContainerRef.value.scrollContainer
          // Smooth scroll to bottom
          container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
          })
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
        // Error occurred
        error.value = errorMsg
        result.value = null
        console.error('Scan completed with error:', errorMsg)
      } else if (resultData) {
        // Success with result
        error.value = null
        result.value = resultData
        console.log('Scan completed successfully with result:', resultData)
        
        if (import.meta.env.DEV) {
          console.log('Full result object:', JSON.stringify(resultData, null, 2))
        }      } else {
        
        // Completed but no result data - might be normal completion without data
        error.value = null
        console.warn('Scan completed but no result data provided')
        // Keep existing result if any, but don't set error
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

function handleSymbolClick(symbol) {
  emit('symbol-click', symbol)
}
</script>

