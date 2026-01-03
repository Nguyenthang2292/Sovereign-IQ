<template>
  <div v-if="results" class="mt-5">
    <!-- Summary Section -->
    <div class="glass-panel bg-gradient-to-br from-gray-800/70 to-gray-900/70 p-4 md:p-6 rounded-xl mb-6">
      <h3 class="flex items-center gap-2 text-xl font-bold text-green-400 mb-5">
        <span class="text-2xl">📊</span>
        <span>{{ $t('results.summary') }}</span>
      </h3>
      <div class="grid grid-cols-2 md:grid-cols-3 gap-4 justify-items-stretch">
        <div class="glass-panel bg-gray-800/50 p-4 md:p-5 rounded-lg flex items-center gap-3 border-2 border-green-500/50 hover:border-green-400 transition-all hover:transform hover:-translate-y-1 shadow-md hover:shadow-neon-cyan w-full">
          <span class="text-3xl">📈</span>
          <div class="flex-1">
            <div class="text-xs uppercase text-gray-400 font-semibold tracking-wide">{{ $t('results.long') }}</div>
            <div class="text-2xl font-bold text-green-400">{{ summary.longCount || 0 }}</div>
          </div>
        </div>
        <div class="glass-panel bg-gray-800/50 p-4 md:p-5 rounded-lg flex items-center gap-3 border-2 border-gray-500/50 hover:border-gray-400 transition-all hover:transform hover:-translate-y-1 shadow-md w-full">
          <span class="text-3xl">➖</span>
          <div class="flex-1">
            <div class="text-xs uppercase text-gray-400 font-semibold tracking-wide">{{ $t('results.none') }}</div>
            <div class="text-2xl font-bold text-gray-300">{{ summary.noneCount || 0 }}</div>
          </div>
        </div>
        <div class="glass-panel bg-gray-800/50 p-4 md:p-5 rounded-lg flex items-center gap-3 border-2 border-purple-500/50 hover:border-purple-400 transition-all hover:transform hover:-translate-y-1 shadow-md hover:shadow-neon-purple w-full md:col-span-1 col-span-2">
          <span class="text-3xl">🔢</span>
          <div class="flex-1">
            <div class="text-xs uppercase text-gray-400 font-semibold tracking-wide">{{ $t('results.total') }}</div>
            <div class="text-2xl font-bold text-purple-400">{{ summary.total || 0 }}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Signals Table Section -->
    <div v-if="longSignals.length > 0" class="glass-panel rounded-xl overflow-hidden" style="position: relative;">
      <!-- Filter and Sort Controls -->
      <div class="p-4 bg-gray-700/30 border-b border-gray-600/50 flex flex-nowrap gap-4 items-center">
        <div class="min-w-[200px] flex-1 max-w-[400px]">
          <input
            v-model="filterText"
            type="text"
            :placeholder="'🔍 ' + $t('results.searchPlaceholder')"
            class="w-full px-4 py-2 bg-gray-700/50 border border-gray-600/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 backdrop-blur-sm"
          />
        </div>
        <div class="flex gap-2 items-center" style="flex-shrink: 0;">
          <CustomDropdown
            v-model="sortBy"
            :options="sortByOptions"
            option-label="label"
            option-value="value"
            :placeholder="$t('results.sortBy.confidence')"
            style="flex-shrink: 0; width: 160px; min-width: 160px; max-width: 160px;"
          />
          <button
            @click="sortOrder = sortOrder === 'asc' ? 'desc' : 'asc'"
            class="bg-gray-700/50 border border-gray-600/50 rounded-lg text-white hover:bg-gray-600/50 transition-colors backdrop-blur-sm flex items-center justify-center"
            style="width: 48px; height: 40px; flex-shrink: 0; min-width: 48px; max-width: 48px;"
            :title="sortOrder === 'asc' ? $t('results.sortBy.ascending') : $t('results.sortBy.descending')"
          >
            <span class="text-lg">{{ sortOrder === 'asc' ? '↑' : '↓' }}</span>
          </button>
          <CustomDropdown
            v-model="itemsPerPage"
            :options="[5, 10, 20, 50, 100]"
            :placeholder="$t('results.itemsPerPage.title')"
            style="flex-shrink: 0; width: 120px; min-width: 120px; max-width: 120px;"
            data-testid="items-per-page-selector"
          />
        </div>
      </div>

      <!-- Table -->
      <div v-if="filteredSignals.length > 0" ref="tableContainerRef" class="overflow-x-auto" style="will-change: scroll-position; min-height: 200px; position: relative; scroll-behavior: auto; overflow-anchor: none; display: block; direction: ltr; text-align: left; scrollbar-gutter: stable;">
        <table 
          ref="tableRef"
          style="table-layout: fixed; width: 1231px; border-collapse: separate; border-spacing: 0; contain: layout style paint; backface-visibility: hidden; transform: translateZ(0); opacity: 1; transition: none; position: relative; left: 0; margin: 0; display: table;"
        >
          <colgroup>
            <col style="width: 308px;">
            <col style="width: 384px;">
            <col style="width: 231px;">
            <col style="width: 308px;">
          </colgroup>
          <thead class="bg-gray-700/50">
            <tr>
              <th 
                @click="sortTable('symbol')"
                class="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-gray-600"
                style="width: 308px; min-width: 308px; max-width: 308px;"
              >
                {{ $t('results.table.symbol') }}
                <span v-if="sortBy === 'symbol'" class="ml-1">
                  {{ sortOrder === 'asc' ? '↑' : '↓' }}
                </span>
              </th>
              <th 
                @click="sortTable('confidence')"
                class="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider cursor-pointer hover:bg-gray-600"
                style="width: 384px; min-width: 384px; max-width: 384px;"
              >
                {{ $t('results.table.confidence') }}
                <span v-if="sortBy === 'confidence'" class="ml-1">
                  {{ sortOrder === 'asc' ? '↑' : '↓' }}
                </span>
              </th>
              <th 
                class="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider"
                style="width: 231px; min-width: 231px; max-width: 231px;"
              >
                {{ $t('results.table.signal') }}
              </th>
              <th 
                class="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider"
                style="width: 308px; min-width: 308px; max-width: 308px;"
              >
                {{ $t('results.table.actions') }}
              </th>
            </tr>
          </thead>
          <tbody class="bg-gray-800/30 divide-y divide-gray-700/50">
            <tr 
              v-for="signal in paginatedSignals" 
              :key="signal.symbol"
              class="hover:bg-gray-700/50 transition-colors"
              style="will-change: auto;"
            >
              <td class="px-6 py-4 whitespace-nowrap" style="width: 308px; min-width: 308px; max-width: 308px; overflow: hidden;">
                <div class="text-sm font-medium text-white truncate">{{ signal.symbol }}</div>
              </td>
              <td class="px-6 py-4 whitespace-nowrap" style="width: 384px; min-width: 384px; max-width: 384px; overflow: hidden;">
                <div class="flex items-center gap-2">
                  <div class="flex-1 bg-gray-700/50 rounded-full h-2 overflow-hidden min-w-0">
                    <div 
                      class="h-full transition-all bg-green-500"
                      :style="{ width: `${(signal.confidence || 0) * 100}%` }"
                    ></div>
                  </div>
                  <span class="text-sm text-gray-300 min-w-[50px] flex-shrink-0">
                    {{ formatConfidence(signal.confidence) }}
                  </span>
                </div>
              </td>
              <td class="px-6 py-4 whitespace-nowrap" style="width: 231px; min-width: 231px; max-width: 231px; overflow: hidden;">
                <span 
                  class="px-3 py-1 rounded-full text-xs font-semibold inline-block bg-green-500/20 text-green-400 border border-green-500/50"
                >
                  {{ signal.signal }}
                </span>
              </td>
              <td class="px-6 py-4 whitespace-nowrap" style="width: 308px; min-width: 308px; max-width: 308px; overflow: hidden;">
                <button
                  @click="handleSymbolClick(signal.symbol)"
                  data-testid="analyze-button"
                  class="px-4 py-2 btn-gradient hover:shadow-glow-purple text-white rounded-lg text-sm font-medium transition-all duration-300 hover:scale-105 active:scale-95 whitespace-nowrap"
                >
                  {{ $t('results.table.analyze') }}
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Empty State for Filtered Results -->
      <div v-else class="p-8 md:p-12 text-center">
        <div class="text-6xl mb-4">🔍</div>
        <h3 class="text-xl font-bold text-gray-300 mb-2">{{ $t('results.empty.filtered.title', 'No results found') }}</h3>
        <p class="text-gray-400">{{ $t('results.empty.filtered.description', 'Try adjusting your search or filter criteria.') }}</p>
      </div>

      <!-- Pagination -->
      <div v-if="filteredSignals.length > 0 && totalPages > 1" class="p-4 bg-gray-700/50 border-t border-gray-600/50 flex flex-col md:flex-row items-center justify-between gap-4">
        <div data-testid="row-count" class="text-sm text-gray-300">
          {{ $t('results.pagination.showing') }} {{ startIndex + 1 }} - {{ endIndex }} {{ $t('results.pagination.of') }} {{ filteredSignals.length }} {{ $t('results.pagination.results') }}
        </div>
        <div class="flex gap-2">
          <button
            @click="currentPage = Math.max(1, currentPage - 1)"
            :disabled="currentPage === 1"
            :class="[
              'px-4 py-2 rounded-lg font-medium transition-colors',
              currentPage === 1
                ? 'bg-gray-600/50 text-gray-400 cursor-not-allowed border border-gray-600/50'
                : 'bg-gray-600/50 text-white hover:bg-gray-500/50 border border-gray-600/50 backdrop-blur-sm'
            ]"
          >
            {{ $t('results.pagination.previous') }}
          </button>
          <span data-testid="pagination-page" class="px-4 py-2 text-gray-300">
            {{ $t('results.pagination.page') }} {{ currentPage }} / {{ totalPages }}
          </span>
          <button
            @click="currentPage = Math.min(totalPages, currentPage + 1)"
            :disabled="currentPage === totalPages"
            :class="[
              'px-4 py-2 rounded-lg font-medium transition-colors',
              currentPage === totalPages
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-gray-600 text-white hover:bg-gray-500'
            ]"
            data-testid="pagination-next"
          >
            {{ $t('results.pagination.next') }}
          </button>
        </div>
      </div>
    </div>

    <!-- Empty State -->
    <div v-else class="glass-panel rounded-xl p-8 md:p-12 text-center">
      <div class="text-6xl mb-4">📭</div>
      <h3 class="text-xl font-bold text-gray-300 mb-2">{{ $t('results.empty.title') }}</h3>
      <p class="text-gray-400">{{ $t('results.empty.description') }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUpdated, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import CustomDropdown from './CustomDropdown.vue'

// #region agent log
const logDebug = (location, message, data, hypothesisId) => {
  fetch('http://127.0.0.1:7242/ingest/f5c9debd-a932-41da-9194-5079ce360f94', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      location,
      message,
      data,
      timestamp: Date.now(),
      sessionId: 'debug-session',
      runId: 'run1',
      hypothesisId
    })
  }).catch(() => {});
};
// #endregion

const { t } = useI18n()

const props = defineProps({
  results: {
    type: Object,
    default: null
  }
})

const emit = defineEmits(['symbol-click'])

// State
const filterText = ref('')
const sortBy = ref('confidence')
const sortOrder = ref('desc')
const currentPage = ref(1)
const itemsPerPage = ref(20)
const tableRef = ref(null)
const tableContainerRef = ref(null)

// Computed options for sortBy dropdown
const sortByOptions = computed(() => [
  { value: 'confidence', label: t('results.sortBy.confidence') },
  { value: 'symbol', label: t('results.sortBy.symbol') }
])

// Computed
const longSymbols = computed(() => {
  if (!props.results) return []
  const symbols = props.results.long_symbols_with_confidence || props.results.longSymbolsWithConfidence || props.results.long_symbols || props.results.longSymbols || []
  return normalizeSymbols(symbols)
})

const longSignals = computed(() => {
  return longSymbols.value.map(s => ({ ...s, signal: 'LONG' }))
})

const filteredSignals = computed(() => {
  let signals = longSignals.value
  if (filterText.value) {
    const filter = filterText.value.toLowerCase()
    signals = signals.filter(s => s.symbol.toLowerCase().includes(filter))
  }
  return signals
})

const sortedSignals = computed(() => {
  const signals = [...filteredSignals.value]
  if (sortBy.value) {
    signals.sort((a, b) => {
      let aVal, bVal
      if (sortBy.value === 'symbol') {
        aVal = a.symbol || ''
        bVal = b.symbol || ''
        return sortOrder.value === 'asc' 
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal)
      } else if (sortBy.value === 'confidence') {
        aVal = a.confidence || 0
        bVal = b.confidence || 0
        return sortOrder.value === 'asc' ? aVal - bVal : bVal - aVal
      }
      return 0
    })
  }
  return signals
})

const paginatedSignals = computed(() => {
  const start = (currentPage.value - 1) * itemsPerPage.value
  const end = start + itemsPerPage.value
  return sortedSignals.value.slice(start, end)
})

const totalPages = computed(() => {
  return Math.ceil(filteredSignals.value.length / (itemsPerPage.value || 1))
})

const startIndex = computed(() => {
  if (filteredSignals.value.length === 0) return 0
  return (currentPage.value - 1) * itemsPerPage.value
})

const endIndex = computed(() => {
  if (filteredSignals.value.length === 0) return 0
  return Math.min(startIndex.value + itemsPerPage.value, filteredSignals.value.length)
})

const summary = computed(() => {
  if (!props.results) {
    return { longCount: 0, shortCount: 0, noneCount: 0, total: 0 }
  }
  
  // Try to use summary object from server first (more accurate)
  const serverSummary = props.results.summary
  if (serverSummary && typeof serverSummary === 'object') {
    const result = {
      longCount: Number(serverSummary.long_count || serverSummary.longCount || 0) || 0,
      shortCount: Number(serverSummary.short_count || serverSummary.shortCount || 0) || 0,
      noneCount: Number(serverSummary.none_count || serverSummary.noneCount || 0) || 0,
      total: Number(serverSummary.total_symbols || serverSummary.totalSymbols || serverSummary.scanned_symbols || serverSummary.scannedSymbols || 0) || 0
    }
    
    // Debug logging in development
    if (import.meta.env.DEV) {
      console.log('Summary from server:', serverSummary, 'Parsed:', result)
    }
    
    return result
  }
  
  // Fallback: calculate from longSignals if summary not available
  const longCount = longSignals.value.length
  // Add fallback for camelCase variant of none_symbols
  const noneSymbols =
    props.results.none_symbols ||
    props.results.noneSymbols ||
    []
  const noneCount = noneSymbols.length
  
  const result = {
    longCount: Number(longCount) || 0,
    shortCount: 0,
    noneCount: Number(noneCount) || 0,
    total: Number(longCount + noneCount) || 0
  }
  
  // Debug logging in development
  if (import.meta.env.DEV) {
    console.log('Summary calculated from longSignals:', result, 'longSignals length:', longSignals.value.length)
  }
  
  return result
})

// Methods
function normalizeSymbols(symbols) {
  if (!Array.isArray(symbols)) {
    console.error('normalizeSymbols: Expected an array, but got', symbols)
    return []
  }
  return symbols.map(s => {
    // Array form: [symbol, confidence, timeframe_breakdown]
    if (Array.isArray(s)) {
      return {
        symbol: s[0] !== undefined ? s[0] : 'UNKNOWN',
        confidence: typeof s[1] === 'number' ? s[1] : null,
        timeframe_breakdown: s.length > 2 ? s[2] : null
      }
    }
    // String form: just the symbol name
    else if (typeof s === 'string') {
      return {
        symbol: s,
        confidence: null,
        timeframe_breakdown: null
      }
    }
    // Object form: must have symbol property, may have confidence & timeframe_breakdown
    else if (typeof s === 'object' && s !== null) {
      return {
        symbol: s.symbol !== undefined ? s.symbol : 'UNKNOWN',
        confidence: typeof s.confidence === 'number' ? s.confidence : null,
        timeframe_breakdown: s.hasOwnProperty('timeframe_breakdown') ? s.timeframe_breakdown : null
      }
    }
    // Not recognized — cast to string for symbol, null other fields.
    else {
      console.warn('normalizeSymbols: Unknown symbol data format', s)
      return {
        symbol: String(s),
        confidence: null,
        timeframe_breakdown: null
      }
    }
  })
}

function formatConfidence(confidence) {
  if (
    confidence === null ||
    confidence === undefined ||
    typeof confidence !== 'number' ||
    isNaN(confidence)
  ) {
    return 'N/A'
  }
  return `${(confidence * 100).toFixed(0)}%`
}

function sortTable(column) {
  if (sortBy.value === column) {
    sortOrder.value = sortOrder.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortBy.value = column
    sortOrder.value = column === 'confidence' ? 'desc' : 'asc'
  }
  currentPage.value = 1
}

function handleSymbolClick(symbol) {
  emit('symbol-click', symbol)
}

// Helper function to measure table dimensions
const measureTableDimensions = (hypothesisId) => {
  nextTick(() => {
    const table = tableRef.value || document.querySelector('table');
    if (!table) {
      logDebug('ResultsTable.vue:measureTableDimensions', 'Table not found', {}, hypothesisId);
      return;
    }
    
    const container = tableContainerRef.value || table.parentElement;
    const containerRect = container ? container.getBoundingClientRect() : null;
    const containerScrollLeft = container?.scrollLeft || 0;
    const containerScrollWidth = container?.scrollWidth || 0;
    const containerClientWidth = container?.clientWidth || 0;
    
    const tableRect = table.getBoundingClientRect();
    const tableWidth = table.offsetWidth;
    const tableStyle = window.getComputedStyle(table);
    const columns = table.querySelectorAll('th');
    const columnWidths = Array.from(columns).map((th, idx) => {
      const style = window.getComputedStyle(th);
      const col = table.querySelector(`colgroup > col:nth-child(${idx + 1})`);
      const thRect = th.getBoundingClientRect();
      return {
        index: idx,
        text: th.textContent.trim(),
        offsetWidth: th.offsetWidth,
        computedWidth: style.width,
        minWidth: style.minWidth,
        maxWidth: style.maxWidth,
        colWidth: col ? col.style.width : null,
        left: thRect.left,
        right: thRect.right
      };
    });
    
    const firstRow = table.querySelector('tbody tr');
    const rowCellWidths = firstRow ? Array.from(firstRow.querySelectorAll('td')).map((td, idx) => {
      const style = window.getComputedStyle(td);
      const tdRect = td.getBoundingClientRect();
      return {
        index: idx,
        offsetWidth: td.offsetWidth,
        computedWidth: style.width,
        minWidth: style.minWidth,
        maxWidth: style.maxWidth,
        left: tdRect.left,
        right: tdRect.right
      };
    }) : [];
    
    logDebug('ResultsTable.vue:measureTableDimensions', 'Table dimensions measured', {
      tableWidth,
      tableLeft: tableRect.left,
      tableRight: tableRect.right,
      containerLeft: containerRect?.left || null,
      containerRight: containerRect?.right || null,
      containerWidth: containerRect?.width || null,
      containerScrollLeft,
      containerScrollWidth,
      containerClientWidth,
      tableLayout: tableStyle.tableLayout,
      columnCount: columns.length,
      columnWidths,
      rowCellWidths,
      filteredSignalsCount: filteredSignals.value.length,
      paginatedSignalsCount: paginatedSignals.value.length
    }, hypothesisId);
  });
};

// Watchers
watch(filterText, () => {
  currentPage.value = 1
})

watch(itemsPerPage, () => {
  currentPage.value = 1
})

// #region agent log
onMounted(() => {
  logDebug('ResultsTable.vue:onMounted', 'Component mounted', {
    hasResults: !!props.results,
    longSignalsCount: longSignals.value.length
  }, 'C');
  
  measureTableDimensions('C');
});
// #endregion

// #region agent log
onUpdated(() => {
  logDebug('ResultsTable.vue:onUpdated', 'Component updated', {
    filteredSignalsCount: filteredSignals.value.length,
    paginatedSignalsCount: paginatedSignals.value.length
  }, 'D');
  
  nextTick(() => {
    measureTableDimensions('D');
  });
});
// #endregion
</script>

