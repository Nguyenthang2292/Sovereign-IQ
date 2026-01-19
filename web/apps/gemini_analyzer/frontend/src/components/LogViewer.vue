<template>
  <div 
    ref="scrollContainerRef"
    class="bg-gray-900 rounded-lg border border-gray-700 p-4 max-h-96 overflow-y-auto"
    role="log"
    aria-label="Application logs"
    aria-live="polite"
  >
    <ul class="space-y-1 font-mono text-sm list-none">
      <li
        v-for="log in processedLogs"
        :key="log.id"
        class="px-2 py-1 rounded flex items-start gap-2 hover:bg-gray-800 transition-colors"
        :class="getLogContainerClass(log.level)"
      >
        <!-- Log level icon -->
        <span 
          class="flex-shrink-0 mt-0.5" 
          :class="getLogIconClass(log.level)"
          :aria-label="log.level"
        >
          {{ getLogIcon(log.level) }}
        </span>
        
        <!-- Log content with ANSI color support -->
        <span class="flex-1">
          <span v-for="(part, partIndex) in log.parts" :key="partIndex" :class="part.colorClass">
            {{ part.text }}
          </span>
        </span>
      </li>
      <li v-if="processedLogs.length === 0" class="text-gray-500 italic px-2 py-1 flex items-center gap-2">        
        <span>⏳</span>
        <span>{{ $t('common.waiting') }}</span>
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { parseAnsiCodes, detectLogLevel, getAnsiColorClass, cleanAnsiCodes, AnsiPart } from '../utils/logParser'

interface LogEntry {
  id?: string | number
  message?: string
  text?: string
  [key: string]: any
}

interface ProcessedLog {
  id: string
  original: string
  clean: string
  level: string
  parts: (AnsiPart & { colorClass: string })[]
}

interface Props {
  logs: (string | LogEntry)[]
}

const props = withDefaults(defineProps<Props>(), {
  logs: () => []
})

// Expose scrollable container ref to parent
const scrollContainerRef = ref<HTMLElement | null>(null)
defineExpose({
  scrollContainer: scrollContainerRef
})

// Counter for generating unique IDs when needed (scoped to component instance)
const idCounter = ref(0)

// Map to cache IDs for log content (ensures same content gets same ID)
const logIdCache = ref(new Map<string, string>())

// Clear cache when logs array reference changes to prevent memory leak
watch(() => props.logs, () => {
  logIdCache.value.clear()
  idCounter.value = 0
}, { flush: 'pre' })
/**
 * Generate a stable unique ID for a log entry
 * Uses content-based caching to ensure identical log content gets the same ID
 * across renders, improving cache hits and Vue component reuse.
 * Position is not included in the cache key since log content identity
 * is determined by its content, not its position in the list.
 * 
 * @param {string|object} log - The log entry (string or object with id property)
 * @param {number} index - The original index in the logs array (used for duplicate handling)
 * @param {Set<string>} usedIdsInRender - Set of IDs already used in current render (for duplicate detection)
 * @returns {string} A unique identifier
 */
function generateLogId(log: string | LogEntry, index: number, usedIdsInRender: Set<string> | null = null): string {
  // If log is an object with an id property (backend-provided), use it
  if (typeof log === 'object' && log !== null && log.id) {
    return String(log.id)
  }
  
  // Extract the log string content
  const logString = typeof log === 'string' ? log : (log?.message || log?.text || String(log))
  
  // Use content-based cache key (without index) to improve cache hits
  // This ensures identical log content gets the same ID regardless of position
  const cacheKey = logString
  
  // Return cached ID if exists, otherwise generate new one
  if (logIdCache.value.has(cacheKey)) {
    const cachedId = logIdCache.value.get(cacheKey)!
    // For duplicates in the same render, append index to ensure Vue key uniqueness
    if (usedIdsInRender && usedIdsInRender.has(cachedId)) {
      return `${cachedId}-${index}`
    }
    if (usedIdsInRender) {
      usedIdsInRender.add(cachedId)
    }
    return cachedId
  }
  
  // Generate a unique ID: timestamp + counter + simple hash of content
  const timestamp = Date.now()
  const hash = simpleHash(logString)
  const id = `log-${timestamp}-${++idCounter.value}-${hash}`
  
  logIdCache.value.set(cacheKey, id)
  if (usedIdsInRender) {
    usedIdsInRender.add(id)
  }
  return id
}

/**
 * Simple hash function for log content
 * @param {string} str - String to hash
 * @returns {string} Hash value
 */
function simpleHash(str: string): string {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i)
    hash = ((hash << 5) - hash) + char
    hash = hash & hash // Convert to 32-bit integer
  }
  return Math.abs(hash).toString(36)
}

const processedLogs = computed<ProcessedLog[]>(() => {
  // Track IDs used in this render to handle duplicates
  const usedIdsInRender = new Set<string>()
  
  return props.logs.map((log: string | LogEntry, index: number) => {
    try {
      // Ensure log is a string
      const logString = typeof log === 'string' ? log : (log?.message || log?.text || String(log))
      
      const cleanText = cleanAnsiCodes(logString)
      const level = detectLogLevel(cleanText)
      const parts = parseAnsiCodes(logString)

      return {
        id: generateLogId(log, index, usedIdsInRender),
        original: logString,
        clean: cleanText,
        level,
        parts: parts.map((part) => ({
          ...part,
          colorClass: getAnsiColorClass(part.color),
        })),
      }
    } catch (error) {
      console.error('Failed to parse log:', error)
      // Return a safe fallback for failed logs
      return {
        id: generateLogId(log, index, usedIdsInRender),
        original: String(log),
        clean: String(log),
        level: 'default',
        parts: [{ text: String(log), color: 'default', colorClass: '' }],
      }
    }
  })
})

function getLogIcon(level: string): string {
  const icons: Record<string, string> = {
    error: '❌',
    warning: '⚠️',
    info: 'ℹ️',
    success: '✅',
    debug: '🔍',
    default: '📝',
  }
  return icons[level] || icons.default
}

function getLogIconClass(level: string): string {
  const classes: Record<string, string> = {
    error: 'text-red-400',
    warning: 'text-yellow-400',
    info: 'text-blue-400',
    success: 'text-green-400',
    debug: 'text-gray-400',
    default: 'text-gray-300',
  }
  return classes[level] || classes.default
}

function getLogContainerClass(level: string): string {
  const classes: Record<string, string> = {
    error: 'bg-red-900/10 border-l-2 border-red-500',
    warning: 'bg-yellow-900/10 border-l-2 border-yellow-500',
    info: 'bg-blue-900/10 border-l-2 border-blue-500',
    success: 'bg-green-900/10 border-l-2 border-green-500',
    debug: 'bg-gray-800/50 border-l-2 border-gray-500',
    default: '',
  }
  return classes[level] || classes.default
}
</script>

<style scoped>
/* Additional styling for log viewer */
</style>

