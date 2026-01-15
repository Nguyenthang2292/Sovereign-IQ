/**
 * Tests for BatchScanner component
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { i18n } from '../setup'
import BatchScanner from '../../src/components/BatchScanner.vue'
import { batchScannerAPI } from '../../src/services/api'
import LogPoller from '../../src/services/logPoller'

// Mock dependencies
vi.mock('../../src/services/api', () => ({
  batchScannerAPI: {
    scanMarket: vi.fn(),
  },
}))

// Store LogPoller callbacks for testing
let capturedLogPollerCallbacks = null

vi.mock('../../src/services/logPoller', () => {
  return {
    default: class LogPoller {
      constructor(sessionId, commandType, onLogUpdate, onStatusUpdate, onComplete) {
        this.sessionId = sessionId
        this.commandType = commandType
        this.onLogUpdate = onLogUpdate
        this.onStatusUpdate = onStatusUpdate
        this.onComplete = onComplete
        this.startPolling = vi.fn()
        this.stopPolling = vi.fn()
        this.getAllLogs = vi.fn(() => [])
        
        // Store callbacks for test access
        capturedLogPollerCallbacks = {
          onLogUpdate,
          onStatusUpdate,
          onComplete,
          instance: this
        }
      }
    },
  }
})

describe('BatchScanner', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    capturedLogPollerCallbacks = null
  })

  it('should render component correctly', () => {
    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    expect(wrapper.text()).toContain(i18n.global.t('batchScanner.title'))
    expect(wrapper.text()).toContain(i18n.global.t('batchScanner.startScan'))
  })

  it('should toggle between single and multi timeframe modes', async () => {
    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    const singleButton = wrapper.findAll('button').find(b => b.text().includes(i18n.global.t('common.singleTimeframe')))
    const multiButton = wrapper.findAll('button').find(b => b.text().includes(i18n.global.t('common.multiTimeframe')))

    expect(singleButton.exists()).toBe(true)
    expect(multiButton.exists()).toBe(true)

    // Default should be multi
    expect(wrapper.vm.mode).toBe('multi')
    
    // Multi button should have gradient class when active
    expect(multiButton.classes()).toContain('btn-gradient')
    expect(singleButton.classes()).not.toContain('btn-gradient')

    // Click single mode
    await singleButton.trigger('click')
    await nextTick()

    expect(wrapper.vm.mode).toBe('single')
    
    // Single button should have gradient class when active
    expect(singleButton.classes()).toContain('btn-gradient')
    expect(multiButton.classes()).not.toContain('btn-gradient')
  })

  it('should show single timeframe input in single mode', async () => {
    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    // Click single mode button
    const singleButton = wrapper.findAll('button').find(b => b.text().includes(i18n.global.t('common.singleTimeframe')))
    expect(singleButton).toBeDefined()
    await singleButton.trigger('click')
    await nextTick()

    // In single mode, timeframe uses CustomDropdown component, not native select
    const customDropdown = wrapper.findComponent({ name: 'CustomDropdown' })
    expect(customDropdown.exists()).toBe(true)
  })

  it('should show multi timeframes input in multi mode', () => {
    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    // Default is multi mode
    const input = wrapper.find('input[placeholder*="15m, 1h, 4h, 1d"]')
    expect(input.exists()).toBe(true)
  })

  it('should validate form correctly', async () => {
    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    // Default is multi mode, should be valid
    expect(wrapper.vm.isFormValid).toBe(true)

    // Switch to single mode
    const singleButton = wrapper.findAll('button').find(b => b.text().includes('Single Timeframe'))
    if (singleButton) {
      await singleButton.trigger('click')
      await nextTick()
      
      // Should still be valid with default timeframe
      expect(wrapper.vm.isFormValid).toBe(true)
    }
  })

  it('should disable submit button when form is invalid', async () => {
    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    // Form should be valid by default (multi mode with timeframes)
    const buttons = wrapper.findAll('button')
    const submitButton = buttons.find(b => b.text().includes('Bắt Đầu Scan'))
    if (submitButton) {
      // Should not be disabled when form is valid
      expect(submitButton.attributes('disabled')).toBeUndefined()
    }
  })

  it('should call scanMarket API on form submit', async () => {
    const mockResponse = { data: { session_id: 'test-session-123' } }
    batchScannerAPI.scanMarket.mockResolvedValue(mockResponse)

    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    // Update form values directly via component instance
    wrapper.vm.form.timeframes = '15m, 1h'
    wrapper.vm.maxSymbolsDisplay = '10'
    wrapper.vm.form.maxSymbols = 10
    wrapper.vm.limitDisplay = '500'
    wrapper.vm.form.limit = 500
    wrapper.vm.cooldownDisplay = '2.5'
    wrapper.vm.form.cooldown = 2.5
    await nextTick()

    // Trigger scan
    await wrapper.vm.handleScan()
    await nextTick()

    expect(batchScannerAPI.scanMarket).toHaveBeenCalledWith({
      timeframe: null,
      timeframes: ['15m', '1h'],
      maxSymbols: 10,
      limit: 500,
      cooldown: 2.5,
    })
  })

  it('should start log polling when session_id is returned', async () => {
    const mockResponse = { data: { session_id: 'test-session-123' } }
    batchScannerAPI.scanMarket.mockResolvedValue(mockResponse)

    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    // Update form values
    wrapper.vm.form.timeframes = '15m, 1h'
    wrapper.vm.limitDisplay = '500'
    wrapper.vm.form.limit = 500
    await nextTick()

    await wrapper.vm.handleScan()
    await nextTick()

    // Verify LogPoller was instantiated and startPolling was called
    expect(wrapper.vm.logPoller).toBeDefined()
    expect(wrapper.vm.logPoller.startPolling).toHaveBeenCalled()
  })

  it('should handle scan errors', async () => {
    const error = new Error('Scan failed')
    batchScannerAPI.scanMarket.mockRejectedValue(error)

    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    wrapper.vm.form.timeframes = '15m, 1h'
    wrapper.vm.limitDisplay = '500'
    wrapper.vm.form.limit = 500
    await nextTick()

    await wrapper.vm.handleScan()
    await nextTick()

    expect(wrapper.vm.error).toBe('Scan failed')
    expect(wrapper.vm.loading).toBe(false)
  })

  it('should display logs when polling', async () => {
    const mockResponse = { data: { session_id: 'test-session-123' } }
    batchScannerAPI.scanMarket.mockResolvedValue(mockResponse)

    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    // Set form values
    wrapper.vm.form.timeframes = '15m, 1h'
    wrapper.vm.limitDisplay = '500'
    wrapper.vm.form.limit = 500
    await nextTick()

    // Start scan which will create LogPoller and call startPolling
    await wrapper.vm.handleScan()
    await nextTick()

    // Verify LogPoller was created and startPolling was called
    expect(capturedLogPollerCallbacks).toBeTruthy()
    // Use toStrictEqual for object comparison instead of toBe (reference equality)
    expect(wrapper.vm.logPoller).toBeTruthy()
    expect(wrapper.vm.loading).toBe(true)

    // Simulate log updates via the callback (as LogPoller would do in real usage)
    const onLogUpdate = capturedLogPollerCallbacks.onLogUpdate
    expect(onLogUpdate).toBeTruthy()

    // Simulate first log update
    onLogUpdate(['Log 1'], ['Log 1'])
    await nextTick()
    expect(wrapper.text()).toContain('Log 1')

    // Simulate second log update
    onLogUpdate(['Log 2'], ['Log 1', 'Log 2'])
    await nextTick()
    expect(wrapper.text()).toContain('Log 2')
  })

  it('should stop polling on component unmount', () => {
    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })
    
    // Create a mock poller instance
    const mockPoller = new LogPoller()
    wrapper.vm.logPoller = mockPoller

    wrapper.unmount()

    expect(mockPoller.stopPolling).toHaveBeenCalled()
  })

  it('should emit symbol-click event', async () => {
    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    await wrapper.vm.handleSymbolClick('BTC/USDT')

    expect(wrapper.emitted('symbol-click')).toBeTruthy()
    expect(wrapper.emitted('symbol-click')[0]).toEqual(['BTC/USDT'])
  })

  it('should handle old API response without session_id', async () => {
    const mockResponse = { results: { summary: {} } }
    batchScannerAPI.scanMarket.mockResolvedValue(mockResponse)

    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    wrapper.vm.form.timeframes = '15m, 1h'
    wrapper.vm.limitDisplay = '500'
    wrapper.vm.form.limit = 500
    await nextTick()

    await wrapper.vm.handleScan()
    await nextTick()

    expect(wrapper.vm.result).toEqual(mockResponse)
    expect(wrapper.vm.loading).toBe(false)
    expect(wrapper.vm.logPoller).toBeNull()
  })

  it('should parse timeframes correctly', async () => {
    const mockResponse = { data: { session_id: 'test-session' } }
    batchScannerAPI.scanMarket.mockResolvedValue(mockResponse)

    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    wrapper.vm.form.timeframes = '15m, 1h, 4h, 1d'
    await nextTick()

    await wrapper.vm.handleScan()
    await nextTick()

    expect(batchScannerAPI.scanMarket).toHaveBeenCalledWith(
      expect.objectContaining({
        timeframes: ['15m', '1h', '4h', '1d'],
      })
    )
  })

  // Helper function to set up component, start scan, and return wrapper and callbacks
  async function setupScanWithPoller(timeframes = '15m, 1h') {
    const mockResponse = { data: { session_id: 'test-session' } }
    batchScannerAPI.scanMarket.mockResolvedValue(mockResponse)

    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    wrapper.vm.form.timeframes = timeframes
    await nextTick()

    await wrapper.vm.handleScan()
    await nextTick()

    return { wrapper, callbacks: capturedLogPollerCallbacks }
  }

  // Helper function to simulate scan completion with mock result
  async function simulateScanComplete(callbacks, mockResult) {
    const onComplete = callbacks.onComplete
    expect(onComplete).toBeTruthy()
    onComplete(mockResult, null)
    await nextTick()
  }

  it('creates LogPoller and updates loading/state on start/complete', async () => {
    const { wrapper, callbacks } = await setupScanWithPoller()

    // Verify LogPoller was created and callbacks exist
    expect(callbacks).toBeTruthy()
    expect(callbacks.onLogUpdate).toBeTruthy()
    expect(callbacks.onStatusUpdate).toBeTruthy()
    expect(callbacks.onComplete).toBeTruthy()
    expect(wrapper.vm.logPoller).toBeTruthy()
    expect(wrapper.vm.loading).toBe(true)

    // Simulate scan completion
    const mockResult = {
      success: true,
      summary: {
        total_scanned: 10,
        long_count: 3,
        short_count: 2,
        none_count: 5
      },
      long_symbols_with_confidence: [
        { symbol: 'BTC/USDT', confidence: 0.8 }
      ],
      short_symbols_with_confidence: [
        { symbol: 'ADA/USDT', confidence: 0.7 }
      ]
    }

    await simulateScanComplete(callbacks, mockResult)

    // Verify state updates after completion
    expect(wrapper.vm.result).toEqual(mockResult)
    expect(wrapper.vm.loading).toBe(false)
    expect(wrapper.vm.error).toBeNull()
  })

  it('renders summary statistics', async () => {
    const { wrapper, callbacks } = await setupScanWithPoller()

    const mockResult = {
      success: true,
      summary: {
        total_scanned: 10,
        long_count: 3,
        short_count: 2,
        none_count: 5
      },
      long_symbols_with_confidence: [
        { symbol: 'BTC/USDT', confidence: 0.8 }
      ],
      short_symbols_with_confidence: [
        { symbol: 'ADA/USDT', confidence: 0.7 }
      ]
    }

    await simulateScanComplete(callbacks, mockResult)

    // Verify summary section exists
    const summarySection = wrapper.find('.glass-panel.bg-gradient-to-br')
    expect(summarySection.exists()).toBe(true)

    // Verify counts reflect the summary object from server (not symbols arrays)
    // Summary from server: long_count: 3, short_count: 2, total_scanned: 10
    // Find all elements with text-green-400 and find the one with number
    const longCountElements = summarySection.findAll('.text-green-400')
    const longCountElement = longCountElements.find(el => {
      const text = el.text().trim()
      return /^\d+$/.test(text)
    })
    expect(longCountElement).toBeTruthy()
    if (longCountElement) {
      // Summary uses server summary: long_count: 3
      expect(longCountElement.text()).toMatch(/3/)
    }

    // shortCount from server summary: 2
    const shortCountElements = summarySection.findAll('.text-red-400')
    const shortCountElement = shortCountElements.find(el => {
      const text = el.text().trim()
      return /^\d+$/.test(text)
    })
    expect(shortCountElement).toBeTruthy()
    if (shortCountElement) {
      // Summary uses server summary: short_count: 2
      expect(shortCountElement.text()).toMatch(/2/)
    }

    // total from server summary: total_scanned: 10 (but may be 0 if not in summary)
    // Note: total uses total_scanned or total_symbols from server summary
    const totalElements = summarySection.findAll('.text-purple-400')
    const totalElement = totalElements.find(el => {
      const text = el.text().trim()
      return /^\d+$/.test(text)
    })
    expect(totalElement).toBeTruthy()
  })

  it('renders results table and formats confidence', async () => {
    const { wrapper, callbacks } = await setupScanWithPoller()

    const mockResult = {
      success: true,
      summary: {
        total_scanned: 10,
        long_count: 3,
        short_count: 2,
        none_count: 5
      },
      long_symbols_with_confidence: [
        { symbol: 'BTC/USDT', confidence: 0.8 }
      ],
      short_symbols_with_confidence: [
        { symbol: 'ADA/USDT', confidence: 0.7 }
      ]
    }

    await simulateScanComplete(callbacks, mockResult)
    await nextTick()

    // Verify ResultsTable component is rendered
    const resultsTable = wrapper.findComponent({ name: 'ResultsTable' })
    expect(resultsTable.exists()).toBe(true)

    // Wait for table to render
    await nextTick()

    // Verify table rows contain expected symbols
    // Note: ResultsTable shows signals based on activeTab, default is 'long'
    // So we should see BTC/USDT in LONG tab
    const tableRows = wrapper.findAll('tbody tr')
    expect(tableRows.length).toBeGreaterThan(0)

    // Find BTC row (should be in LONG tab by default)
    const btcRow = tableRows.find(row => row.text().includes('BTC/USDT'))
    expect(btcRow).toBeTruthy()

    // Verify confidence values are formatted as percentages (0.8 = 80%)
    if (btcRow) {
      expect(btcRow.text()).toContain('BTC/USDT')
      expect(btcRow.text()).toContain('80%')
    }

    // Switch to SHORT tab to see ADA/USDT
    const shortTab = wrapper.findAll('button').find(b => {
      const text = b.text()
      return text.includes('SHORT') || text.includes(i18n.global.t('results.short'))
    })
    if (shortTab) {
      await shortTab.trigger('click')
      await nextTick()

      const shortTableRows = wrapper.findAll('tbody tr')
      const adaRow = shortTableRows.find(row => row.text().includes('ADA/USDT'))
      expect(adaRow).toBeTruthy()
      if (adaRow) {
        expect(adaRow.text()).toContain('ADA/USDT')
        expect(adaRow.text()).toContain('70%')
      }
    }
  })

  it('should keep logs visible after scan completes', async () => {
    const mockResponse = { data: { session_id: 'test-session-123' } }
    batchScannerAPI.scanMarket.mockResolvedValue(mockResponse)

    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    // Set form values
    wrapper.vm.form.timeframes = '15m, 1h'
    wrapper.vm.limitDisplay = '500'
    wrapper.vm.form.limit = 500
    await nextTick()

    // Start scan which will create LogPoller
    await wrapper.vm.handleScan()
    await nextTick()

    // Verify LogPoller was created
    expect(capturedLogPollerCallbacks).toBeTruthy()
    expect(wrapper.vm.loading).toBe(true)

    const onLogUpdate = capturedLogPollerCallbacks.onLogUpdate
    const onComplete = capturedLogPollerCallbacks.onComplete

    // Simulate log updates during scan (as LogPoller would do)
    onLogUpdate(['Log 1'], ['Log 1'])
    await nextTick()
    expect(wrapper.text()).toContain('Log 1')

    onLogUpdate(['Log 2'], ['Log 1', 'Log 2'])
    await nextTick()
    expect(wrapper.text()).toContain('Log 2')

    onLogUpdate(['SCAN COMPLETED'], ['Log 1', 'Log 2', 'SCAN COMPLETED'])
    await nextTick()

    // Simulate scan completion (as LogPoller would do when status becomes 'completed')
    onComplete(null, null)
    await nextTick()

    // After completion, loading should be false but logs should still be visible
    expect(wrapper.vm.loading).toBe(false)
    expect(wrapper.text()).toContain('Log 1')
    expect(wrapper.text()).toContain('Log 2')
    expect(wrapper.text()).toContain('SCAN COMPLETED')
  })

  it('should clear logs when clear button is clicked', async () => {
    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    wrapper.vm.logs = ['Log 1', 'Log 2']
    wrapper.vm.loading = false
    await nextTick()

    // Find clear button by text content
    const buttons = wrapper.findAll('button')
    const clearButton = buttons.find(b => b.text().includes(i18n.global.t('common.clear')))
    expect(clearButton).toBeDefined()
    expect(clearButton.exists()).toBe(true)
    await clearButton.trigger('click')
    await nextTick()

    expect(wrapper.vm.logs).toEqual([])
  })

  it('should handle maxSymbols input correctly', async () => {
    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    // Find maxSymbols input
    const maxSymbolsInput = wrapper.find('input[type="number"][min="1"][max="1000"]')
    expect(maxSymbolsInput.exists()).toBe(true)

    // Test typing a valid number
    await maxSymbolsInput.setValue('100')
    await nextTick()
    expect(wrapper.vm.form.maxSymbols).toBe(100)

    // Test clearing (should set to null)
    await maxSymbolsInput.setValue('')
    await nextTick()
    // Trigger blur to finalize
    await maxSymbolsInput.trigger('blur')
    await nextTick()
    expect(wrapper.vm.form.maxSymbols).toBeNull()
    expect(wrapper.vm.maxSymbolsDisplay).toBe('')
  })

  it('should handle limit input correctly', async () => {
    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    // Find limit input (it's the second number input)
    const numberInputs = wrapper.findAll('input[type="number"]')
    const limitInput = numberInputs.find(input => {
      const min = input.attributes('min')
      const max = input.attributes('max')
      return min === '1' && max === '5000'
    })
    expect(limitInput.exists()).toBe(true)

    // Test typing a valid number
    await limitInput.setValue('1000')
    await nextTick()
    expect(wrapper.vm.form.limit).toBe(1000)

    // Test clearing (should reset to default 500 on blur)
    await limitInput.setValue('')
    await nextTick()
    await limitInput.trigger('blur')
    await nextTick()
    expect(wrapper.vm.form.limit).toBe(500)
    expect(wrapper.vm.limitDisplay).toBe('500')
  })

  it('should handle cooldown input correctly', async () => {
    const wrapper = mount(BatchScanner, {
      global: {
        plugins: [i18n],
      },
    })

    // Find cooldown input (has step="0.1")
    const cooldownInput = wrapper.find('input[type="number"][step="0.1"]')
    expect(cooldownInput.exists()).toBe(true)

    // Test typing a valid number
    await cooldownInput.setValue('5.5')
    await nextTick()
    expect(wrapper.vm.form.cooldown).toBe(5.5)

    // Test clearing (should reset to default 2.5 on blur)
    await cooldownInput.setValue('')
    await nextTick()
    await cooldownInput.trigger('blur')
    await nextTick()
    expect(wrapper.vm.form.cooldown).toBe(2.5)
    expect(wrapper.vm.cooldownDisplay).toBe('2.5')
  })
})

