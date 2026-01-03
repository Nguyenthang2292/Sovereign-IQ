/**
 * Tests for ChartAnalyzer component
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { i18n } from '../setup'
import ChartAnalyzer from '../../src/components/ChartAnalyzer.vue'
import { chartAnalyzerAPI } from '../../src/services/api'
import LogPoller from '../../src/services/logPoller'

// Mock dependencies
vi.mock('../../src/services/api', () => ({
  chartAnalyzerAPI: {
    analyzeSingle: vi.fn(),
    analyzeMulti: vi.fn(),
  },
}))

vi.mock('../../src/services/logPoller', () => {
  const logPollerConstructor = vi.fn(function(...args) {
    this.startPolling = vi.fn()
    this.stopPolling = vi.fn()
    this.getAllLogs = vi.fn(() => [])
  })
  return {
    default: logPollerConstructor,
  }
})

vi.mock('vue-router', () => ({
  useRoute: vi.fn(() => ({
    query: {},
  })),
}))

describe('ChartAnalyzer', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('should render component correctly', () => {
    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    expect(wrapper.text()).toContain(i18n.global.t('chartAnalyzer.title'))
    expect(wrapper.text()).toContain(i18n.global.t('chartAnalyzer.startAnalyze'))
  })

  it('should toggle between single and multi timeframe modes', async () => {
    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    const singleButton = wrapper.findAll('button').find(b => b.text().includes(i18n.global.t('common.singleTimeframe')))
    const multiButton = wrapper.findAll('button').find(b => b.text().includes(i18n.global.t('common.multiTimeframe')))

    expect(singleButton.exists()).toBe(true)
    expect(multiButton.exists()).toBe(true)

    // Default should be single
    expect(wrapper.vm.mode).toBe('single')
    
    // Single button should have gradient class when active
    expect(singleButton.classes()).toContain('btn-gradient')
    expect(multiButton.classes()).not.toContain('btn-gradient')

    // Click multi mode
    await multiButton.trigger('click')
    await nextTick()

    expect(wrapper.vm.mode).toBe('multi')
    
    // Multi button should have gradient class when active
    expect(multiButton.classes()).toContain('btn-gradient')
    expect(singleButton.classes()).not.toContain('btn-gradient')
  })

  it('should load symbol from route query parameter', () => {
    useRoute.mockReturnValue({
      query: { symbol: 'BTC/USDT' },
    })

    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    expect(wrapper.vm.form.symbol).toBe('BTC/USDT')
  })

  it('should validate form correctly', async () => {
    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    // Form should be invalid without symbol
    const symbolInput = wrapper.find('[data-testid="symbol-input"]')
    await symbolInput.setValue('')
    await nextTick()

    expect(wrapper.vm.isFormValid).toBe(false)

    // Form should be valid with symbol and timeframe
    await symbolInput.setValue('BTC/USDT')
    await nextTick()

    // Ensure single mode is selected
    const singleModeButton = wrapper.find('[data-testid="mode-single-button"]')
    await singleModeButton.trigger('click')
    await nextTick()

    // Select timeframe
    const timeframeSelect = wrapper.find('[data-testid="timeframe-select"]')
    const triggerButton = timeframeSelect.find('.dropdown-trigger')
    await triggerButton.trigger('click')
    await nextTick()
    
    // Find and click the '1h' option
    const options = timeframeSelect.findAll('.dropdown-option')
    const option1h = options.find(opt => opt.text().includes('1h'))
    expect(option1h.exists()).toBe(true)
    await option1h.trigger('click')
    await nextTick()

    expect(wrapper.vm.isFormValid).toBe(true)
  })

  it('should call analyzeSingle API in single mode', async () => {
    const mockResponse = { session_id: 'analyze-session-123' }
    chartAnalyzerAPI.analyzeSingle.mockResolvedValue(mockResponse)

    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    // Select single mode
    const singleModeButton = wrapper.find('[data-testid="mode-single-button"]')
    await singleModeButton.trigger('click')
    await nextTick()

    // Set symbol
    const symbolInput = wrapper.find('[data-testid="symbol-input"]')
    await symbolInput.setValue('BTC/USDT')
    await nextTick()

    // Select timeframe
    const timeframeSelect = wrapper.find('[data-testid="timeframe-select"]')
    const triggerButton = timeframeSelect.find('.dropdown-trigger')
    await triggerButton.trigger('click')
    await nextTick()
    
    const options = timeframeSelect.findAll('.dropdown-option')
    const option1h = options.find(opt => opt.text().includes('1h'))
    expect(option1h.exists()).toBe(true)
    await option1h.trigger('click')
    await nextTick()

    // Set form fields that don't have data-testid (for now, keeping direct assignment for nested/advanced fields)
    wrapper.vm.form.indicators.maPeriods = '20, 50'
    wrapper.vm.form.indicators.rsiPeriod = 14
    wrapper.vm.form.indicators.enableMacd = true
    wrapper.vm.form.indicators.enableBb = false
    wrapper.vm.form.promptType = 'detailed'
    wrapper.vm.form.limit = 500
    await nextTick()

    await wrapper.vm.handleAnalyze()
    await nextTick()

    expect(chartAnalyzerAPI.analyzeSingle).toHaveBeenCalledWith(
      'BTC/USDT',
      '1h',
      expect.objectContaining({
        indicators: expect.objectContaining({
          ma_periods: [20, 50],
          rsi_period: 14,
          enable_macd: true,
          enable_bb: false,
        }),
        promptType: 'detailed',
        limit: 500,
      })
    )
  })

  it('should call analyzeMulti API in multi mode', async () => {
    const mockResponse = { session_id: 'analyze-session-456' }
    chartAnalyzerAPI.analyzeMulti.mockResolvedValue(mockResponse)

    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    // Select multi mode
    const multiModeButton = wrapper.find('[data-testid="mode-multi-button"]')
    await multiModeButton.trigger('click')
    await nextTick()

    // Set symbol
    const symbolInput = wrapper.find('[data-testid="symbol-input"]')
    await symbolInput.setValue('ETH/USDT')
    await nextTick()

    // Set timeframes
    const timeframesInput = wrapper.find('[data-testid="timeframes-input"]')
    await timeframesInput.setValue('15m, 1h, 4h')
    await nextTick()

    // Set form fields that don't have data-testid (for now, keeping direct assignment for nested/advanced fields)
    wrapper.vm.form.indicators.maPeriods = '20, 50, 200'
    wrapper.vm.form.indicators.rsiPeriod = 14
    wrapper.vm.form.limit = 300
    await nextTick()

    await wrapper.vm.handleAnalyze()
    await nextTick()

    expect(chartAnalyzerAPI.analyzeMulti).toHaveBeenCalledWith(
      'ETH/USDT',
      ['15m', '1h', '4h'],
      expect.objectContaining({
        indicators: expect.objectContaining({
          ma_periods: [20, 50, 200],
          rsi_period: 14,
        }),
        limit: 300,
      })
    )
  })

  it('should start log polling when session_id is returned', async () => {
    const mockResponse = { data: { session_id: 'analyze-session-123' } }
    chartAnalyzerAPI.analyzeSingle.mockResolvedValue(mockResponse)

    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    wrapper.vm.mode = 'single'
    wrapper.vm.form.symbol = 'BTC/USDT'
    wrapper.vm.form.timeframe = '1h'
    await nextTick()

    await wrapper.vm.handleAnalyze()
    await nextTick()

    // Verify LogPoller was instantiated
    expect(wrapper.vm.logPoller).toBeDefined()
    expect(wrapper.vm.logPoller).not.toBeNull()
    expect(wrapper.vm.logPoller.startPolling).toHaveBeenCalled()
  })

  it('should handle analysis errors', async () => {    const error = new Error('Analysis failed')
    chartAnalyzerAPI.analyzeSingle.mockRejectedValue(error)

    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    wrapper.vm.mode = 'single'
    wrapper.vm.form.symbol = 'BTC/USDT'
    wrapper.vm.form.timeframe = '1h'
    await nextTick()

    await wrapper.vm.handleAnalyze()
    await nextTick()

    expect(wrapper.vm.error).toBe('Analysis failed')
    expect(wrapper.vm.loading).toBe(false)
  })

  it('should display single timeframe results', async () => {
    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    wrapper.vm.mode = 'single'
    wrapper.vm.result = {
      symbol: 'BTC/USDT',
      timeframe: '1h',
      signal: 'LONG',
      confidence: 0.85,
      analysis: 'Test analysis',
      chart_url: '/static/charts/btc_1h.png',
    }
    await nextTick()

    expect(wrapper.text()).toContain('LONG')
    expect(wrapper.text()).toContain('85%')
    expect(wrapper.text()).toContain('Test analysis')
  })

  it('should display multi-timeframe results', async () => {
    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    wrapper.vm.mode = 'multi'
    wrapper.vm.result = {
      symbol: 'ETH/USDT',
      aggregated: {
        signal: 'SHORT',
        confidence: 0.75,
      },
      timeframes_results: {
        '15m': { signal: 'SHORT', confidence: 0.7 },
        '1h': { signal: 'SHORT', confidence: 0.8 },
      },
    }
    await nextTick()

    expect(wrapper.text()).toContain('SHORT')
    expect(wrapper.text()).toContain('75%')
    expect(wrapper.text()).toContain('15m')
    expect(wrapper.text()).toContain('1h')
  })

  it('should parse indicators correctly', async () => {
    const mockResponse = { session_id: 'test-session' }
    chartAnalyzerAPI.analyzeSingle.mockResolvedValue(mockResponse)

    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    wrapper.vm.mode = 'single'
    wrapper.vm.form.symbol = 'BTC/USDT'
    wrapper.vm.form.timeframe = '1h'
    wrapper.vm.form.indicators.maPeriods = '20, 50, 200'
    wrapper.vm.form.indicators.rsiPeriod = 14
    wrapper.vm.form.indicators.enableMacd = true
    wrapper.vm.form.indicators.enableBb = false
    await nextTick()

    await wrapper.vm.handleAnalyze()
    await nextTick()

    expect(chartAnalyzerAPI.analyzeSingle).toHaveBeenCalledWith(
      'BTC/USDT',
      '1h',
      expect.objectContaining({
        indicators: {
          ma_periods: [20, 50, 200],
          rsi_period: 14,
          enable_macd: true,
          enable_bb: false,
        },
      })
    )
  })

  it('should handle custom prompt when promptType is custom', async () => {
    const mockResponse = { session_id: 'test-session' }
    chartAnalyzerAPI.analyzeSingle.mockResolvedValue(mockResponse)

    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    wrapper.vm.mode = 'single'
    wrapper.vm.form.symbol = 'BTC/USDT'
    wrapper.vm.form.timeframe = '1h'
    wrapper.vm.form.promptType = 'custom'
    wrapper.vm.form.customPrompt = 'Custom analysis prompt'
    await nextTick()

    await wrapper.vm.handleAnalyze()
    await nextTick()

    expect(chartAnalyzerAPI.analyzeSingle).toHaveBeenCalledWith(
      'BTC/USDT',
      '1h',
      expect.objectContaining({
        customPrompt: 'Custom analysis prompt',
      })
    )
  })

  it('should handle mixed API response with both session_id and direct result data', async () => {
    // Mock response containing both session_id and direct result data
    const mockResponse = {
      data: {
        session_id: 'analyze-session-mixed-123',
        symbol: 'BTC/USDT',
        signal: 'LONG',
        confidence: 0.85,
      }
    }
    chartAnalyzerAPI.analyzeSingle.mockResolvedValue(mockResponse)

    // LogPoller is already mocked at the top of the file
    const logPollerMock = vi.mocked(LogPoller)

    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    wrapper.vm.mode = 'single'
    wrapper.vm.form.symbol = 'BTC/USDT'
    wrapper.vm.form.timeframe = '1h'
    await nextTick()

    await wrapper.vm.handleAnalyze()
    await nextTick()

    // Assert that polling is started (logPoller defined)
    expect(wrapper.vm.logPoller).toBeDefined()
    expect(wrapper.vm.logPoller).not.toBeNull()
    
    // LogPoller mock was called (constructor is mocked at top of file)
    expect(logPollerMock).toHaveBeenCalled()
    
    // Assert that result is not set directly (should wait for polling to complete)
    expect(wrapper.vm.result).toBeNull()
  })

  it('should correctly manage loading state during analysis', async () => {
    let resolvePromise
    const controlledPromise = new Promise(resolve => {
      resolvePromise = resolve
    })
    chartAnalyzerAPI.analyzeSingle.mockReturnValue(controlledPromise)

    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    wrapper.vm.mode = 'single'
    wrapper.vm.form.symbol = 'BTC/USDT'
    wrapper.vm.form.timeframe = '1h'
    await nextTick()

    // Start analysis (async, don't await yet)
    const analyzePromise = wrapper.vm.handleAnalyze()
    
    // Wait a bit to ensure loading is set
    await nextTick()
    
    // Assert loading is true before resolving
    expect(wrapper.vm.loading).toBe(true)

    // Resolve the promise with response that has no session_id (old API format)
    // This will set loading to false immediately
    resolvePromise({ symbol: 'BTC/USDT', signal: 'LONG', confidence: 0.85 })
    
    // Wait for handleAnalyze to complete
    await analyzePromise
    await nextTick()

    // Assert loading is false after resolution (when no session_id, loading is set to false)
    expect(wrapper.vm.loading).toBe(false)
    expect(wrapper.vm.result).toEqual({ symbol: 'BTC/USDT', signal: 'LONG', confidence: 0.85 })
  })

  it('should clear error when starting new analysis and after success', async () => {
    // First, simulate a failing analyzeSingle
    const error = new Error('First analysis failed')
    chartAnalyzerAPI.analyzeSingle.mockRejectedValueOnce(error)

    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    wrapper.vm.mode = 'single'
    wrapper.vm.form.symbol = 'BTC/USDT'
    wrapper.vm.form.timeframe = '1h'
    await nextTick()

    // First call - should fail
    await wrapper.vm.handleAnalyze()
    await nextTick()

    // Assert error is set after the first failure
    expect(wrapper.vm.error).toBe('First analysis failed')
    expect(wrapper.vm.loading).toBe(false)

    // Now mock a succeeding call
    const successResponse = { data: { session_id: 'success-session-789' } }
    chartAnalyzerAPI.analyzeSingle.mockResolvedValueOnce(successResponse)

    // Call handleAnalyze again
    await wrapper.vm.handleAnalyze()
    await nextTick()

    // After handleAnalyze with session_id, loading should be true (polling active)
    await nextTick()
    expect(wrapper.vm.error).toBeNull() // Component clears error when handleAnalyze starts
    expect(wrapper.vm.loading).toBe(true) // Loading true while polling
    expect(wrapper.vm.logPoller).toBeDefined()
    expect(wrapper.vm.logPoller).not.toBeNull()
  })

  it('should handle invalid indicator format gracefully', async () => {
    const mockResponse = { session_id: 'test-session-invalid' }
    chartAnalyzerAPI.analyzeSingle.mockResolvedValue(mockResponse)

    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    wrapper.vm.mode = 'single'
    wrapper.vm.form.symbol = 'BTC/USDT'
    wrapper.vm.form.timeframe = '1h'
    
    // Test with malformed maPeriods: "20,, 50" (should normalize to [20, 50])
    wrapper.vm.form.indicators.maPeriods = '20,, 50'
    await nextTick()

    // Should not throw
    await expect(wrapper.vm.handleAnalyze()).resolves.not.toThrow()
    await nextTick()

    // Assert the component normalized the value - should filter out empty strings and NaN
    expect(chartAnalyzerAPI.analyzeSingle).toHaveBeenCalledWith(
      'BTC/USDT',
      '1h',
      expect.objectContaining({
        indicators: expect.objectContaining({
          ma_periods: [20, 50], // Empty string between commas should be filtered out
        }),
      })
    )

    // Reset mock and test with "abc" (should normalize to empty array, then use default)
    chartAnalyzerAPI.analyzeSingle.mockClear()
    chartAnalyzerAPI.analyzeSingle.mockResolvedValue(mockResponse)
    
    wrapper.vm.form.indicators.maPeriods = 'abc'
    await nextTick()

    // Should not throw
    await expect(wrapper.vm.handleAnalyze()).resolves.not.toThrow()
    await nextTick()

    // Assert the component handled invalid input - all NaN values filtered out
    // Note: empty array is passed (not default [20, 50, 200]) because maPeriods is truthy
    expect(chartAnalyzerAPI.analyzeSingle).toHaveBeenCalledWith(
      'BTC/USDT',
      '1h',
      expect.objectContaining({
        indicators: expect.objectContaining({
          ma_periods: [], // "abc" parsed results in empty array after filtering NaN
        }),
      })
    )
  })

  it('should stop polling on component unmount', async () => {
    const mockResponse = { data: { session_id: 'analyze-session-123' } }
    chartAnalyzerAPI.analyzeSingle.mockResolvedValue(mockResponse)

    const wrapper = mount(ChartAnalyzer, {
      global: {
        plugins: [i18n],
      },
    })

    // Set up form data
    wrapper.vm.mode = 'single'
    wrapper.vm.form.symbol = 'BTC/USDT'
    wrapper.vm.form.timeframe = '1h'
    await nextTick()

    // Trigger analysis to create logPoller
    await wrapper.vm.handleAnalyze()
    await nextTick()

    // Verify logPoller was created
    expect(wrapper.vm.logPoller).toBeDefined()
    expect(wrapper.vm.logPoller).not.toBeNull()

    // Spy on stopPolling method before unmount
    const stopPollingSpy = vi.spyOn(wrapper.vm.logPoller, 'stopPolling')

    // Unmount component
    wrapper.unmount()

    // Assert stopPolling was called
    expect(stopPollingSpy).toHaveBeenCalled()
  })
})
