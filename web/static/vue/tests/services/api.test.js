/**
 * Tests for API service
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import axios from 'axios'

// Mock axios module BEFORE importing API
vi.mock('axios', async () => {
  const actual = await vi.importActual('axios')
  const mockInstance = {
    post: vi.fn(),
    get: vi.fn(),
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() },
    },
  }
  return {
    default: {
      ...actual.default,
      create: vi.fn(() => mockInstance),
    },
    mockInstance, // Export for use in tests
  }
})

// Now import the API module (after mocking)
import { chartAnalyzerAPI, batchScannerAPI, logsAPI, chartAnalyzerStatusAPI } from '../../src/services/api'

// Get the mock instance
const mockAxiosInstance = vi.mocked(axios.create)()

describe('API Service', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Setup default mock behavior: return data directly (interceptor extracts response.data)
    mockAxiosInstance.post.mockImplementation(async (url, data) => {
      // Simulate axios: always resolve to { data: ... } so test-specific mocks and defaults alike
      return { data: { session_id: 'test-session' } }
    })
    mockAxiosInstance.get.mockImplementation(async (url, config) => {
      return { data: { success: true } }
    })
  })

  describe('chartAnalyzerAPI', () => {
    it('should call analyzeSingle with correct parameters', async () => {
      mockAxiosInstance.post.mockResolvedValueOnce({ data: { session_id: 'test-session-123' } })

      const result = await chartAnalyzerAPI.analyzeSingle('BTC/USDT', '1h', {
        indicators: { ma_periods: [20, 50] },
        promptType: 'detailed',
        limit: 500,
      })

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/analyze/single', {
        symbol: 'BTC/USDT',
        timeframe: '1h',
        indicators: { ma_periods: [20, 50] },
        prompt_type: 'detailed',
        custom_prompt: null,
        limit: 500,
        chart_figsize: [16, 10],
        chart_dpi: 150,
        no_cleanup: false,
      })
    })

    it('should call analyzeMulti with correct parameters', async () => {
      mockAxiosInstance.post.mockResolvedValueOnce({ data: { session_id: 'test-session-456' } })

      await chartAnalyzerAPI.analyzeMulti('ETH/USDT', ['1h', '4h'], {
        indicators: { rsi_period: 14 },
        promptType: 'simple',
        limit: 300,
      })

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/analyze/multi', {
        symbol: 'ETH/USDT',
        timeframes: ['1h', '4h'],
        indicators: { rsi_period: 14 },
        prompt_type: 'simple',
        custom_prompt: null,
        limit: 300,
        chart_figsize: [16, 10],
        chart_dpi: 150,
        no_cleanup: false,
      })
    })

    it('should use default values when config is not provided', async () => {
      mockAxiosInstance.post.mockResolvedValueOnce({ data: { session_id: 'test-session' } })

      await chartAnalyzerAPI.analyzeSingle('BTC/USDT', '1h')

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/analyze/single', {
        symbol: 'BTC/USDT',
        timeframe: '1h',
        indicators: {},
        prompt_type: 'detailed',
        custom_prompt: null,
        limit: 500,
        chart_figsize: [16, 10],
        chart_dpi: 150,
        no_cleanup: false,
      })
    })
  })

  describe('batchScannerAPI', () => {
    it('should call scanMarket with correct parameters for single timeframe', async () => {
      mockAxiosInstance.post.mockResolvedValueOnce({ data: { session_id: 'scan-session-123' } })

      await batchScannerAPI.scanMarket({
        timeframe: '1h',
        maxSymbols: 10,
        limit: 500,
        cooldown: 2.5,
      })

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/batch/scan', {
        timeframe: '1h',
        max_symbols: 10,
        limit: 500,
        cooldown: 2.5,
        quote_currency: 'USDT',
        exchange_name: 'binance',
      })
    })

    it('should call scanMarket with correct parameters for multi-timeframe', async () => {
      mockAxiosInstance.post.mockResolvedValueOnce({ data: { session_id: 'scan-session-456' } })

      await batchScannerAPI.scanMarket({
        timeframes: ['15m', '1h', '4h'],
        maxSymbols: 20,
        chartsPerBatch: 5,
      })

      expect(mockAxiosInstance.post).toHaveBeenCalledWith('/batch/scan', {
        timeframes: ['15m', '1h', '4h'],
        max_symbols: 20,
        limit: 500,
        cooldown: 2.5,
        charts_per_batch: 5,
        quote_currency: 'USDT',
        exchange_name: 'binance',
      })
    })

    it('should not include charts_per_batch when null', async () => {
      mockAxiosInstance.post.mockResolvedValueOnce({ data: { session_id: 'scan-session' } })

      await batchScannerAPI.scanMarket({ timeframe: '1h' })

      const callArgs = mockAxiosInstance.post.mock.calls[0][1]
      expect(callArgs).not.toHaveProperty('charts_per_batch')
    })

    it('should call getResults with filename', async () => {
      mockAxiosInstance.get.mockResolvedValueOnce({ data: { results: {} } })

      await batchScannerAPI.getResults('results_2024.json')

      expect(mockAxiosInstance.get).toHaveBeenCalledWith('/batch/results/results_2024.json')
    })

    it('should call listResults', async () => {
      mockAxiosInstance.get.mockResolvedValueOnce({ data: { files: [] } })

      await batchScannerAPI.listResults()

      expect(mockAxiosInstance.get).toHaveBeenCalledWith('/batch/list')
    })

    it('should call getBatchScanStatus with sessionId', async () => {
      mockAxiosInstance.get.mockResolvedValueOnce({ data: { status: 'running' } })

      await batchScannerAPI.getBatchScanStatus('session-123')

      expect(mockAxiosInstance.get).toHaveBeenCalledWith('/batch/scan/session-123/status', {})
    })
  })

  describe('logsAPI', () => {
    it('should call getLogs with default parameters', async () => {
      mockAxiosInstance.get.mockResolvedValueOnce({ data: { logs: 'test log', offset: 100 } })

      await logsAPI.getLogs('session-123')

      expect(mockAxiosInstance.get).toHaveBeenCalledWith('/logs/session-123', {
        params: {
          offset: 0,
          command_type: 'scan',
        },
      })
    })

    it('should call getLogs with custom offset and commandType', async () => {
      mockAxiosInstance.get.mockResolvedValueOnce({ data: { logs: 'test log', offset: 200 } })

      await logsAPI.getLogs('session-456', 100, 'analyze')

      expect(mockAxiosInstance.get).toHaveBeenCalledWith('/logs/session-456', {
        params: {
          offset: 100,
          command_type: 'analyze',
        },
      })
    })
  })

  describe('chartAnalyzerStatusAPI', () => {
    it('should call getAnalyzeStatus with sessionId', async () => {
      mockAxiosInstance.get.mockResolvedValueOnce({ data: { status: 'completed', result: {} } })

      await chartAnalyzerStatusAPI.getAnalyzeStatus('analyze-session-123')

      expect(mockAxiosInstance.get).toHaveBeenCalledWith('/analyze/analyze-session-123/status', {})
    })
  })

  describe('API Error Handling', () => {
    it('should handle network errors', async () => {
      const networkError = {
        request: {},
        response: undefined,
        message: 'Network Error',
      }
      mockAxiosInstance.post.mockRejectedValueOnce(networkError)

      await expect(chartAnalyzerAPI.analyzeSingle('BTC/USDT', '1h')).rejects.toMatchObject({ message: 'Network Error' })
    })

    it('should handle server errors with detail message', async () => {
      const serverError = {
        response: {
          data: { detail: 'Invalid symbol format' },
        },
      }
      mockAxiosInstance.post.mockRejectedValueOnce(serverError)

      await expect(chartAnalyzerAPI.analyzeSingle('INVALID', '1h')).rejects.toMatchObject({ response: { data: { detail: 'Invalid symbol format' } } })
    })
  })
})
