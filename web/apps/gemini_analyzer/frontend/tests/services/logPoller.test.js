/**
 * Tests for LogPoller service
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { LogPoller } from '../../src/services/logPoller'
import { logsAPI } from '../../src/services/api'
import { batchScannerAPI } from '../../src/services/api'
import { chartAnalyzerStatusAPI } from '../../src/services/api'

// Mock API services
vi.mock('../../src/services/api', () => ({
  logsAPI: {
    getLogs: vi.fn(),
  },
  batchScannerAPI: {
    getBatchScanStatus: vi.fn(),
  },
  chartAnalyzerStatusAPI: {
    getAnalyzeStatus: vi.fn(),
  },
}))

describe('LogPoller', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('Constructor', () => {
    it('should initialize with correct parameters', () => {
      const onLogUpdate = vi.fn()
      const onStatusUpdate = vi.fn()
      const onComplete = vi.fn()

      const poller = new LogPoller('session-123', 'scan', onLogUpdate, onStatusUpdate, onComplete)

      expect(poller.sessionId).toBe('session-123')
      expect(poller.commandType).toBe('scan')
      expect(poller.onLogUpdate).toBe(onLogUpdate)
      expect(poller.onStatusUpdate).toBe(onStatusUpdate)
      expect(poller.onComplete).toBe(onComplete)
      expect(poller.logOffset).toBe(0)
      expect(poller.isPolling).toBe(false)
      expect(poller.allLogs).toEqual([])
    })

    it('should use default callbacks when not provided', () => {
      const poller = new LogPoller('session-123')

      expect(poller.onLogUpdate).toBeDefined()
      expect(poller.onStatusUpdate).toBeDefined()
      expect(poller.onComplete).toBeDefined()
    })
  })

  describe('startPolling', () => {
    it('should start polling logs and status', async () => {
      const onLogUpdate = vi.fn()
      const onStatusUpdate = vi.fn()
      const onComplete = vi.fn()

      logsAPI.getLogs.mockResolvedValue({
        success: true,
        logs: 'Line 1\nLine 2',
        offset: 20,
        has_more: true,
      })

      batchScannerAPI.getBatchScanStatus.mockResolvedValue({
        success: true,
        status: 'running',
      })

      const poller = new LogPoller('session-123', 'scan', onLogUpdate, onStatusUpdate, onComplete)
      poller.startPolling()

      expect(poller.isPolling).toBe(true)

      // Fast-forward time to trigger intervals
      await vi.advanceTimersByTimeAsync(500)

      expect(logsAPI.getLogs).toHaveBeenCalledWith('session-123', 0, 'scan', expect.objectContaining({
        signal: expect.any(AbortSignal)
      }))
      expect(onLogUpdate).toHaveBeenCalled()

      await vi.advanceTimersByTimeAsync(500)

      expect(batchScannerAPI.getBatchScanStatus).toHaveBeenCalledWith('session-123', expect.objectContaining({
        signal: expect.any(AbortSignal)
      }))

      poller.stopPolling()
    })

    it('should not start polling if already polling', async () => {
      const poller = new LogPoller('session-123')

      // Actually start polling to begin intervals
      poller.startPolling()
      const initialLogInterval = poller.logInterval
      const initialStatusInterval = poller.statusInterval
      expect(poller.isPolling).toBe(true)
      expect(initialLogInterval).not.toBeNull()
      expect(initialStatusInterval).not.toBeNull()

      // Attempt to start polling again while already polling
      poller.startPolling()

      // Ensure the intervals were not replaced/cleared (same references)
      expect(poller.logInterval).toBe(initialLogInterval)
      expect(poller.statusInterval).toBe(initialStatusInterval)

      poller.stopPolling()
    })

    it('should handle log updates correctly', async () => {
      const onLogUpdate = vi.fn()

      logsAPI.getLogs
        .mockResolvedValueOnce({
          success: true,
          logs: 'Line 1\nLine 2',
          offset: 20,
        })
        .mockResolvedValueOnce({
          success: true,
          logs: 'Line 3',
          offset: 30,
        })

      batchScannerAPI.getBatchScanStatus.mockResolvedValue({
        success: true,
        status: 'running',
      })

      const poller = new LogPoller('session-123', 'scan', onLogUpdate)
      poller.startPolling()

      await vi.advanceTimersByTimeAsync(500)
      expect(onLogUpdate).toHaveBeenCalledWith(['Line 1', 'Line 2'], ['Line 1', 'Line 2'])

      await vi.advanceTimersByTimeAsync(500)
      expect(onLogUpdate).toHaveBeenCalledWith(['Line 3'], ['Line 1', 'Line 2', 'Line 3'])

      poller.stopPolling()
    })

    it('should stop polling when status is completed', async () => {
      const onComplete = vi.fn()

      logsAPI.getLogs.mockResolvedValue({
        success: true,
        logs: '',
        offset: 0,
      })

      batchScannerAPI.getBatchScanStatus.mockResolvedValue({
        success: true,
        status: 'completed',
        result: { summary: {} },
      })

      const poller = new LogPoller('session-123', 'scan', null, null, onComplete)
      poller.startPolling()

      await vi.advanceTimersByTimeAsync(1000)

      expect(poller.isPolling).toBe(false)
      expect(onComplete).toHaveBeenCalledWith({ summary: {} }, null)
    })

    it('should stop polling when status is error', async () => {
      const onComplete = vi.fn()

      logsAPI.getLogs.mockResolvedValue({
        success: true,
        logs: '',
        offset: 0,
      })

      batchScannerAPI.getBatchScanStatus.mockResolvedValue({
        success: true,
        status: 'error',
        error: 'Scan failed',
      })

      const poller = new LogPoller('session-123', 'scan', null, null, onComplete)
      poller.startPolling()

      await vi.advanceTimersByTimeAsync(1000)

      expect(poller.isPolling).toBe(false)
      expect(onComplete).toHaveBeenCalledWith(null, 'Scan failed')
    })

    it('should use chartAnalyzerStatusAPI for analyze command type', async () => {
      const onStatusUpdate = vi.fn()

      logsAPI.getLogs.mockResolvedValue({
        success: true,
        logs: '',
        offset: 0,
      })

      chartAnalyzerStatusAPI.getAnalyzeStatus.mockResolvedValue({
        success: true,
        status: 'running',
      })

      const poller = new LogPoller('session-123', 'analyze', null, onStatusUpdate)
      poller.startPolling()

      await vi.advanceTimersByTimeAsync(1000)

      expect(chartAnalyzerStatusAPI.getAnalyzeStatus).toHaveBeenCalledWith('session-123', expect.objectContaining({
        signal: expect.any(AbortSignal)
      }))
      expect(batchScannerAPI.getBatchScanStatus).not.toHaveBeenCalled()

      poller.stopPolling()
    })

    it('should handle log polling errors gracefully', async () => {
      const onLogUpdate = vi.fn()
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

      logsAPI.getLogs.mockRejectedValue(new Error('Network error'))
      batchScannerAPI.getBatchScanStatus.mockResolvedValue({
        success: true,
        status: 'running',
      })

      const poller = new LogPoller('session-123', 'scan', onLogUpdate)
      poller.startPolling()

      await vi.advanceTimersByTimeAsync(500)

      expect(consoleErrorSpy).toHaveBeenCalled()
      expect(poller.isPolling).toBe(true) // Should continue polling

      poller.stopPolling()
      consoleErrorSpy.mockRestore()
    })

    it('should handle status polling errors gracefully', async () => {
      const onStatusUpdate = vi.fn()
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

      logsAPI.getLogs.mockResolvedValue({
        success: true,
        logs: '',
        offset: 0,
      })

      batchScannerAPI.getBatchScanStatus.mockRejectedValue(new Error('Status error'))

      const poller = new LogPoller('session-123', 'scan', null, onStatusUpdate)
      poller.startPolling()

      await vi.advanceTimersByTimeAsync(1000)

      expect(consoleErrorSpy).toHaveBeenCalled()
      expect(poller.isPolling).toBe(true) // Should continue polling

      poller.stopPolling()
      consoleErrorSpy.mockRestore()
    })
  })

  describe('stopPolling', () => {
    /**
     * Helper function to capture AbortSignals from API calls
     * @returns {{ logSignal: AbortSignal, statusSignal: AbortSignal }}
     */
    function captureAbortSignals() {
      const logCallArgs = logsAPI.getLogs.mock.calls[0]
      const logSignal = logCallArgs?.[3]?.signal
      const statusCallArgs = batchScannerAPI.getBatchScanStatus.mock.calls[0]
      const statusSignal = statusCallArgs?.[1]?.signal
      return { logSignal, statusSignal }
    }

    it('should stop polling and clear intervals', async () => {
      const poller = new LogPoller('session-123')

      // Stub APIs to prevent errors and infinite loops
      logsAPI.getLogs.mockResolvedValue({
        success: true,
        logs: '',
        offset: 0,
      })
      batchScannerAPI.getBatchScanStatus.mockResolvedValue({
        success: true,
        status: 'running',
      })

      poller.startPolling()
      // Advance time to trigger both log and status polling
      await vi.advanceTimersByTimeAsync(1000)

      // Capture the AbortSignals before stopping
      const { logSignal, statusSignal } = captureAbortSignals()

      // Verify signals exist and are not aborted yet
      expect(logSignal).toBeDefined()
      expect(logSignal.aborted).toBe(false)
      expect(statusSignal).toBeDefined()
      expect(statusSignal.aborted).toBe(false)

      poller.stopPolling()

      expect(poller.isPolling).toBe(false)
      expect(poller.logInterval).toBeNull()
      expect(poller.statusInterval).toBeNull()

      // Verify signals are aborted after stopping
      expect(logSignal.aborted).toBe(true)
      expect(statusSignal.aborted).toBe(true)
    })

    it('should abort pending requests when stopping', async () => {
      const poller = new LogPoller('session-123')

      logsAPI.getLogs.mockResolvedValue({
        success: true,
        logs: '',
        offset: 0,
      })
      batchScannerAPI.getBatchScanStatus.mockResolvedValue({
        success: true,
        status: 'running',
      })

      poller.startPolling()
      // Advance timers to trigger both log and status intervals
      await vi.advanceTimersByTimeAsync(1000)

      // Capture the AbortSignals that were passed to both APIs
      expect(logsAPI.getLogs).toHaveBeenCalled()
      expect(batchScannerAPI.getBatchScanStatus).toHaveBeenCalled()
      
      const { logSignal, statusSignal } = captureAbortSignals()

      expect(logSignal).toBeDefined()
      expect(logSignal.aborted).toBe(false)
      expect(statusSignal).toBeDefined()
      expect(statusSignal.aborted).toBe(false)

      poller.stopPolling()

      expect(logSignal.aborted).toBe(true)
      expect(statusSignal.aborted).toBe(true)
    })

    it('should not stop if not polling', () => {
      const poller = new LogPoller('session-123')
      poller.isPolling = false

      poller.stopPolling()

      expect(poller.isPolling).toBe(false)
    })
  })

  describe('getAllLogs', () => {
    it('should return all accumulated logs', async () => {
      logsAPI.getLogs
        .mockResolvedValueOnce({
          success: true,
          logs: 'Log 1\nLog 2',
          offset: 20,
        })
        .mockResolvedValueOnce({
          success: true,
          logs: 'Log 3',
          offset: 30,
        })

      batchScannerAPI.getBatchScanStatus.mockResolvedValue({
        success: true,
        status: 'running',
      })

      const poller = new LogPoller('session-123', 'scan')
      poller.startPolling()

      await vi.advanceTimersByTimeAsync(500)
      await vi.advanceTimersByTimeAsync(500)

      const allLogs = poller.getAllLogs()
      expect(allLogs).toEqual(['Log 1', 'Log 2', 'Log 3'])

      poller.stopPolling()
    })
  })

  describe('resetOffset', () => {
    it('should reset log offset and clear all logs', async () => {
      logsAPI.getLogs.mockResolvedValue({
        success: true,
        logs: 'Log 1',
        offset: 20,
      })

      batchScannerAPI.getBatchScanStatus.mockResolvedValue({
        success: true,
        status: 'running',
      })

      const poller = new LogPoller('session-123', 'scan')
      poller.startPolling()

      await vi.advanceTimersByTimeAsync(500)

      expect(poller.logOffset).toBe(20)
      expect(poller.getAllLogs().length).toBeGreaterThan(0)

      poller.resetOffset()

      expect(poller.logOffset).toBe(0)
      expect(poller.getAllLogs()).toEqual([])

      poller.stopPolling()
    })
  })
})

