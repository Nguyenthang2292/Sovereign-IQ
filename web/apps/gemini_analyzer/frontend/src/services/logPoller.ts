/**
 * Log Poller service for polling logs and status from backend.
 */

import { logsAPI, batchScannerAPI, chartAnalyzerStatusAPI } from './api'

// Maximum number of log lines to retain in memory
const MAX_LOG_LINES = 5000

export class LogPoller {
    private sessionId: string
    private commandType: string
    private onLogUpdate: (newLogs: string[], allLogs: string[]) => void
    private onStatusUpdate: (status: string, data: any) => void
    private onComplete: (result: any, error: string | null) => void

    private logOffset: number
    private isPolling: boolean
    private logInterval: ReturnType<typeof setInterval> | null
    private statusInterval: ReturnType<typeof setInterval> | null
    private allLogs: string[]
    private maxDurationTimeout: ReturnType<typeof setTimeout> | null

    // Cancellation mechanism
    private _retryToken: number
    private _retryTimers: ReturnType<typeof setTimeout>[]
    private _abortController: AbortController | null

    constructor(
        sessionId: string,
        commandType: string = 'scan',
        onLogUpdate?: (newLogs: string[], allLogs: string[]) => void,
        onStatusUpdate?: (status: string, data: any) => void,
        onComplete?: (result: any, error: string | null) => void
    ) {
        this.sessionId = sessionId
        this.commandType = commandType
        this.onLogUpdate = onLogUpdate || (() => { })
        this.onStatusUpdate = onStatusUpdate || (() => { })
        this.onComplete = onComplete || (() => { })

        this.logOffset = 0
        this.isPolling = false
        this.logInterval = null
        this.statusInterval = null
        this.allLogs = []
        this.maxDurationTimeout = null

        // Cancellation mechanism
        this._retryToken = 0
        this._retryTimers = []
        this._abortController = null
    }

    /**
     * Start polling logs and status
     */
    startPolling({ maxDuration = 60000 } = {}) { // maxDuration default: 1 minute (in ms)
        if (this.isPolling) {
            return
        }

        this.isPolling = true

        // Increment retry token to invalidate any stale retries
        this._retryToken++

        // Create new AbortController for this polling session
        this._abortController = new AbortController()

        // Clear any existing retry timers
        this._clearRetryTimers()

        // Set timeout to stop polling after maxDuration
        if (maxDuration > 0) {
            this.maxDurationTimeout = setTimeout(() => {
                this.stopPolling()
            }, maxDuration)
        }

        // Poll logs every 0.5 seconds
        this.logInterval = setInterval(async () => {
            // Check if polling was stopped
            if (!this.isPolling || !this._abortController) {
                return
            }

            try {
                const response = await logsAPI.getLogs(
                    this.sessionId,
                    this.logOffset,
                    this.commandType,
                    { signal: this._abortController.signal }
                );

                // Check again after async operation
                if (!this.isPolling) {
                    return
                }

                const logData = (response as any).data || response;
                if (logData.success && logData.logs) {
                    const newLogs = logData.logs;
                    if (newLogs && newLogs.length > 0) {
                        // Split by newlines and filter empty lines
                        const logLines = (newLogs as string).split('\n').filter(line => line.trim());

                        // Add new log lines
                        this.allLogs.push(...logLines);

                        // Keep allLogs only up to MAX_LOG_LINES (circular buffer style).
                        if (this.allLogs.length > MAX_LOG_LINES) {
                            this.allLogs = this.allLogs.slice(-MAX_LOG_LINES);
                        }

                        // Update offset
                        this.logOffset = logData.offset;

                        // Check before calling callback
                        if (this.isPolling) {
                            this.onLogUpdate(logLines, this.allLogs);
                        }
                    }
                }
            } catch (error: any) {
                // Ignore abort errors
                if (error.name === 'AbortError' || error.name === 'CanceledError') {
                    return
                }
                console.error('Error polling logs:', error);
            }
        }, 500); // 0.5 seconds

        // Poll status every 1 second
        this.statusInterval = setInterval(async () => {
            // Check if polling was stopped
            if (!this.isPolling || !this._abortController) {
                return
            }

            try {
                let statusResponse

                if (this.commandType === 'scan') {
                    statusResponse = await batchScannerAPI.getBatchScanStatus(this.sessionId, {
                        signal: this._abortController.signal
                    })
                } else {
                    statusResponse = await chartAnalyzerStatusAPI.getAnalyzeStatus(this.sessionId, {
                        signal: this._abortController.signal
                    })
                }

                // Check again after async operation
                if (!this.isPolling) {
                    return
                }

                const statusData = (statusResponse as any).data || statusResponse;
                if (statusData.success) {
                    const status = statusData.status

                    // Check before calling callback
                    if (this.isPolling) {
                        this.onStatusUpdate(status, statusData)
                    }

                    // If completed or error, stop polling
                    if (status === 'completed' || status === 'error') {
                        this.stopPolling()

                        // Capture current retry token to check staleness
                        const currentToken = this._retryToken

                        if (status === 'error') {
                            // Handle error status
                            const error = statusData.error || 'Unknown error'
                            // Check token before callback
                            if (this._retryToken === currentToken) {
                                this.onComplete(null, error)
                            }
                        } else if (status === 'completed') {
                            // Handle completed status
                            const result = statusData.result

                            if (result) {
                                // Result available immediately
                                console.log('Result available immediately:', result)
                                console.log('Result type:', typeof result, 'Keys:', result ? Object.keys(result) : 'none')
                                // Check token before callback
                                if (this._retryToken === currentToken) {
                                    this.onComplete(result, null)
                                }
                            } else {
                                console.warn('Status is completed but no result in statusData:', statusData)
                                // Result might not be set yet, retry with exponential backoff
                                // Increase retries and delay to handle race conditions
                                let retryCount = 0
                                const maxRetries = 10 // Increased from 5 to 10
                                const retryDelay = 1000 // Start with 1 second (increased from 500ms)

                                const retryCheck = async () => {
                                    // Check if token is still valid (not stale)
                                    if (this._retryToken !== currentToken) {
                                        console.log('Retry cancelled: token mismatch (new session started)')
                                        return
                                    }

                                    // Check if polling was stopped
                                    if (!this.isPolling || !this._abortController) {
                                        console.log('Retry cancelled: polling stopped')
                                        return
                                    }

                                    try {
                                        let retryResponse
                                        if (this.commandType === 'scan') {
                                            retryResponse = await batchScannerAPI.getBatchScanStatus(this.sessionId, {
                                                signal: this._abortController.signal
                                            })
                                        } else {
                                            retryResponse = await chartAnalyzerStatusAPI.getAnalyzeStatus(this.sessionId, {
                                                signal: this._abortController.signal
                                            })
                                        }

                                        // Check token and polling state again after async operation
                                        if (this._retryToken !== currentToken || !this.isPolling) {
                                            return
                                        }

                                        const retryData = (retryResponse as any).data || retryResponse;
                                        if (retryData.success && retryData.status === 'completed') {
                                            const retryResult = retryData.result
                                            if (retryResult) {
                                                console.log('Result retrieved after retry:', retryCount + 1, retryResult)
                                                // Check token before callback
                                                if (this._retryToken === currentToken) {
                                                    this.onComplete(retryResult, null)
                                                }
                                                return
                                            } else if (retryCount < maxRetries) {
                                                console.warn(`Retry ${retryCount + 1}: Status is completed but no result yet`, retryData)
                                                // Retry again with exponential backoff
                                                retryCount++
                                                const delay = retryDelay * Math.pow(2, retryCount - 1)
                                                console.log(`No result yet, retrying in ${delay}ms (attempt ${retryCount + 1}/${maxRetries})`)

                                                // Check token before scheduling retry
                                                if (this._retryToken === currentToken && this.isPolling) {
                                                    const timerId = setTimeout(retryCheck, delay)
                                                    this._retryTimers.push(timerId)
                                                }
                                                return
                                            }
                                        }
                                    } catch (err: any) {
                                        // Ignore abort errors
                                        if (err.name === 'AbortError' || err.name === 'CanceledError') {
                                            return
                                        }

                                        console.error('Error retrying status check:', err)
                                        if (retryCount < maxRetries && this._retryToken === currentToken && this.isPolling) {
                                            retryCount++
                                            const delay = retryDelay * Math.pow(2, retryCount - 1)
                                            const timerId = setTimeout(retryCheck, delay)
                                            this._retryTimers.push(timerId)
                                            return
                                        }
                                    }

                                    // If still no result after all retries, complete without error
                                    // Check token before callback
                                    if (this._retryToken === currentToken) {
                                        console.warn('Task completed but no result data after all retries:', statusData)
                                        this.onComplete(null, null)
                                    }
                                }

                                // Check token before scheduling initial retry
                                if (this._retryToken === currentToken) {
                                    const timerId = setTimeout(retryCheck, retryDelay)
                                    this._retryTimers.push(timerId)
                                }
                            }
                        }
                    }
                }
            } catch (error: any) {
                // Ignore abort errors
                if (error.name === 'AbortError' || error.name === 'CanceledError') {
                    return
                }
                console.error('Error polling status:', error)
                // Don't stop on error, might be temporary network issue
            }
        }, 1000) // 1 second
    }

    /**
     * Clear all retry timers
     */
    _clearRetryTimers() {
        this._retryTimers.forEach(timerId => {
            clearTimeout(timerId)
        })
        this._retryTimers = []
    }

    /**
     * Stop polling
     */
    stopPolling() {
        if (!this.isPolling) {
            return
        }

        this.isPolling = false

        // Abort any in-flight requests
        if (this._abortController) {
            this._abortController.abort()
            this._abortController = null
        }

        // Clear all retry timers
        this._clearRetryTimers()

        if (this.logInterval) {
            clearInterval(this.logInterval)
            this.logInterval = null
        }

        if (this.statusInterval) {
            clearInterval(this.statusInterval)
            this.statusInterval = null
        }

        if (this.maxDurationTimeout) {
            clearTimeout(this.maxDurationTimeout)
            this.maxDurationTimeout = null
        }
    }

    /**
     * Get all accumulated logs
     */
    getAllLogs() {
        return this.allLogs
    }

    /**
     * Reset log offset (useful for re-reading from beginning)
     */
    resetOffset() {
        this.logOffset = 0
        this.allLogs = []
    }
}

export default LogPoller
