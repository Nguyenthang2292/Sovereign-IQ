import axios, { AxiosInstance, AxiosResponse, AxiosRequestConfig } from 'axios'

// Types
export interface IndicatorsConfig {
    ma_periods?: number[]
    rsi_period?: number
    enable_macd?: boolean
    enable_bb?: boolean
    [key: string]: any // For backward compatibility and custom indicators
}

export interface AnalyzeConfig {
    indicators?: IndicatorsConfig
    promptType?: string
    customPrompt?: string | null
    limit?: number
    chartFigsize?: [number, number]
    chartDpi?: number
    noCleanup?: boolean
}

export interface BatchScanConfig {
    timeframe?: string | null
    timeframes?: string[] | null
    maxSymbols?: number | null
    limit?: number
    cooldown?: number
    chartsPerBatch?: number | null
    quoteCurrency?: string
    exchangeName?: string
}

export interface LogConfig {
    params?: Record<string, any>
    [key: string]: any
}

// Create axios instance with base configuration
// Axios instance for batch scans and long operations (longer timeout)
const api: AxiosInstance = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
    timeout: 300000, // 5 minutes for batch scans
    headers: {
        'Content-Type': 'application/json',
    },
})

// Axios instance for standard/quick operations (shorter timeout)
const quickApi: AxiosInstance = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
    timeout: 30000, // 30 seconds for fast operations
    headers: {
        'Content-Type': 'application/json',
    },
})

// Shared response interceptor handlers
const responseSuccessHandler = (response: AxiosResponse) => {
    // Return the full Axios response object instead of unwrapping .data
    return response
}

const responseErrorHandler = (error: any) => {
    if (error.response) {
        // Server responded with error
        const message = error.response.data?.detail || error.response.data?.message || 'An error occurred'
        return Promise.reject(new Error(message))
    } else if (error.request) {
        // Request made but no response
        return Promise.reject(new Error('No response from server. Please check your connection.'))
    } else {
        // Error in request setup
        return Promise.reject(error)
    }
}

// Response interceptor for api (long operations)
api.interceptors.response.use(responseSuccessHandler, responseErrorHandler)

// Response interceptor for quickApi (quick operations) - same error handling
quickApi.interceptors.response.use(responseSuccessHandler, responseErrorHandler)

// Chart Analyzer API
export const chartAnalyzerAPI = {
    /**
     * Analyze single symbol on single timeframe
     * @param {string} symbol - Trading symbol (e.g., 'BTC/USDT')
     * @param {string} timeframe - Timeframe (e.g., '1h', '4h', '1d')
     * @param {AnalyzeConfig} config - Configuration object
     * @returns {Promise} Analysis result
     */
    analyzeSingle(symbol: string, timeframe: string, config: AnalyzeConfig = {}) {
        // Input validation
        if (typeof symbol !== 'string' || symbol.trim() === '') {
            return Promise.reject(new Error('Symbol is required and must be a non-empty string.'));
        }
        if (typeof timeframe !== 'string' || timeframe.trim() === '') {
            return Promise.reject(new Error('Timeframe is required and must be a non-empty string.'));
        }

        const {
            indicators = {},
            promptType = 'detailed',
            customPrompt = null,
            limit = 500,
            chartFigsize = [16, 10],
            chartDpi = 150,
            noCleanup = false,
        } = config

        return api.post('/analyze/single', {
            symbol,
            timeframe,
            indicators,
            prompt_type: promptType,
            custom_prompt: customPrompt,
            limit,
            chart_figsize: chartFigsize,
            chart_dpi: chartDpi,
            no_cleanup: noCleanup,
        })
    },

    /**
     * Analyze single symbol across multiple timeframes
     * @param {string} symbol - Trading symbol (e.g., 'BTC/USDT')
     * @param {Array<string>} timeframes - List of timeframes (e.g., ['15m', '1h', '4h', '1d'])
     * @param {AnalyzeConfig} config - Configuration object
     * @returns {Promise} Multi-timeframe analysis result
     */
    analyzeMulti(symbol: string, timeframes: string[], config: AnalyzeConfig = {}) {
        // Input validation
        if (typeof symbol !== 'string' || symbol.trim() === '') {
            return Promise.reject(new Error('Symbol is required and must be a non-empty string.'));
        }
        if (!Array.isArray(timeframes) || timeframes.length === 0) {
            return Promise.reject(new Error('Timeframes is required and must be a non-empty array.'));
        }
        if (!timeframes.every(tf => typeof tf === 'string' && tf.trim() !== '')) {
            return Promise.reject(new Error('All timeframes must be non-empty strings.'));
        }

        const {
            indicators = {},
            promptType = 'detailed',
            customPrompt = null,
            limit = 500,
            chartFigsize = [16, 10],
            chartDpi = 150,
            noCleanup = false,
        } = config

        return api.post('/analyze/multi', {
            symbol,
            timeframes,
            indicators,
            prompt_type: promptType,
            custom_prompt: customPrompt,
            limit,
            chart_figsize: chartFigsize,
            chart_dpi: chartDpi,
            no_cleanup: noCleanup,
        })
    },

}

// Batch Scanner API
export const batchScannerAPI = {
    /**
     * Scan entire market (single or multi-timeframe)
     * @param {BatchScanConfig} config - Configuration object
     * @returns {Promise} Batch scan results
     */
    scanMarket(config: BatchScanConfig = {}) {
        const {
            timeframe = null,
            timeframes = null,
            maxSymbols = null,
            limit = 500,
            cooldown = 2.5,
            chartsPerBatch = null, // Auto-calculated by backend if null
            quoteCurrency = 'USDT',
            exchangeName = 'binance',
        } = config

        // Validation: At least one timeframe or timeframes must be provided
        const hasTimeframe = typeof timeframe === 'string' && timeframe.trim() !== '';
        const hasTimeframes = Array.isArray(timeframes) && timeframes.length > 0;
        if (!hasTimeframe && !hasTimeframes) {
            return Promise.reject(new Error('Either "timeframe" or "timeframes" must be provided for scanMarket.'));
        }

        // Construct the request body, omitting null/undefined values
        const requestBody: any = {
            limit,
            cooldown,
            quote_currency: quoteCurrency,
            exchange_name: exchangeName,
        };
        if (timeframe !== null && timeframe !== undefined && timeframe !== '') {
            requestBody.timeframe = timeframe;
        }
        if (Array.isArray(timeframes) && timeframes.length > 0) {
            requestBody.timeframes = timeframes;
        }
        if (maxSymbols !== null && maxSymbols !== undefined) {
            requestBody.max_symbols = maxSymbols;
        }
        if (chartsPerBatch !== null && chartsPerBatch !== undefined) {
            requestBody.charts_per_batch = chartsPerBatch;
        }

        return api.post('/batch/scan', requestBody);
    },

    /**
     * Get saved batch scan results by filename
     * @param {string} filename - Results JSON filename
     * @returns {Promise} Batch scan results
     */
    getResults(filename: string) {
        if (typeof filename !== 'string' || filename.trim() === '') {
            return Promise.reject(new Error('Filename is required and must be a non-empty string.'));
        }
        return quickApi.get(`/batch/results/${filename}`)
    },

    /**
     * List all available batch scan results
     * @returns {Promise} List of results files
     */
    listResults() {
        return quickApi.get('/batch/list')
    },

    /**
     * Get status of a batch scan task
     * @param {string} sessionId - Session ID from scanMarket
     * @param {Object} config - Optional axios config (e.g., signal for AbortController)
     * @returns {Promise} Status and results (if completed)
     */
    getBatchScanStatus(sessionId: string, config: AxiosRequestConfig = {}) {
        return quickApi.get(`/batch/scan/${sessionId}/status`, config)
    },

    /**
     * Cancel a running batch scan task
     * @param {string} sessionId - Session ID from scanMarket
     * @returns {Promise} Cancellation result
     */
    cancelBatchScan(sessionId: string) {
        return quickApi.post(`/batch/scan/${sessionId}/cancel`)
    },
}

// Logs API
export const logsAPI = {
    /**
     * Get log content from a log file
     * @param {string} sessionId - Session ID
     * @param {number} offset - Byte offset to start reading from
     * @param {string} commandType - Type of command: 'scan' or 'analyze'
     * @param {LogConfig} config - Optional axios config (e.g., signal for AbortController)
     * @returns {Promise} Log content, new offset, and has_more flag
     */
    getLogs(sessionId: string, offset: number = 0, commandType: string = 'scan', config: LogConfig = {}) {
        const { params: cfgParams, ...restConfig } = config
        return quickApi.get(`/logs/${sessionId}`, {
            ...restConfig,
            params: {
                offset,
                command_type: commandType,
                ...cfgParams,
            },
        })
    },
}

// Chart Analyzer Status API
export const chartAnalyzerStatusAPI = {
    /**
     * Get status of an analysis task
     * @param {string} sessionId - Session ID from analyzeSingle or analyzeMulti
     * @param {Object} config - Optional axios config (e.g., signal for AbortController)
     * @returns {Promise} Status and results (if completed)
     */
    getAnalyzeStatus(sessionId: string, config: AxiosRequestConfig = {}) {
        return quickApi.get(`/analyze/${sessionId}/status`, config)
    },

    /**
     * Cancel a running analysis task
     * @param {string} sessionId - Session ID from analyzeSingle or analyzeMulti
     * @returns {Promise} Cancellation result
     */
    cancelAnalysis(sessionId: string) {
        return quickApi.post(`/analyze/${sessionId}/cancel`)
    },
}

export default api
