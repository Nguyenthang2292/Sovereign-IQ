const BASE_URL = '/api'

export interface OHLCV {
    time: number
    open: number
    high: number
    low: number
    close: number
    volume: number
}

interface ApiResponse<T> {
    success: boolean
    data?: T
    symbols?: string[]
    timeframes?: string[]
    error?: string
}

export interface ATCSignalParams {
    limit?: number
    ema_len?: number
    hma_len?: number
    wma_len?: number
    dema_len?: number
    lsma_len?: number
    kama_len?: number
    robustness?: number
    lambda_param?: number
    decay?: number
    cutout?: number
}

export interface MovingAverageParams {
    limit?: number
    ema_len?: number
    hma_len?: number
    wma_len?: number
    dema_len?: number
    lsma_len?: number
    kama_len?: number
    robustness?: number
}

export const api = {
    async fetchOHLCV(symbol: string, timeframe: string, limit: number): Promise<OHLCV[]> {
        const params = new URLSearchParams({
            symbol,
            timeframe,
            limit: limit.toString()
        })

        const response = await fetch(`${BASE_URL}/ohlcv?${params}`)
        const data: ApiResponse<OHLCV[]> = await response.json()

        if (!data.success) {
            throw new Error(data.error || 'Failed to fetch OHLCV data')
        }

        return data.data!
    },

    async fetchATCSignals(symbol: string, timeframe: string, params: ATCSignalParams = {}): Promise<any> {
        const queryParams = new URLSearchParams({
            symbol,
            timeframe,
            limit: (params.limit || 1500).toString()
        })

        if (params.ema_len) queryParams.append('ema_len', params.ema_len.toString())
        if (params.hma_len) queryParams.append('hma_len', params.hma_len.toString())
        if (params.wma_len) queryParams.append('wma_len', params.wma_len.toString())
        if (params.dema_len) queryParams.append('dema_len', params.dema_len.toString())
        if (params.lsma_len) queryParams.append('lsma_len', params.lsma_len.toString())
        if (params.kama_len) queryParams.append('kama_len', params.kama_len.toString())
        if (params.robustness) queryParams.append('robustness', params.robustness.toString())
        if (params.lambda_param) queryParams.append('lambda_param', params.lambda_param.toString())
        if (params.decay) queryParams.append('decay', params.decay.toString())
        if (params.cutout) queryParams.append('cutout', params.cutout.toString())

        const response = await fetch(`${BASE_URL}/atc-signals?${queryParams}`)
        const data: ApiResponse<any> = await response.json()

        if (!data.success) {
            throw new Error(data.error || 'Failed to compute ATC signals')
        }

        return data.data
    },

    async fetchMovingAverages(symbol: string, timeframe: string, params: MovingAverageParams = {}): Promise<any> {
        const queryParams = new URLSearchParams({
            symbol,
            timeframe,
            limit: (params.limit || 1500).toString()
        })

        if (params.ema_len) queryParams.append('ema_len', params.ema_len.toString())
        if (params.hma_len) queryParams.append('hma_len', params.hma_len.toString())
        if (params.wma_len) queryParams.append('wma_len', params.wma_len.toString())
        if (params.dema_len) queryParams.append('dema_len', params.dema_len.toString())
        if (params.lsma_len) queryParams.append('lsma_len', params.lsma_len.toString())
        if (params.kama_len) queryParams.append('kama_len', params.kama_len.toString())
        if (params.robustness) queryParams.append('robustness', params.robustness.toString())

        const response = await fetch(`${BASE_URL}/moving-averages?${queryParams}`)
        const data: ApiResponse<any> = await response.json()

        if (!data.success) {
            throw new Error(data.error || 'Failed to compute moving averages')
        }

        return data.data
    },

    async fetchSymbols(): Promise<string[]> {
        const response = await fetch(`${BASE_URL}/symbols`)
        const data: ApiResponse<any> = await response.json()

        if (!data.success) {
            throw new Error(data.error || 'Failed to fetch symbols')
        }

        return data.symbols!
    },

    async fetchTimeframes(): Promise<string[]> {
        const response = await fetch(`${BASE_URL}/timeframes`)
        const data: ApiResponse<any> = await response.json()

        if (!data.success) {
            throw new Error(data.error || 'Failed to fetch timeframes')
        }

        return data.timeframes!
    }
}
