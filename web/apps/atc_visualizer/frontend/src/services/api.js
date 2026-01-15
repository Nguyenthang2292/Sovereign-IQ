const BASE_URL = '/api'

export const api = {
  async fetchOHLCV(symbol, timeframe, limit) {
    const params = new URLSearchParams({
      symbol,
      timeframe,
      limit: limit.toString()
    })

    const response = await fetch(`${BASE_URL}/ohlcv?${params}`)
    const data = await response.json()

    if (!data.success) {
      throw new Error(data.error || 'Failed to fetch OHLCV data')
    }

    return data.data
  },

  async fetchATCSignals(symbol, timeframe, params = {}) {
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
    if (params.robustness) queryParams.append('robustness', params.robustness)
    if (params.lambda_param) queryParams.append('lambda_param', params.lambda_param.toString())
    if (params.decay) queryParams.append('decay', params.decay.toString())
    if (params.cutout) queryParams.append('cutout', params.cutout.toString())

    const response = await fetch(`${BASE_URL}/atc-signals?${queryParams}`)
    const data = await response.json()

    if (!data.success) {
      throw new Error(data.error || 'Failed to compute ATC signals')
    }

    return data.data
  },

  async fetchMovingAverages(symbol, timeframe, params = {}) {
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
    if (params.robustness) queryParams.append('robustness', params.robustness)

    const response = await fetch(`${BASE_URL}/moving-averages?${queryParams}`)
    const data = await response.json()

    if (!data.success) {
      throw new Error(data.error || 'Failed to compute moving averages')
    }

    return data.data
  },

  async fetchSymbols() {
    const response = await fetch(`${BASE_URL}/symbols`)
    const data = await response.json()

    if (!data.success) {
      throw new Error(data.error || 'Failed to fetch symbols')
    }

    return data.symbols
  },

  async fetchTimeframes() {
    const response = await fetch(`${BASE_URL}/timeframes`)
    const data = await response.json()

    if (!data.success) {
      throw new Error(data.error || 'Failed to fetch timeframes')
    }

    return data.timeframes
  }
}
