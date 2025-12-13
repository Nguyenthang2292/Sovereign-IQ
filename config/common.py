"""
Common/Shared Configuration.

Configuration constants used across multiple components.
"""

# Exchange Settings
DEFAULT_EXCHANGES = [
    "binance",
    "kraken",
    "kucoin",
    "gate",
    "okx",
    "bybit",
    "mexc",
    "huobi",
]
DEFAULT_EXCHANGE_STRING = ",".join(DEFAULT_EXCHANGES)
DEFAULT_REQUEST_PAUSE = 0.2  # Pause between API requests (seconds)
DEFAULT_CONTRACT_TYPE = "future"  # Contract type: 'spot', 'margin', or 'future'

# Data Fetching Settings
DEFAULT_SYMBOL = "BTC/USDT"  # Default trading pair
DEFAULT_QUOTE = "USDT"  # Default quote currency
DEFAULT_TIMEFRAME = "15m"  # Default timeframe
DEFAULT_LIMIT = 1500  # Default number of candles to fetch

