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

# Default Symbols and Timeframes for Training and Testing
DEFAULT_CRYPTO_SYMBOLS_FOR_TRAINING_DL = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "MATICUSDT",
    "AVAXUSDT",
]  # Default list of crypto symbols for training

DEFAULT_TIMEFRAMES_FOR_TRAINING_DL = ["15m", "30m", "1h"]  # Default list of timeframes for training
