# DataFetcher Module

Modular data fetching system for cryptocurrency market data with exchange fallback and intelligent caching.

## Architecture

The DataFetcher module has been refactored using a **composition pattern** to improve modularity and maintainability while preserving 100% backward compatibility.

### Structure

```
modules/common/core/data_fetcher/
├── __init__.py              # Main DataFetcher class with composition
├── base.py                  # Base infrastructure (cache, shutdown handling)
├── exceptions.py            # SymbolFetchError exception
├── binance_prices.py        # Binance current prices fetching
├── binance_futures.py       # Binance Futures positions & balance
├── symbol_discovery.py      # Symbol discovery (spot & futures)
└── ohlcv.py                 # OHLCV fetching with exchange fallback
```

### Components

#### 1. **DataFetcherBase** (`base.py`)
Core infrastructure providing:
- Exchange manager integration
- OHLCV dataframe caching
- Market prices storage
- Shutdown signal handling

#### 2. **BinancePriceFetcher** (`binance_prices.py`)
Handles real-time price fetching:
- `fetch_current_prices_from_binance()`: Fetch current ticker prices for multiple symbols

#### 3. **BinanceFuturesFetcher** (`binance_futures.py`)
Manages Binance Futures operations:
- `fetch_binance_futures_positions()`: Get open futures positions
- `fetch_binance_account_balance()`: Retrieve account balance
- Helper methods for position parsing and validation

#### 4. **SymbolDiscovery** (`symbol_discovery.py`)
Symbol discovery for different market types:
- `list_binance_futures_symbols()`: Discover futures symbols sorted by volume
- `get_spot_symbols()`: Fetch spot symbols with retry logic and error handling

#### 5. **OHLCVFetcher** (`ohlcv.py`)
OHLCV data fetching with intelligent fallback:
- `fetch_ohlcv_with_fallback_exchange()`: Multi-exchange OHLCV fetching with caching
- `dataframe_to_close_series()`: Convert OHLCV DataFrame to close price Series

## Usage

### Basic Usage

```python
from modules.common.core import DataFetcher, ExchangeManager

# Initialize
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Fetch OHLCV data
df, exchange = data_fetcher.fetch_ohlcv_with_fallback_exchange(
    symbol="BTC/USDT",
    timeframe="1h",
    limit=1000
)

# Fetch spot symbols
symbols = data_fetcher.get_spot_symbols(
    exchange_name="binance",
    quote_currency="USDT"
)

# Fetch futures positions
positions = data_fetcher.fetch_binance_futures_positions(
    api_key="your_key",
    api_secret="your_secret"
)
```

### Advanced Usage

```python
# OHLCV with freshness check
df, exchange = data_fetcher.fetch_ohlcv_with_fallback_exchange(
    symbol="ETH/USDT",
    timeframe="15m",
    limit=500,
    check_freshness=True,  # Ensure data is recent
    exchanges=["binance", "kraken", "coinbase"]
)

# Futures symbols with filtering
symbols = data_fetcher.list_binance_futures_symbols(
    exclude_symbols={"BTC/USDT", "ETH/USDT"},
    max_candidates=50
)

# Account balance
balance = data_fetcher.fetch_binance_account_balance(
    currency="USDT"
)
```

## Backward Compatibility

**All existing code continues to work without modifications.** The refactoring maintains the exact same public API:

```python
# Old code (still works)
from modules.common.core.data_fetcher import DataFetcher, SymbolFetchError

# New code (also works, same result)
from modules.common.core.data_fetcher import DataFetcher, SymbolFetchError
```

All methods remain accessible at the DataFetcher level through delegation.

## Benefits of Modular Design

### 1. **Separation of Concerns**
Each component handles a specific responsibility:
- Prices, futures, symbols, and OHLCV are now separate modules
- Easier to understand and maintain

### 2. **Testability**
Components can be tested independently:
```python
# Test only symbol discovery
from modules.common.core.data_fetcher.symbol_discovery import SymbolDiscovery
from modules.common.core.data_fetcher.base import DataFetcherBase

base = DataFetcherBase(exchange_manager)
symbol_discovery = SymbolDiscovery(base)
symbols = symbol_discovery.get_spot_symbols()
```

### 3. **Extensibility**
Easy to add new functionality:
- Create new fetcher component (e.g., `kraken_futures.py`)
- Add to DataFetcher composition
- No changes to existing code

### 4. **Reduced Complexity**
- Original file: 746 lines (monolithic)
- New structure: 6 focused modules (~150 lines each)
- Easier to navigate and modify

## Migration Notes

No migration required! The module is 100% backward compatible.

If you want to use the new modular structure directly:

```python
# Access specialized components (optional)
data_fetcher._binance_prices.fetch_current_prices_from_binance(symbols)
data_fetcher._ohlcv.fetch_ohlcv_with_fallback_exchange(symbol)
```

But this is **not recommended** - use the public API instead.

## Error Handling

### SymbolFetchError

Raised when symbol fetching fails:

```python
from modules.common.core.data_fetcher import SymbolFetchError

try:
    symbols = data_fetcher.get_spot_symbols()
except SymbolFetchError as e:
    if e.is_retryable:
        print(f"Retryable error: {e}")
        # Can retry the operation
    else:
        print(f"Non-retryable error: {e}")
        # Don't retry, fix the issue
```

## Caching

The DataFetcher uses intelligent caching:

- **OHLCV cache**: In-memory cache for OHLCV dataframes
- **Cache key**: `(symbol, timeframe, limit)`
- **Invalidation**: Automatic for freshness checks
- **Benefits**: Reduces exchange API calls, improves performance

## Testing

Run tests for the data_fetcher module:

```bash
pytest tests/common/core/test_data_fetcher.py -v
```

## Related Documentation

- [ExchangeManager Documentation](../docs/ExchangeManager-en.md)
- [Common Utils](../../utils/README.md)
- [Testing Guide](../../../../tests/docs/test_memory_usage_guide.md)
