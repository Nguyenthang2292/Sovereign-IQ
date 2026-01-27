# ExchangeManager Module

Modular exchange connection management system for cryptocurrency trading with support for multiple exchanges, authentication, rate limiting, and resource management.

## Architecture

The ExchangeManager has been refactored using a **composition pattern** to improve modularity and maintainability while preserving 100% backward compatibility.

### Structure

```
modules/common/core/exchange_manager/
├── __init__.py                  # Main ExchangeManager facade with composition
├── base.py                      # ExchangeWrapper and shared utilities
├── authenticated.py             # AuthenticatedExchangeManager (credential-based)
├── public.py                    # PublicExchangeManager (no credentials)
├── connection_factory.py        # Exchange-specific connection methods
└── README.md                    # This file
```

### Components

#### 1. **ExchangeWrapper** (`base.py`)
Reference-counted wrapper for ccxt.Exchange instances:
- Thread-safe reference counting
- Automatic cleanup when not in use
- Prevents premature connection closure

#### 2. **ExchangeConnectionFactory** (`connection_factory.py`)
Exchange-specific connection methods:
- `connect_to_binance_with_credentials()`
- `connect_to_kraken_with_credentials()`
- `connect_to_kucoin_with_credentials()`
- `connect_to_gate_with_credentials()`
- `connect_to_okx_with_credentials()`
- `connect_to_bybit_with_credentials()`
- `connect_to_mexc_with_credentials()`
- `connect_to_huobi_with_credentials()`

#### 3. **AuthenticatedExchangeManager** (`authenticated.py`)
Manages authenticated exchange connections:
- API key/secret management
- Credential caching per exchange
- Reference counting and context managers
- Automatic cleanup of unused connections
- Rate limiting (throttled_call)

#### 4. **PublicExchangeManager** (`public.py`)
Manages public (unauthenticated) exchange connections:
- No credentials required
- Multi-exchange fallback support
- Exchange priority configuration
- Rate limiting
- Automatic cleanup

#### 5. **ExchangeManager** (`__init__.py`)
Main facade that composes authenticated and public managers:
- Single entry point for all exchange operations
- Delegates to appropriate sub-manager
- Maintains backward compatibility

---

## Usage

### Basic Usage

```python
from modules.common.core import ExchangeManager

# Initialize
exchange_manager = ExchangeManager()

# Access sub-managers
authenticated = exchange_manager.authenticated
public = exchange_manager.public
```

### Authenticated Exchange Operations

```python
from modules.common.core import ExchangeManager

em = ExchangeManager(api_key="your_key", api_secret="your_secret")

# Connect to Binance with credentials
exchange = em.authenticated.connect_to_binance_with_credentials()

# Make authenticated API calls
balance = em.authenticated.throttled_call(exchange.fetch_balance)

# Use context manager for automatic cleanup
with em.authenticated.exchange_context("binance") as exchange:
    positions = exchange.fetch_positions()
```

### Public Exchange Operations

```python
from modules.common.core import ExchangeManager

em = ExchangeManager()

# Connect to exchange without credentials
exchange = em.public.connect_to_exchange_with_no_credentials("binance")

# Fetch public data
ticker = em.public.throttled_call(exchange.fetch_ticker, "BTC/USDT")

# Configure fallback priority
em.public.exchange_priority_for_fallback = ["binance", "kraken", "coinbase"]
```

### Multi-Exchange Fallback

```python
from modules.common.core import ExchangeManager

em = ExchangeManager()

# Try multiple exchanges in order
for exchange_id in ["binance", "kraken", "kucoin"]:
    try:
        exchange = em.public.connect_to_exchange_with_no_credentials(exchange_id)
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1h", limit=100)
        print(f"Successfully fetched from {exchange_id}")
        break
    except Exception as e:
        print(f"{exchange_id} failed: {e}")
        continue
```

### Reference Counting & Context Managers

```python
from modules.common.core import ExchangeManager

em = ExchangeManager()

# Manual reference counting
exchange = em.authenticated.connect_to_exchange_with_credentials("binance")
# ... use exchange ...
em.authenticated.release_exchange("binance")

# Automatic with context manager (recommended)
with em.authenticated.exchange_context("binance") as exchange:
    # Exchange is automatically released when exiting context
    balance = exchange.fetch_balance()
```

### Resource Cleanup

```python
from modules.common.core import ExchangeManager

em = ExchangeManager()

# Cleanup unused connections (older than 1 hour)
em.cleanup_unused_exchanges(max_age_hours=1.0)

# Close specific exchange
em.close_exchange("binance")
```

---

## Backward Compatibility

**All existing code continues to work without modifications.** The refactoring maintains the exact same public API:

```python
# Old code (still works)
from modules.common.core.exchange_manager import ExchangeManager

# New code (also works, same result)
from modules.common.core.exchange_manager import ExchangeManager
```

All methods remain accessible through the ExchangeManager facade.

---

## Benefits of Modular Design

### 1. **Separation of Concerns**
Each component handles a specific responsibility:
- `base.py`: Core infrastructure (ExchangeWrapper, utilities)
- `connection_factory.py`: Exchange-specific connection logic
- `authenticated.py`: Credential management and authenticated operations
- `public.py`: Public API operations and fallback logic
- `__init__.py`: Facade and composition

### 2. **Testability**
Components can be tested independently:
```python
# Test only connection factory
from modules.common.core.exchange_manager.connection_factory import ExchangeConnectionFactory

factory = ExchangeConnectionFactory()
# Test exchange-specific connections
```

### 3. **Extensibility**
Easy to add new exchanges:
```python
# Add to connection_factory.py
def connect_to_new_exchange_with_credentials(self, manager, ...):
    return manager.connect_to_exchange_with_credentials("new_exchange", ...)
```

### 4. **Reduced Complexity**
- Original file: 1025 lines (monolithic)
- New structure: 5 focused modules (~150-250 lines each)
- Easier to navigate and modify

---

## Design Patterns

### Composition Pattern

The ExchangeManager uses composition to delegate to specialized managers:

```python
class ExchangeManager:
    def __init__(self, ...):
        # Specialized managers
        self.authenticated = AuthenticatedExchangeManager(...)
        self.public = PublicExchangeManager(...)

    # Delegate methods for convenience
    def cleanup_unused_exchanges(self, ...):
        self.authenticated.cleanup_unused_exchanges(...)
        self.public.cleanup_unused_exchanges(...)
```

### Factory Pattern

ExchangeConnectionFactory encapsulates exchange-specific creation logic:

```python
class ExchangeConnectionFactory:
    def connect_to_binance_with_credentials(self, manager):
        return manager.connect_to_exchange_with_credentials("binance", ...)

    def connect_to_kraken_with_credentials(self, manager, ...):
        return manager.connect_to_exchange_with_credentials("kraken", ...)
```

### Wrapper Pattern

ExchangeWrapper adds reference counting to ccxt.Exchange:

```python
class ExchangeWrapper:
    def __init__(self, exchange):
        self.exchange = exchange
        self._refcount = 0

    def increment_refcount(self):
        self._refcount += 1

    # ... other methods
```

---

## Configuration

### Environment Variables

```bash
# Default credentials (used when not specified)
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"

# Default contract type
export DEFAULT_CONTRACT_TYPE="future"  # or "spot", "margin"
```

### Exchange Priority

```python
from modules.common.core import ExchangeManager

em = ExchangeManager()

# Set fallback priority
em.public.exchange_priority_for_fallback = [
    "binance",
    "kraken",
    "kucoin",
    "gate",
    "okx",
    "bybit",
    "mexc",
    "huobi"
]
```

### Rate Limiting

```python
from modules.common.core import ExchangeManager

# Set request pause (seconds between API calls)
em = ExchangeManager()
em.authenticated = AuthenticatedExchangeManager(request_pause=0.5)
em.public = PublicExchangeManager(request_pause=0.5)
```

---

## Error Handling

### Connection Errors

```python
from modules.common.core import ExchangeManager

em = ExchangeManager()

try:
    exchange = em.public.connect_to_exchange_with_no_credentials("binance")
except Exception as e:
    print(f"Connection failed: {e}")
```

### API Errors

```python
from modules.common.core import ExchangeManager

em = ExchangeManager()

try:
    exchange = em.authenticated.connect_to_binance_with_credentials()
    balance = em.authenticated.throttled_call(exchange.fetch_balance)
except ccxt.AuthenticationError as e:
    print(f"Authentication failed: {e}")
except ccxt.NetworkError as e:
    print(f"Network error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Thread Safety

All managers are thread-safe:
- **ExchangeWrapper**: Uses threading.Lock for reference counting
- **AuthenticatedExchangeManager**: Thread-safe exchange caching
- **PublicExchangeManager**: Thread-safe exchange management

```python
import threading
from modules.common.core import ExchangeManager

em = ExchangeManager()

def fetch_data(symbol):
    exchange = em.public.connect_to_exchange_with_no_credentials("binance")
    return em.public.throttled_call(exchange.fetch_ticker, symbol)

# Safe to use in multiple threads
threads = [
    threading.Thread(target=fetch_data, args=("BTC/USDT",)),
    threading.Thread(target=fetch_data, args=("ETH/USDT",)),
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

---

## Supported Exchanges

### With Credentials (AuthenticatedExchangeManager)
- Binance (spot, futures, margin)
- Kraken
- KuCoin
- Gate.io
- OKX
- Bybit
- MEXC
- Huobi

### Without Credentials (PublicExchangeManager)
- All exchanges supported by ccxt library

---

## Migration Notes

No migration required! The module is 100% backward compatible.

If you want to use the new modular structure directly:

```python
# Access specialized components (optional)
from modules.common.core.exchange_manager import (
    ExchangeWrapper,
    AuthenticatedExchangeManager,
    PublicExchangeManager,
    ExchangeManager
)

# Use directly
authenticated = AuthenticatedExchangeManager(api_key="...", api_secret="...")
exchange = authenticated.connect_to_binance_with_credentials()
```

But this is **not recommended** - use the ExchangeManager facade instead.

---

## Testing

Run tests for the exchange_manager module:

```bash
pytest tests/common/core/test_exchange_manager.py -v
```

---

## Related Documentation

- [Common Architecture](../../ARCHITECTURE.md)
- [DataFetcher Module](../data_fetcher/README.md)
- [ExchangeManager Multi-Language Docs](../docs/ExchangeManager.md)
- [Project Overview](../../../../README.md)

---

## Performance Considerations

### Connection Pooling

Exchanges are cached and reused:
- Authenticated exchanges cached per (exchange_id, testnet, contract_type)
- Public exchanges cached per exchange_id
- Reference counting prevents premature closure

### Rate Limiting

All API calls can be rate-limited:
```python
# Throttled call (adds delay between requests)
result = em.authenticated.throttled_call(exchange.fetch_ticker, "BTC/USDT")

# Direct call (no throttling)
result = exchange.fetch_ticker("BTC/USDT")
```

### Resource Cleanup

Automatic cleanup of unused exchanges:
```python
# Cleanup connections older than 2 hours
em.cleanup_unused_exchanges(max_age_hours=2.0)

# Close specific exchange immediately
em.close_exchange("binance")
```

---

## Best Practices

### 1. Use Context Managers

```python
# ✅ GOOD: Automatic cleanup
with em.authenticated.exchange_context("binance") as exchange:
    balance = exchange.fetch_balance()

# ❌ BAD: Manual cleanup required
exchange = em.authenticated.connect_to_exchange_with_credentials("binance")
balance = exchange.fetch_balance()
em.authenticated.release_exchange("binance")  # Easy to forget!
```

### 2. Use Rate Limiting

```python
# ✅ GOOD: Rate limited
ticker = em.public.throttled_call(exchange.fetch_ticker, "BTC/USDT")

# ❌ RISKY: No rate limiting (may hit exchange limits)
ticker = exchange.fetch_ticker("BTC/USDT")
```

### 3. Handle Errors

```python
# ✅ GOOD: Proper error handling
try:
    exchange = em.public.connect_to_exchange_with_no_credentials("binance")
    data = em.public.throttled_call(exchange.fetch_ohlcv, "BTC/USDT")
except ccxt.NetworkError as e:
    # Try fallback exchange
    exchange = em.public.connect_to_exchange_with_no_credentials("kraken")
    data = em.public.throttled_call(exchange.fetch_ohlcv, "BTC/USDT")
```

### 4. Cleanup Resources

```python
# ✅ GOOD: Regular cleanup
em.cleanup_unused_exchanges(max_age_hours=1.0)

# ⚠️ OK: Manual cleanup when needed
em.close_exchange("binance")
```

---

## Troubleshooting

### Connection Fails

**Problem**: Cannot connect to exchange

**Solutions**:
1. Check API credentials are correct
2. Verify network connection
3. Check exchange status (may be under maintenance)
4. Try fallback exchange

### Authentication Error

**Problem**: API returns authentication error

**Solutions**:
1. Verify API key and secret
2. Check API key permissions
3. Ensure API key is not expired
4. Check IP whitelist if enabled

### Rate Limit Error

**Problem**: Exchange returns rate limit error

**Solutions**:
1. Use `throttled_call()` instead of direct calls
2. Increase `request_pause` parameter
3. Use multiple API keys (round-robin)

---

## Contributing

When adding support for a new exchange:

1. Add connection method to `connection_factory.py`
2. Update `AuthenticatedExchangeManager` to delegate to new method
3. Test connection with valid credentials
4. Update this README with new exchange

---

## Changelog

### Version 2.0 (Modular Refactoring)
- Refactored monolithic file (1025 lines) into 5 modules
- Introduced composition pattern
- Created ExchangeConnectionFactory for exchange-specific logic
- Improved testability and maintainability
- **100% backward compatible** - no breaking changes

### Version 1.0 (Original)
- Monolithic exchange_manager.py
- AuthenticatedExchangeManager and PublicExchangeManager classes
- Support for 8 major exchanges
