# 📚 ExchangeManager Documentation

> **Language / Ngôn ngữ**: [English](ExchangeManager-en.md) | [Tiếng Việt](ExchangeManager-vi.md)

## Table of Contents

1. [Overview](#overview)
2. [AuthenticatedExchangeManager](#authenticatedexchangemanager)
3. [PublicExchangeManager](#publicexchangemanager)
4. [ExchangeManager (Composite)](#exchangemanager-composite)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)

---

## Overview

`ExchangeManager` is a system for managing connections to cryptocurrency exchanges through the `ccxt` library. The system is designed with 3 layers:

1. **AuthenticatedExchangeManager**: Manages connections that require credentials (API key/secret)
2. **PublicExchangeManager**: Manages connections that don't require credentials (public data)
3. **ExchangeManager**: Composite manager that combines both managers above

### When to Use What?

| Data Type | Requires Credentials? | Use Which Manager? |
|-----------|----------------------|---------------------|
| Current price (ticker) | ✅ Yes | `authenticated.connect_to_binance_with_credentials()` |
| Symbol list (markets) | ✅ Yes | `authenticated.connect_to_binance_with_credentials()` |
| Positions from account | ✅ Yes | `authenticated.connect_to_binance_with_credentials()` |
| OHLCV data (historical) | ❌ No | `public.connect_to_exchange_with_no_credentials()` |
| Other public data | ❌ No | `public.connect_to_exchange_with_no_credentials()` |

---

## AuthenticatedExchangeManager

### Purpose

Manages exchange connections that **require authentication** via API key and secret. Used for operations related to your account.

### Initialization

```python
from modules.ExchangeManager import AuthenticatedExchangeManager

# Method 1: Pass credentials directly
auth_manager = AuthenticatedExchangeManager(
    api_key="your_api_key",
    api_secret="your_api_secret",
    testnet=False  # True if using testnet
)

# Method 2: Get from environment variables or config file
auth_manager = AuthenticatedExchangeManager()  # Automatically gets from env/config
```

**Credential priority order:**

1. Parameters passed during initialization
2. Environment variables: `BINANCE_API_KEY`, `BINANCE_API_SECRET`
3. Config file: `modules/config_api.py`

### Methods

#### `connect_to_exchange_with_credentials(exchange_id, ...) -> ccxt.Exchange`

**Purpose**: Connect to any authenticated exchange - **REQUIRES credentials**.

**Supported exchanges**: binance, okx, kucoin, bybit, gate, mexc, huobi, kraken, and all exchanges supported by ccxt.

**When to use:**

- ✅ Get current price (`fetch_ticker`)
- ✅ List symbols (`load_markets`)
- ✅ Get positions from account (`fetch_positions`)
- ✅ Any API call that requires authentication

**Parameters:**

- `exchange_id` (str): Exchange name (e.g., 'binance', 'okx', 'kucoin', 'bybit')
- `api_key` (Optional[str]): API key for this exchange (optional)
- `api_secret` (Optional[str]): API secret for this exchange (optional)
- `testnet` (Optional[bool]): Use testnet if True (optional)
- `contract_type` (Optional[str]): Contract type ('spot', 'margin', 'future') (optional)

**Examples:**

```python
# Connect to OKX
okx = auth_manager.connect_to_exchange_with_credentials('okx', 
    api_key='okx_key', 
    api_secret='okx_secret'
)

# Connect to KuCoin with testnet
kucoin = auth_manager.connect_to_exchange_with_credentials('kucoin',
    api_key='kucoin_key',
    api_secret='kucoin_secret',
    testnet=True
)

# Connect to Bybit with spot trading
bybit = auth_manager.connect_to_exchange_with_credentials('bybit',
    api_key='bybit_key',
    api_secret='bybit_secret',
    contract_type='spot'
)
```

**Notes:**

- ⚠️ **Required** to have API key and secret (can be set via `set_exchange_credentials()` or passed directly)
- ⚠️ If credentials are missing, will raise `ValueError`
- ✅ Instance is cached, only created once (lazy initialization)
- ✅ Automatically enables rate limiting
- ✅ Supports testnet for Binance, OKX, KuCoin, Bybit, Gate

---

#### `set_exchange_credentials(exchange_id, api_key, api_secret)`

**Purpose**: Set credentials for a specific exchange to use later.

**When to use:**

- ✅ When you want to set credentials once and use multiple times
- ✅ When managing credentials for multiple exchanges

**Example:**

```python
# Set credentials for OKX
auth_manager.set_exchange_credentials('okx', 'okx_key', 'okx_secret')

# Set credentials for KuCoin
auth_manager.set_exchange_credentials('kucoin', 'kucoin_key', 'kucoin_secret')

# Then can use without passing credentials
okx = auth_manager.connect_to_exchange_with_credentials('okx')
kucoin = auth_manager.connect_to_exchange_with_credentials('kucoin')
```

**Notes:**

- ✅ Credentials are stored per-exchange
- ✅ When setting new credentials, the cache for that exchange is cleared to force reconnection

---

#### `connect_to_binance_with_credentials() -> ccxt.Exchange`

**Purpose**: Connect to authenticated Binance exchange - **REQUIRES credentials**.

**DEPRECATED**: Should use `connect_to_exchange_with_credentials('binance')` instead. Kept for backward compatibility.

**When to use:**

- ✅ Get current price (`fetch_ticker`)
- ✅ List symbols (`load_markets`)
- ✅ Get positions from account (`fetch_positions`)
- ✅ Any API call that requires authentication

**Example:**

```python
# Connect to authenticated Binance exchange (requires credentials)
exchange = auth_manager.connect_to_binance_with_credentials()

# Get current price of BTC/USDT
ticker = exchange.fetch_ticker("BTC/USDT")
print(f"Current price: {ticker['last']}")

# List all markets
markets = exchange.load_markets()
print(f"Total markets: {len(markets)}")

# Get positions from account
positions = exchange.fetch_positions()
for pos in positions:
    print(f"Symbol: {pos['symbol']}, Size: {pos['size']}")
```

**Notes:**

- ⚠️ **Required** to have API key and secret
- ⚠️ If credentials are missing, will raise `ValueError`
- ✅ Instance is cached, only created once (lazy initialization)
- ✅ Automatically enables rate limiting

**Possible errors:**

```python
# If credentials are missing
try:
    exchange = auth_manager.connect_to_binance_with_credentials()
except ValueError as e:
    print(e)  # "API Key and API Secret are required..."
```

---

#### Convenience Methods for Exchanges

Convenience methods to connect to popular exchanges:

- `connect_to_kraken_with_credentials(api_key, api_secret, testnet, contract_type)`
- `connect_to_kucoin_with_credentials(api_key, api_secret, testnet, contract_type)`
- `connect_to_gate_with_credentials(api_key, api_secret, testnet, contract_type)`
- `connect_to_okx_with_credentials(api_key, api_secret, testnet, contract_type)`
- `connect_to_bybit_with_credentials(api_key, api_secret, testnet, contract_type)`
- `connect_to_mexc_with_credentials(api_key, api_secret, testnet, contract_type)`
- `connect_to_huobi_with_credentials(api_key, api_secret, testnet, contract_type)`

All these methods are wrappers of `connect_to_exchange_with_credentials()` with the corresponding exchange_id.

**Example:**

```python
# Method 1: Set credentials first
auth_manager.set_exchange_credentials('okx', 'okx_key', 'okx_secret')
okx = auth_manager.connect_to_okx_with_credentials()

# Method 2: Pass credentials directly
kucoin = auth_manager.connect_to_kucoin_with_credentials(
    api_key='kucoin_key',
    api_secret='kucoin_secret'
)

# Method 3: With testnet and contract type
bybit = auth_manager.connect_to_bybit_with_credentials(
    api_key='bybit_key',
    api_secret='bybit_secret',
    testnet=True,
    contract_type='spot'
)
```

---

#### `throttled_call(func, *args, **kwargs)`

**Purpose**: Call a function with automatic rate limiting to avoid exceeding API limits.

**When to use:**

- ✅ Any API call that needs to ensure rate limits are not exceeded
- ✅ When making multiple consecutive API calls
- ✅ To avoid IP ban due to too many requests

**How it works:**

- Automatically calculates wait time between requests
- Ensures each request is at least `request_pause` seconds apart (default 0.2s)
- Thread-safe (can be used in multi-threading)

**Example:**

```python
exchange = auth_manager.connect_to_binance_with_credentials()

# Call API with rate limiting
ticker = auth_manager.throttled_call(
    exchange.fetch_ticker,
    "BTC/USDT"
)

# Make multiple consecutive API calls
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
for symbol in symbols:
    ticker = auth_manager.throttled_call(
        exchange.fetch_ticker,
        symbol
    )
    print(f"{symbol}: {ticker['last']}")
```

**Parameters:**

- `func`: Function to call (usually an exchange method)
- `*args`: Positional arguments for the function
- `**kwargs`: Keyword arguments for the function

**Notes:**

- ✅ Automatically sleeps if needed to ensure rate limit
- ✅ Thread-safe (uses lock)
- ✅ Can adjust `request_pause` via environment variable `BINANCE_REQUEST_SLEEP`

---

## PublicExchangeManager

### Purpose

Manages exchange connections that **don't require authentication** (public data). Used for operations that fetch public data.

### Initialization

```python
from modules.ExchangeManager import PublicExchangeManager

# Initialize (no credentials needed)
public_manager = PublicExchangeManager()
```

### Methods

#### `connect_to_exchange_with_no_credentials(exchange_id: str) -> ccxt.Exchange`

**Purpose**: Connect to a public exchange (NO credentials required).

**When to use:**

- ✅ Fetch OHLCV data (historical prices)
- ✅ Fetch other public data
- ✅ When you need to fallback to another exchange if Binance doesn't have data

**Example:**

```python
# Connect to Binance public (no credentials needed)
binance = public_manager.connect_to_exchange_with_no_credentials("binance")
ohlcv = binance.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=100)

# Connect to Kraken public
kraken = public_manager.connect_to_exchange_with_no_credentials("kraken")
ohlcv = kraken.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=100)

# Connect to other exchanges
kucoin = public_manager.connect_to_exchange_with_no_credentials("kucoin")
gate = public_manager.connect_to_exchange_with_no_credentials("gate")
okx = public_manager.connect_to_exchange_with_no_credentials("okx")
```

**Parameters:**

- `exchange_id` (str): Exchange name (e.g., "binance", "kraken", "kucoin", "gate", "okx", "bybit", "mexc", "huobi")

**Notes:**

- ✅ **No** API key/secret required
- ✅ Instance is cached, only created once per exchange
- ✅ Automatically enables rate limiting
- ✅ Automatically sets `defaultType: 'future'` for futures trading

**Possible errors:**

```python
# If exchange is not supported
try:
    exchange = public_manager.connect_to_exchange_with_no_credentials("unknown_exchange")
except ValueError as e:
    print(e)  # "Exchange 'unknown_exchange' is not supported by ccxt."
```

**Supported exchanges:**

- `binance` - Binance
- `kraken` - Kraken
- `kucoin` - KuCoin
- `gate` - Gate.io
- `okx` - OKX
- `bybit` - Bybit
- `mexc` - MEXC
- `huobi` - Huobi
- And all exchanges supported by ccxt

---

#### `throttled_call(func, *args, **kwargs)`

**Purpose**: Similar to `AuthenticatedExchangeManager.throttled_call()`, but for public calls.

**Example:**

```python
exchange = public_manager.connect_to_exchange_with_no_credentials("kraken")

# Call API with rate limiting
ohlcv = public_manager.throttled_call(
    exchange.fetch_ohlcv,
    "BTC/USDT",
    timeframe="1h",
    limit=100
)
```

---

#### `exchange_priority_for_fallback` (property)

**Purpose**: List of exchanges in priority order when fallback is needed.

**Example:**

```python
# View current priority list
print(public_manager.exchange_priority_for_fallback)
# Output: ['binance', 'kraken', 'kucoin', 'gate', 'okx', 'bybit', 'mexc', 'huobi']

# Change priority order
public_manager.exchange_priority_for_fallback = ['kraken', 'binance', 'kucoin']

# Or get from environment variable
# Set OHLCV_FALLBACKS="kraken,binance,kucoin"
```

**Usage in fallback:**

```python
# Try to fetch OHLCV from exchanges in priority order
for exchange_id in public_manager.exchange_priority_for_fallback:
    try:
        exchange = public_manager.connect_to_exchange_with_no_credentials(exchange_id)
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=100)
        if ohlcv:
            print(f"Successfully fetched from {exchange_id}")
            break
    except Exception as e:
        print(f"Failed to fetch from {exchange_id}: {e}")
        continue
```

**Notes:**

- ✅ Can be set via environment variable `OHLCV_FALLBACKS`
- ✅ Default: `"binance,kraken,kucoin,gate,okx,bybit,mexc,huobi"`
- ✅ Equivalent to `em.exchange_priority_for_fallback` (in ExchangeManager)

---

## ExchangeManager (Composite)

### Purpose

Composite manager that combines both `AuthenticatedExchangeManager` and `PublicExchangeManager`, providing a unified interface and maintaining backward compatibility.

### Initialization

```python
from modules.ExchangeManager import ExchangeManager

# Initialize with credentials
em = ExchangeManager(
    api_key="your_api_key",
    api_secret="your_api_secret",
    testnet=False
)

# Or without credentials (public only)
em = ExchangeManager()
```

### Structure

```python
em = ExchangeManager(api_key, api_secret)

# Access authenticated manager
em.authenticated  # AuthenticatedExchangeManager instance

# Access public manager
em.public  # PublicExchangeManager instance
```

### Methods

#### `normalize_symbol(market_symbol: str) -> str`

**Purpose**: Normalize symbol from Binance futures format.

**Example:**

```python
# Normalize symbol
symbol1 = em.normalize_symbol("BTC/USDT:USDT")  # → "BTC/USDT"
symbol2 = em.normalize_symbol("ETHUSDT")        # → "ETH/USDT"
symbol3 = em.normalize_symbol("BNB/USDT")       # → "BNB/USDT"
```

**When to use:**

- ✅ When receiving symbol from Binance markets (format `BTC/USDT:USDT`)
- ✅ Need to normalize to `BASE/QUOTE` format

---

#### `exchange_priority_for_fallback` (property)

**Purpose**: List of exchange priorities for OHLCV fallback.

**Example:**

```python
# View list
print(em.exchange_priority_for_fallback)

# Change
em.exchange_priority_for_fallback = ['kraken', 'binance', 'kucoin']
```

**Notes:**

- ✅ Equivalent to `em.public.exchange_priority_for_fallback`
- ✅ Can be set/get as property
- ✅ Used for OHLCV fallback mechanism

---

## Usage Examples

### Example 1: Get current price from Binance (requires credentials)

```python
from modules.ExchangeManager import ExchangeManager

# Initialize
em = ExchangeManager(api_key="...", api_secret="...")

# Connect to authenticated Binance (requires credentials)
exchange = em.authenticated.connect_to_binance_with_credentials()

# Get price with rate limiting
ticker = em.authenticated.throttled_call(
    exchange.fetch_ticker,
    "BTC/USDT"
)

print(f"BTC/USDT price: {ticker['last']}")
```

### Example 1b: Get prices from multiple exchanges

```python
from modules.ExchangeManager import ExchangeManager

# Initialize
em = ExchangeManager(api_key="binance_key", api_secret="binance_secret")

# Set credentials for other exchanges
em.authenticated.set_exchange_credentials('okx', 'okx_key', 'okx_secret')
em.authenticated.set_exchange_credentials('kucoin', 'kucoin_key', 'kucoin_secret')

# Get price from Binance
binance = em.authenticated.connect_to_binance_with_credentials()
binance_ticker = em.authenticated.throttled_call(
    binance.fetch_ticker, "BTC/USDT"
)

# Get price from OKX
okx = em.authenticated.connect_to_okx_with_credentials()
okx_ticker = em.authenticated.throttled_call(
    okx.fetch_ticker, "BTC/USDT"
)

# Get price from KuCoin
kucoin = em.authenticated.connect_to_kucoin_with_credentials()
kucoin_ticker = em.authenticated.throttled_call(
    kucoin.fetch_ticker, "BTC/USDT"
)

print(f"Binance: {binance_ticker['last']}")
print(f"OKX: {okx_ticker['last']}")
print(f"KuCoin: {kucoin_ticker['last']}")
```

### Example 2: Fetch OHLCV data (no credentials needed)

```python
from modules.ExchangeManager import ExchangeManager

# Initialize (no credentials needed)
em = ExchangeManager()

# Try to fetch from exchanges in priority order
for exchange_id in em.public.exchange_priority_for_fallback:
    try:
        exchange = em.public.connect_to_exchange_with_no_credentials(exchange_id)
        ohlcv = em.public.throttled_call(
            exchange.fetch_ohlcv,
            "BTC/USDT",
            timeframe="1h",
            limit=100
        )
        if ohlcv:
            print(f"✓ Fetched {len(ohlcv)} candles from {exchange_id}")
            break
    except Exception as e:
        print(f"✗ {exchange_id}: {e}")
        continue
```

### Example 3: List symbols from Binance (requires credentials)

```python
from modules.ExchangeManager import ExchangeManager

em = ExchangeManager(api_key="...", api_secret="...")

# Connect to authenticated Binance (requires credentials)
exchange = em.authenticated.connect_to_binance_with_credentials()

# Load markets
markets = exchange.load_markets()

# Filter futures USDT pairs
futures_usdt = [
    symbol for symbol, market in markets.items()
    if market.get('contract') and market.get('quote') == 'USDT'
]

print(f"Total futures USDT pairs: {len(futures_usdt)}")
```

### Example 3b: List symbols from multiple exchanges

```python
from modules.ExchangeManager import ExchangeManager

em = ExchangeManager(api_key="binance_key", api_secret="binance_secret")

# Set credentials for OKX
em.authenticated.set_exchange_credentials('okx', 'okx_key', 'okx_secret')

# Get markets from Binance
binance = em.authenticated.connect_to_binance_with_credentials()
binance_markets = binance.load_markets()
print(f"Binance markets: {len(binance_markets)}")

# Get markets from OKX
okx = em.authenticated.connect_to_okx_with_credentials()
okx_markets = okx.load_markets()
print(f"OKX markets: {len(okx_markets)}")
```

### Example 4: Usage in DataFetcher

```python
from modules.ExchangeManager import ExchangeManager
from modules.DataFetcher import DataFetcher

# Initialize
em = ExchangeManager(api_key="...", api_secret="...")
data_fetcher = DataFetcher(em)

# Fetch prices (uses authenticated)
data_fetcher.fetch_current_prices_from_binance(["BTC/USDT", "ETH/USDT"])

# Fetch OHLCV (uses public)
df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=100, timeframe="1h")
```

### Example 5: Multi-exchange portfolio management

```python
from modules.ExchangeManager import ExchangeManager

# Initialize
em = ExchangeManager(api_key="binance_key", api_secret="binance_secret")

# Set credentials for other exchanges
em.authenticated.set_exchange_credentials('okx', 'okx_key', 'okx_secret')
em.authenticated.set_exchange_credentials('bybit', 'bybit_key', 'bybit_secret')

# Get positions from multiple exchanges
binance = em.authenticated.connect_to_binance_with_credentials()
okx = em.authenticated.connect_to_okx_with_credentials()
bybit = em.authenticated.connect_to_bybit_with_credentials()

binance_positions = binance.fetch_positions()
okx_positions = okx.fetch_positions()
bybit_positions = bybit.fetch_positions()

print(f"Binance positions: {len(binance_positions)}")
print(f"OKX positions: {len(okx_positions)}")
print(f"Bybit positions: {len(bybit_positions)}")
```

---

## Best Practices

### 1. Clearly distinguish authenticated vs public

```python
# ✅ CORRECT: Use authenticated for authenticated calls
exchange = em.authenticated.connect_to_binance_with_credentials()
ticker = exchange.fetch_ticker("BTC/USDT")  # Requires credentials

# ❌ WRONG: Use public for authenticated calls
exchange = em.public.connect_to_exchange_with_no_credentials("binance")
ticker = exchange.fetch_ticker("BTC/USDT")  # May fail or be inaccurate
```

### 2. Always use throttled_call for API calls

```python
# ✅ CORRECT: Use throttled_call
ticker = em.authenticated.throttled_call(exchange.fetch_ticker, "BTC/USDT")

# ❌ WRONG: Call directly (may exceed rate limit)
ticker = exchange.fetch_ticker("BTC/USDT")
```

### 3. Use fallback for OHLCV

```python
# ✅ CORRECT: Try multiple exchanges if one fails
for exchange_id in em.public.exchange_priority_for_fallback:
    try:
        exchange = em.public.connect_to_exchange_with_no_credentials(exchange_id)
        ohlcv = em.public.throttled_call(
            exchange.fetch_ohlcv, "BTC/USDT", timeframe="1h", limit=100
        )
        if ohlcv:
            break
    except Exception:
        continue
```

### 4. Cache credentials securely

```python
# ✅ CORRECT: Get from environment variables
em = ExchangeManager()  # Automatically gets from env

# ❌ WRONG: Hardcode credentials in code
em = ExchangeManager(api_key="hardcoded_key", api_secret="hardcoded_secret")
```

### 5. Handle errors correctly

```python
# ✅ CORRECT: Handle credential errors
try:
    exchange = em.authenticated.connect_to_binance_with_credentials()
except ValueError as e:
    print(f"Credential error: {e}")
    # Fallback or exit

# ✅ CORRECT: Handle unsupported exchange errors
try:
    exchange = em.public.connect_to_exchange_with_no_credentials("unknown")
except ValueError as e:
    print(f"Exchange not supported: {e}")
    # Try another exchange
```

---

## Summary

| Manager | When to Use | Requires Credentials? | Main Methods |
|---------|------------|----------------------|--------------|
| `AuthenticatedExchangeManager` | Get prices, markets, positions | ✅ Yes | `connect_to_exchange_with_credentials()`, `connect_to_*_with_credentials()`, `set_exchange_credentials()`, `throttled_call()` |
| `PublicExchangeManager` | Get OHLCV, public data | ❌ No | `connect_to_exchange_with_no_credentials()`, `throttled_call()` |
| `ExchangeManager` | Composite, backward compatibility | Depends | All methods above + `normalize_symbol()` |

### Supported Exchanges (Authenticated)

Exchanges supported with convenience methods:

- ✅ Binance (`connect_to_binance_with_credentials()`)
- ✅ Kraken (`connect_to_kraken_with_credentials()`)
- ✅ KuCoin (`connect_to_kucoin_with_credentials()`)
- ✅ Gate.io (`connect_to_gate_with_credentials()`)
- ✅ OKX (`connect_to_okx_with_credentials()`)
- ✅ Bybit (`connect_to_bybit_with_credentials()`)
- ✅ MEXC (`connect_to_mexc_with_credentials()`)
- ✅ Huobi (`connect_to_huobi_with_credentials()`)

Or use `connect_to_exchange_with_credentials(exchange_id)` for any exchange supported by ccxt.

---

## Links

- [ccxt Documentation](https://docs.ccxt.com/)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
