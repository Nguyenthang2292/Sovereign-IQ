# modules/common Architecture

This document describes the architecture and design principles of the `modules/common` package, which provides shared utilities and infrastructure for the entire crypto-probability system.

## Table of Contents

- [Overview](#overview)
- [Layer Architecture](#layer-architecture)
- [Directory Structure](#directory-structure)
- [Dependency Rules](#dependency-rules)
- [Import Guidelines](#import-guidelines)
- [Design Patterns](#design-patterns)
- [Adding New Code](#adding-new-code)

---

## Overview

The `modules/common` package follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
│     (Trading Modules, ML Models, Web Apps, etc.)        │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼──────────┐  ┌──▼─────────────────────┐
│  modules.common  │  │  modules.common.core   │
│     .data        │  │    .data_fetcher       │
│  (orchestration) │  │  (exchange integration)│
└────────┬─────────┘  └───────────┬────────────┘
         │                        │
         └────────┬───────────────┘
                  │
         ┌────────▼────────┐
         │ modules.common  │
         │    .domain      │
         │ (domain logic)  │
         └─────────────────┘
```

**Key Principle**: Dependencies flow downward. Lower layers have no knowledge of upper layers.

---

## Layer Architecture

### Layer 1: Domain (`modules/common/domain/`)

**Purpose**: Pure trading domain logic and concepts

**Responsibilities**:
- Symbol normalization and validation
- Timeframe conversion and calculation
- Trading domain types and constants

**Dependencies**: None (pure functions, no external dependencies)

**Files**:
- `symbols.py`: Symbol normalization utilities
- `timeframes.py`: Timeframe conversion utilities

**Examples**:
```python
from modules.common.domain import normalize_symbol, timeframe_to_minutes

symbol = normalize_symbol("btc")  # Returns "BTC/USDT"
minutes = timeframe_to_minutes("1h")  # Returns 60
```

**Why Domain Layer**:
- Pure functions with no side effects
- Can be used anywhere without dependencies
- Easy to test in isolation
- No coupling to exchanges, databases, or UI

---

### Layer 2: Infrastructure (`modules/common/core/`)

**Purpose**: Low-level infrastructure for external integrations

**Responsibilities**:
- Exchange API communication
- Data fetching with retry logic and fallback
- Hardware and GPU management
- Exchange connection management

**Dependencies**: `domain/` for normalization, external libraries (ccxt, pandas, etc.)

**Key Components**:

#### `core/data_fetcher/` - Exchange Integration
Modular data fetching system using composition pattern:
- `base.py`: Core infrastructure (caching, shutdown handling)
- `ohlcv.py`: OHLCV fetching with multi-exchange fallback
- `binance_prices.py`: Binance current prices
- `binance_futures.py`: Futures positions and balance
- `symbol_discovery.py`: Symbol discovery from exchanges
- `exceptions.py`: Custom exceptions

#### `core/exchange_manager/` - Exchange Connection Management
Modular exchange connection management system using composition pattern:
- `base.py`: ExchangeWrapper with reference counting
- `connection_factory.py`: Exchange-specific connection methods
- `authenticated.py`: AuthenticatedExchangeManager (credential-based)
- `public.py`: PublicExchangeManager (no credentials)
- `__init__.py`: ExchangeManager facade

#### `core/indicator_engine.py`
Technical indicator calculation orchestration

**Examples**:
```python
from modules.common.core import DataFetcher, ExchangeManager

exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Fetch OHLCV with exchange fallback
df, exchange = data_fetcher.fetch_ohlcv_with_fallback_exchange(
    symbol="BTC/USDT",
    timeframe="1h",
    limit=1000
)
```

---

### Layer 3: Data (`modules/common/data/`)

**Purpose**: High-level data operations and orchestration

**Responsibilities**:
- OHLCV data validation
- DataFrame transformations
- Multi-symbol/timeframe orchestration
- Data quality checks

**Dependencies**: `domain/` for normalization, `core.data_fetcher/` for fetching

**Files**:
- `validation.py`: OHLCV DataFrame validation
- `transformation.py`: DataFrame/Series transformations
- `fetchers.py`: High-level OHLCV fetching orchestration

**Examples**:
```python
from modules.common.data import fetch_ohlcv_data_dict, validate_ohlcv_input

# Fetch multiple symbols/timeframes
data = fetch_ohlcv_data_dict(
    symbols=["BTC/USDT", "ETH/USDT"],
    timeframes=["1h", "4h"],
    limit=1000
)
# Returns: {symbol: {timeframe: DataFrame}}

# Validate OHLCV data
is_valid = validate_ohlcv_input(df)
```

**Why Separate from Infrastructure**:
- Orchestrates multiple low-level operations
- Adds validation and error handling
- Provides convenient high-level APIs
- Can switch infrastructure implementations without changing data layer

---

### Layer 4: UI (`modules/common/ui/`)

**Purpose**: User interface utilities (CLI, formatting, logging)

**Responsibilities**:
- Colored console output
- Progress bars
- Logging utilities
- User input handling

**Dependencies**: None (pure UI concerns)

**Files**:
- `logging.py`: Logging utilities with colors
- `formatting.py`: Text formatting and coloring
- `progress_bar.py`: Progress bar display

**Examples**:
```python
from modules.common.ui.logging import log_info, log_success, log_error

log_info("Starting analysis...")
log_success("Analysis completed!")
log_error("Failed to fetch data")
```

---

### Layer 5: Utilities Facade (`modules/common/utils/`)

**Purpose**: Backward compatibility and convenience

**Responsibilities**:
- Re-export commonly used functions from all layers
- Provide single import point for convenience
- Maintain backward compatibility with existing code

**Dependencies**: All other layers (re-exports only)

**Examples**:
```python
# Old code (still works)
from modules.common.utils import normalize_symbol, log_info, dataframe_to_close_series

# New code (recommended - more explicit)
from modules.common.domain import normalize_symbol
from modules.common.ui.logging import log_info
from modules.common.data import dataframe_to_close_series
```

**Note**: New code should import directly from organized packages. The `utils/` facade exists for backward compatibility only.

---

## Directory Structure

```
modules/common/
├── core/                           # Infrastructure layer
│   ├── data_fetcher/              # Modular data fetching
│   │   ├── __init__.py           # DataFetcher composition class
│   │   ├── base.py               # Core infrastructure
│   │   ├── ohlcv.py              # OHLCV fetching
│   │   ├── binance_prices.py    # Price fetching
│   │   ├── binance_futures.py   # Futures operations
│   │   ├── symbol_discovery.py  # Symbol discovery
│   │   └── exceptions.py         # Custom exceptions
│   ├── exchange_manager/          # Exchange connection management
│   │   ├── __init__.py           # ExchangeManager facade
│   │   ├── base.py               # ExchangeWrapper and utilities
│   │   ├── connection_factory.py # Exchange-specific connections
│   │   ├── authenticated.py      # Authenticated operations
│   │   └── public.py             # Public API operations
│   └── indicator_engine.py       # Indicator orchestration
│
├── domain/                         # Domain layer (pure logic)
│   ├── __init__.py
│   ├── symbols.py                # Symbol utilities
│   └── timeframes.py             # Timeframe utilities
│
├── data/                          # Data layer (orchestration)
│   ├── __init__.py
│   ├── validation.py             # Data validation
│   ├── transformation.py         # Data transformations
│   └── fetchers.py               # High-level fetching
│
├── ui/                            # UI layer
│   ├── __init__.py
│   ├── logging.py                # Logging utilities
│   ├── formatting.py             # Text formatting
│   └── progress_bar.py           # Progress display
│
├── system/                        # System utilities
│   ├── hardware_manager.py       # Hardware management
│   └── gpu_manager.py            # GPU management
│
├── io/                            # File I/O utilities
│   └── file_operations.py        # File operations
│
└── utils/                         # Backward compatibility facade
    ├── __init__.py               # Re-exports from all layers
    └── system_utils.py           # System-specific utilities
```

---

## Dependency Rules

### ✅ Allowed Dependencies

```
Application Code
    ↓
data/ → core/ → domain/
    ↓       ↓
   ui/     ui/
```

**Rules**:
1. **Domain layer**: No dependencies (pure functions)
2. **Infrastructure layer** (`core/`): Can use `domain/` and `ui/`
3. **Data layer**: Can use `domain/`, `core/`, and `ui/`
4. **UI layer**: No dependencies on other layers
5. **Utils layer**: Can import from all layers (re-export only)

### ❌ Forbidden Dependencies

- `domain/` cannot depend on anything
- Lower layers cannot depend on upper layers
- Circular dependencies are not allowed

**Example of forbidden dependency**:
```python
# ❌ BAD: domain depending on core
# in domain/symbols.py
from modules.common.core.data_fetcher import DataFetcher  # WRONG!
```

---

## Import Guidelines

### For New Code (Recommended)

**Always import directly from the specific layer**:

```python
# ✅ GOOD: Explicit imports
from modules.common.domain import normalize_symbol, timeframe_to_minutes
from modules.common.data import dataframe_to_close_series, validate_ohlcv_input
from modules.common.ui.logging import log_info, log_success
from modules.common.core import DataFetcher, ExchangeManager
```

### For Legacy Code (Backward Compatible)

**Using utils facade is acceptable but not recommended**:

```python
# ⚠️ ACCEPTABLE: Works but less explicit
from modules.common.utils import normalize_symbol, log_info, dataframe_to_close_series
```

### Import Patterns by Layer

#### When writing code in `domain/`:
```python
# ✅ No imports from other common layers
import math
import re
from config import DEFAULT_QUOTE  # Only config and stdlib
```

#### When writing code in `core/`:
```python
# ✅ Can import from domain and ui
from modules.common.domain import normalize_symbol, timeframe_to_minutes
from modules.common.ui.logging import log_info, log_error
from modules.common.ui.progress_bar import ProgressBar
```

#### When writing code in `data/`:
```python
# ✅ Can import from domain, core, and ui
from modules.common.domain import normalize_symbol
from modules.common.core.data_fetcher import DataFetcher
from modules.common.ui.logging import log_info, log_error
```

---

## Design Patterns

### 1. Composition Pattern (data_fetcher and exchange_manager)

Both DataFetcher and ExchangeManager use composition to delegate specialized functionality:

**DataFetcher composition:**
```python
class DataFetcher(DataFetcherBase):
    def __init__(self, exchange_manager, shutdown_event=None):
        super().__init__(exchange_manager, shutdown_event)

        # Specialized components
        self._binance_prices = BinancePriceFetcher(self)
        self._binance_futures = BinanceFuturesFetcher(self)
        self._symbol_discovery = SymbolDiscovery(self)
        self._ohlcv = OHLCVFetcher(self)

    # Delegate methods for backward compatibility
    def fetch_current_prices_from_binance(self, symbols):
        return self._binance_prices.fetch_current_prices_from_binance(symbols)
```

**ExchangeManager composition:**
```python
class ExchangeManager:
    def __init__(self, api_key=None, api_secret=None, testnet=False):
        # Specialized managers
        self.authenticated = AuthenticatedExchangeManager(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
        )
        self.public = PublicExchangeManager()

    # Delegate methods for convenience
    def cleanup_unused_exchanges(self, max_age_hours=None):
        self.authenticated.cleanup_unused_exchanges(max_age_hours)
        self.public.cleanup_unused_exchanges(max_age_hours)
```

**Benefits**:
- Clear separation of concerns
- Easy to test components independently
- Can add new fetchers/managers without modifying existing code
- Maintains backward compatibility through delegation

### 2. Facade Pattern (utils)

The `utils/` package provides a simplified interface to the complex subsystem:

```python
# utils/__init__.py
from modules.common.domain import normalize_symbol
from modules.common.data import dataframe_to_close_series
from modules.common.ui.logging import log_info

__all__ = ["normalize_symbol", "dataframe_to_close_series", "log_info", ...]
```

**Benefits**:
- Backward compatibility
- Convenience for simple use cases
- Single import point for common utilities

### 3. Factory Pattern (exchange_manager)

ExchangeConnectionFactory encapsulates exchange-specific creation logic:

```python
class ExchangeConnectionFactory:
    def connect_to_binance_with_credentials(self, manager):
        return manager.connect_to_exchange_with_credentials("binance", ...)

    def connect_to_kraken_with_credentials(self, manager, ...):
        return manager.connect_to_exchange_with_credentials("kraken", ...)

    # ... 6 more exchange methods
```

**Benefits**:
- Isolates exchange-specific logic
- Easy to add new exchanges
- Testable independently from manager

### 4. Wrapper Pattern (exchange_manager)

ExchangeWrapper adds reference counting to ccxt.Exchange:

```python
class ExchangeWrapper:
    def __init__(self, exchange):
        self.exchange = exchange
        self._refcount = 0
        self._refcount_lock = threading.Lock()

    def increment_refcount(self):
        with self._refcount_lock:
            self._refcount += 1
            return self._refcount

    def decrement_refcount(self):
        with self._refcount_lock:
            if self._refcount > 0:
                self._refcount -= 1
            return self._refcount
```

**Benefits**:
- Thread-safe reference tracking
- Prevents premature connection closure
- Automatic cleanup when not in use

### 5. Delegation Pattern (data layer)

High-level orchestration delegates to low-level infrastructure:

```python
# data/fetchers.py
def fetch_ohlcv_data_dict(symbols, timeframes, data_fetcher, ...):
    for symbol in symbols:
        for timeframe in timeframes:
            # Delegate to infrastructure layer
            df, exchange = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol, timeframe, ...
            )
            # Add high-level concerns (validation, logging, etc.)
            validate_ohlcv_input(df)
```

---

## Adding New Code

### Adding a New Domain Utility

**Location**: `modules/common/domain/`

**Steps**:
1. Create file in `domain/` (e.g., `orders.py`)
2. Implement pure functions with no dependencies
3. Add to `domain/__init__.py` exports
4. Optionally add to `utils/__init__.py` for convenience

**Example**:
```python
# modules/common/domain/orders.py
def normalize_order_type(order_type: str) -> str:
    """Normalize order type to standard format."""
    return order_type.upper().strip()

# modules/common/domain/__init__.py
from .orders import normalize_order_type

__all__ = [..., "normalize_order_type"]
```

### Adding a New Data Fetcher Component

**Location**: `modules/common/core/data_fetcher/`

**Steps**:
1. Create new file (e.g., `kraken_futures.py`)
2. Implement component class that takes `DataFetcherBase` in constructor
3. Add component to `DataFetcher` composition in `__init__.py`
4. Add delegation methods to `DataFetcher` class

**Example**:
```python
# modules/common/core/data_fetcher/kraken_futures.py
class KrakenFuturesFetcher:
    def __init__(self, base: "DataFetcherBase"):
        self.base = base

    def fetch_positions(self):
        # Implementation
        pass

# modules/common/core/data_fetcher/__init__.py
from .kraken_futures import KrakenFuturesFetcher

class DataFetcher(DataFetcherBase):
    def __init__(self, exchange_manager, shutdown_event=None):
        super().__init__(exchange_manager, shutdown_event)
        self._kraken_futures = KrakenFuturesFetcher(self)

    def fetch_kraken_futures_positions(self):
        return self._kraken_futures.fetch_positions()
```

### Adding a New Data Orchestration Function

**Location**: `modules/common/data/`

**Steps**:
1. Create or update file in `data/`
2. Implement high-level orchestration
3. Use `domain/` for normalization, `core/` for fetching
4. Add validation and error handling
5. Export from `data/__init__.py`

**Example**:
```python
# modules/common/data/fetchers.py
def fetch_multi_exchange_data(symbols, exchanges, data_fetcher):
    """Fetch data from multiple exchanges for comparison."""
    results = {}
    for symbol in symbols:
        for exchange in exchanges:
            # Use infrastructure layer
            df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol, exchanges=[exchange]
            )
            # Add validation
            if validate_ohlcv_input(df):
                results[(symbol, exchange)] = df
    return results
```

### Adding a New Exchange to ExchangeManager

**Location**: `modules/common/core/exchange_manager/`

**Steps**:
1. Add connection method to `connection_factory.py`
2. Test connection with valid credentials
3. Update `AuthenticatedExchangeManager` to delegate to new method (if needed)
4. Update exchange_manager README with new exchange

**Example**:
```python
# modules/common/core/exchange_manager/connection_factory.py
class ExchangeConnectionFactory:
    def connect_to_coinbase_with_credentials(
        self, manager, api_key=None, api_secret=None, testnet=False, contract_type="spot"
    ):
        """Connect to Coinbase with credentials."""
        return manager.connect_to_exchange_with_credentials(
            "coinbase",
            api_key=api_key or manager.default_api_key,
            api_secret=api_secret or manager.default_api_secret,
            testnet=testnet,
            contract_type=contract_type,
        )

# modules/common/core/exchange_manager/authenticated.py (optional)
def connect_to_coinbase_with_credentials(self):
    """Convenience method for Coinbase connection."""
    return self._factory.connect_to_coinbase_with_credentials(self)
```

---

## Best Practices

### 1. Keep Domain Layer Pure

```python
# ✅ GOOD: Pure function
def normalize_symbol(user_input: str, quote: str = "USDT") -> str:
    return f"{user_input.upper()}/{quote}"

# ❌ BAD: Side effects in domain
def normalize_symbol(user_input: str) -> str:
    log_info(f"Normalizing {user_input}")  # Side effect!
    return f"{user_input.upper()}/USDT"
```

### 2. Use Type Hints

```python
# ✅ GOOD: Clear types
def timeframe_to_minutes(timeframe: str) -> int:
    ...

# ❌ BAD: No type hints
def timeframe_to_minutes(timeframe):
    ...
```

### 3. Document Layer Boundaries

```python
# ✅ GOOD: Clear documentation
"""
Symbol normalization utilities for trading.

This module provides domain-level utilities with no external dependencies.
Safe to use anywhere in the application.
"""
```

### 4. Avoid Circular Dependencies

If you encounter circular imports, you're violating layer boundaries. Refactor:

```python
# ❌ BAD: Circular dependency
# domain/symbols.py
from modules.common.core.data_fetcher import DataFetcher

# ✅ GOOD: Use TYPE_CHECKING for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.common.core.data_fetcher import DataFetcher
```

---

## Testing Strategy

### Unit Tests by Layer

**Domain Layer** (`tests/common/domain/`):
- Test pure functions in isolation
- No mocking needed
- Fast and deterministic

**Infrastructure Layer** (`tests/common/core/`):
- Mock external services (exchanges, APIs)
- Test error handling and retries
- Test caching behavior

**Data Layer** (`tests/common/data/`):
- Mock infrastructure layer
- Test orchestration logic
- Test validation and error handling

---

## Migration Guide

### Migrating from `utils/` to Organized Packages

**Before**:
```python
from modules.common.utils import (
    normalize_symbol,
    timeframe_to_minutes,
    dataframe_to_close_series,
    log_info,
)
```

**After**:
```python
from modules.common.domain import normalize_symbol, timeframe_to_minutes
from modules.common.data import dataframe_to_close_series
from modules.common.ui.logging import log_info
```

**Benefits**:
- Clearer dependencies
- Better IDE autocomplete
- Easier to understand code organization
- Prevents accidental circular dependencies

---

## References

- [DataFetcher Module Documentation](core/data_fetcher/README.md)
- [ExchangeManager Module Documentation](core/exchange_manager/README.md)
- [ExchangeManager Multi-Language Documentation](core/docs/ExchangeManager.md)
- [Project Overview](../../../README.md)

---

## Questions?

If you're unsure where to add new code:

1. **Pure logic with no dependencies?** → `domain/`
2. **External API integration?** → `core/`
3. **Orchestrating multiple operations?** → `data/`
4. **UI or logging?** → `ui/`
5. **Convenience re-export?** → `utils/`

When in doubt, follow the dependency rules: **lower layers cannot depend on upper layers**.
