# Common Module

Common utilities and components shared across all algorithms.

## Structure

```tree
modules/common/
├── core/                    # Core business components
│   ├── data_fetcher.py      # Data fetching from exchanges
│   ├── exchange_manager.py  # Exchange connection management
│   ├── indicator_engine.py  # Technical indicator orchestration
│   └── initialization.py    # Component initialization helpers
├── system/                  # System and hardware management
│   ├── hardware_manager.py  # Hardware resource management (CPU/GPU/RAM)
│   ├── memory_manager.py    # Memory monitoring and cleanup
│   ├── pytorch_gpu_manager.py  # PyTorch GPU detection and management
│   └── system.py            # System configuration and GPU utilities
├── domain/                  # Trading domain utilities (NEW)
│   ├── symbols.py           # Symbol normalization
│   └── timeframes.py        # Timeframe conversion and normalization
├── data/                    # Data utilities (NEW)
│   ├── validation.py        # OHLCV and price series validation
│   ├── transformation.py    # DataFrame/Series transformations
│   └── fetchers.py          # Data fetching helpers
├── io/                      # I/O utilities (NEW)
│   └── file.py              # File operations
├── models/                  # Data models and DTOs
│   └── position.py          # Trading position dataclass
├── ui/                      # UI/CLI utilities
│   ├── progress_bar.py      # Progress bar display
│   ├── logging.py           # Logging functions
│   └── formatting.py        # Text formatting utilities
├── utils/                   # Utility functions (backward compatibility)
│   ├── system_utils.py      # System utilities (Windows stdin, error codes)
│   └── __init__.py          # Re-exports from organized packages
├── indicators/              # Technical indicators
└── quantitative_metrics/    # Quantitative metrics
```

## Core Components

### DataFetcher

Fetches market data (OHLCV, prices, positions) from exchanges with caching and fallback support.

### ExchangeManager

Manages exchange connections with separate managers for authenticated and public API calls.

### IndicatorEngine

Orchestrates technical indicator calculation with support for multiple profiles (CORE, XGBOOST, DEEP_LEARNING).

## Models

### Position

Simple dataclass representing a trading position (symbol, direction, entry_price, size_usdt).

## UI Utilities

### ProgressBar

Thread-safe progress bar for displaying task progress.

### Logging

Colored logging functions organized by severity and domain:

- Standard: `log_info`, `log_success`, `log_error`, `log_warn`, `log_debug`
- Domain-specific: `log_data`, `log_analysis`, `log_model`, `log_exchange`, `log_system`, `log_progress`

### Formatting

Text formatting utilities: `color_text`, `format_price`, `prompt_user_input`, `extract_dict_from_namespace`

## Domain Utilities

### Symbols

Symbol normalization and validation:

- `normalize_symbol`: Converts user input to standard symbol format (e.g., 'btc' → 'BTC/USDT')
- `normalize_symbol_key`: Generates comparison-friendly symbol keys

### Timeframes

Timeframe conversion and normalization:

- `normalize_timeframe`: Normalizes timeframe strings (handles '15m', 'm15', '1h', 'h1')
- `timeframe_to_minutes`: Converts timeframe to minutes
- `days_to_candles`: Calculates number of candles for a given number of days

## Data Utilities

### Validation

OHLCV and price series validation:

- `validate_ohlcv_input`: Validates DataFrame has required OHLCV columns
- `validate_price_series`: Validates high, low, close series alignment

### Transformation

DataFrame/Series transformations:

- `dataframe_to_close_series`: Extracts close price Series from OHLCV DataFrame

### Fetching

Data fetching helpers:

- `fetch_ohlcv_data_dict`: Fetches OHLCV data for multiple symbols/timeframes

## I/O Utilities

### File Operations

- `cleanup_old_files`: Delete old files in a directory with pattern exclusion

## System and Hardware

### Hardware Management

- `HardwareManager`: Automatic CPU/GPU/RAM detection and workload optimization
- `get_hardware_manager`: Get global HardwareManager instance

### Memory Management

- `MemoryManager`: Real-time memory monitoring, cleanup, and leak prevention
- `get_memory_manager`: Get global MemoryManager instance (singleton)
- `MemorySnapshot`: Data class for memory usage snapshots
- `track_memory`: Context manager for tracking memory usage of operations
- `temp_series`: Decorator for automatic cleanup after functions with temporary Series
- `cleanup_series`: Explicit cleanup trigger for large Series/DataFrames

**Features:**

- Real-time RAM and GPU memory monitoring (via psutil and CuPy)
- Automatic garbage collection with threshold-based triggers
- Memory leak detection through snapshot comparison
- Context managers for safe memory operations
- tracemalloc support for detailed memory tracking

### GPU Management

- `PyTorchGPUManager`: PyTorch GPU detection and configuration
- `detect_pytorch_gpu_availability`: Check PyTorch GPU availability
- `configure_gpu_memory`: Configure GPU memory settings
- `detect_gpu_availability`: XGBoost GPU detection

### System Configuration

- `configure_windows_stdio`: Configure Windows UTF-8 encoding
- `get_pytorch_env`: Get PyTorch environment variables

## Initialization

Component initialization: `initialize_components`

## Backward Compatibility

All functions from the old `utils.py` are re-exported through `modules/common/utils/__init__.py`, so existing imports like:

```python
from modules.common.utils import normalize_symbol, log_info, ...
```

continue to work without modification.

## Usage

### Recommended (New Organized Packages)

```python
# Core components
from modules.common import DataFetcher, ExchangeManager, IndicatorEngine
from modules.common.core import initialize_components

# Domain utilities
from modules.common.domain import normalize_symbol, normalize_timeframe, timeframe_to_minutes

# Data utilities
from modules.common.data import validate_ohlcv_input, dataframe_to_close_series

# I/O utilities
from modules.common.io import cleanup_old_files

# System and hardware
from modules.common.system import (
    HardwareManager,
    MemoryManager,
    PyTorchGPUManager,
    get_hardware_manager,
    get_memory_manager,
    track_memory,
)

# UI utilities
from modules.common.ui import ProgressBar, log_info
from modules.common.models import Position
```

### Backward Compatible (Legacy Imports)

All functions are still available through `modules.common.utils` for backward compatibility:

```python
from modules.common.utils import normalize_symbol, timeframe_to_minutes, validate_ohlcv_input
from modules.common.utils import cleanup_old_files, HardwareManager, get_hardware_manager, MemoryManager, get_memory_manager
```

**Note:** New code should use the organized package imports for better clarity and maintainability.
