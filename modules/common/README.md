# Common Module

Common utilities and components shared across all algorithms.

## Structure

```
modules/common/
├── core/                    # Core business components
│   ├── data_fetcher.py      # Data fetching from exchanges
│   ├── exchange_manager.py  # Exchange connection management
│   └── indicator_engine.py  # Technical indicator orchestration
├── models/                  # Data models and DTOs
│   └── position.py         # Trading position dataclass
├── ui/                      # UI/CLI utilities
│   ├── progress_bar.py     # Progress bar display
│   ├── logging.py          # Logging functions
│   └── formatting.py       # Text formatting utilities
├── utils/                   # Utility functions (split by domain)
│   ├── system.py           # System utilities (Windows stdio)
│   ├── data.py             # Data manipulation (DataFrame/Series)
│   ├── domain.py           # Trading domain utilities (symbols, timeframes)
│   └── initialization.py  # Component initialization
├── indicators/             # Technical indicators
└── quantitative_metrics/  # Quantitative metrics
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

## Utils

### System
Platform-specific configuration: `configure_windows_stdio`

### Data
DataFrame/Series manipulation: `dataframe_to_close_series`, `validate_ohlcv_input`

### Domain
Trading-specific utilities: `normalize_symbol`, `normalize_symbol_key`, `timeframe_to_minutes`

### Initialization
Component initialization: `initialize_components`

## Backward Compatibility

All functions from the old `utils.py` are re-exported through `modules/common/utils/__init__.py`, so existing imports like:

```python
from modules.common.utils import normalize_symbol, log_info, ...
```

continue to work without modification.

## Usage

```python
from modules.common import DataFetcher, ExchangeManager, IndicatorEngine
from modules.common.models import Position
from modules.common.ui import ProgressBar, log_info
from modules.common.utils import normalize_symbol, timeframe_to_minutes
```
