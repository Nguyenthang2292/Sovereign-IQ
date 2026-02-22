# Gemini Chart Analyzer Module

AI-powered cryptocurrency chart analysis using Google Gemini with a modern service-oriented architecture.

> **Note**: Vietnamese documentation is available in [README_vi.md](README_vi.md)

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI Tools](#cli-tools)
  - [Programmatic API](#programmatic-api)
- [Module Structure](#module-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Recent Refactoring](#recent-refactoring-january-2026)
- [Changelog](CHANGELOG.md)

## Overview

The Gemini Chart Analyzer is a comprehensive module for analyzing cryptocurrency charts using Google's Gemini AI models. It features a clean service-oriented architecture with clear separation between business logic, CLI interfaces, and core functionality.

**Key Highlights:**

- 🏗️ **Service Layer Architecture**: Clean separation of concerns with dedicated service layer
- 🔧 **Modular CLI**: Organized prompts, runners, and configuration management
- 📊 **Type-Safe**: Full typing with dataclasses and enums
- 🛡️ **Exception Hierarchy**: Granular error handling with custom exceptions
- ⚡ **Batch Processing**: Scan entire markets efficiently with cooldown management
- 🎨 **Multi-Timeframe Support**: Analyze across multiple timeframes simultaneously
- 📈 **Random Forest Integration**: ML-powered pre-filtering for better signal quality

## Architecture

### Layered Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                      CLI Layer                               │
│  cli/                                                        │
│  ├── config/       (Configuration management)                │
│  ├── exchange/     (Symbol fetching)                         │
│  ├── models/       (ML model management)                     │
│  ├── prompts/      (User interaction)                        │
│  └── runners/      (Execution orchestration)                 │
└──────────────────────┬───────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                            │
│  services/                                                  │
│  ├── batch_scan_service.py     (Batch market scanning)      │
│  ├── chart_analysis_service.py (Individual analysis)        │
│  └── model_training_service.py (ML model training)          │
└──────────────────────┬───────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                     Core Layer                              │
│  core/                                                      │
│  ├── analyzers/    (Gemini integration)                     │
│  ├── generators/   (Chart creation)                         │
│  ├── scanners/     (Market batch scanner)                   │
│  ├── reporting/    (HTML/CSV/JSON reports)                  │
│  ├── types.py      (Type definitions)                       │
│  └── exceptions.py (Custom exceptions)                      │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Separation of Concerns**: CLI, services, and core logic are clearly separated
2. **Dependency Injection**: Services accept configuration objects and dependencies
3. **Type Safety**: All public APIs use type hints and dataclasses
4. **Error Handling**: Granular exceptions with proper error context
5. **Testability**: Each layer can be tested independently

## Features

### Batch Market Scanning

- **Multi-Exchange Support**: Fetch symbols from Binance, Kraken, KuCoin, etc.
- **Intelligent Pre-filtering**: Optional pre-filter with ATC, Range Oscillator, SPC
- **Random Forest Integration**: ML-based filtering for higher quality signals
- **Cooldown Management**: Configurable delays to respect API rate limits
- **Progress Tracking**: Real-time progress display with confidence scores

### Chart Analysis

- **Single & Multi-Timeframe**: Analyze one or multiple timeframes
- **Custom Indicators**: Configurable technical indicators
- **Flexible Prompts**: General, swing trading, or custom analysis prompts
- **Visual Reports**: HTML reports with embedded charts and analysis

### Model Management

- **ML Model Lifecycle**: Check compatibility, train, and manage Random Forest models
- **Deprecated Feature Detection**: Automatically detect outdated model versions
- **Interactive Training**: CLI-guided training workflow
- **Model Validation**: Comprehensive compatibility checks

## Installation

### Prerequisites

- Python 3.9+ (3.10+ recommended)
- Google Gemini API key (set in environment or `config/config_api.py`)

### Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# ML dependencies (for Random Forest features)
pip install -r requirements-ml.txt
```

### API Key Setup

Run the appropriate setup script for your platform:

```bash
# Windows PowerShell
.\setup\setup_api_keys.ps1

# Linux/Mac
chmod +x setup/setup_api_keys.sh
./setup/setup_api_keys.sh
```

See `setup/QUICK_START_API_KEYS.md` for details.

## Usage

### CLI Tools

#### Batch Market Scanner

Scan the entire market for trading signals:

```bash
# Interactive mode (recommended)
python modules/gemini_chart_analyzer/cli/batch_scanner_main.py

# Load saved configuration
python modules/gemini_chart_analyzer/cli/batch_scanner_main.py --config my_config.json

# Quick scan with defaults
python modules/gemini_chart_analyzer/cli/batch_scanner_main.py --quick
```

**Features:**

- Single or multi-timeframe analysis
- Pre-filter mode (Voting/Hybrid)
- SPC enhancements (volatility adjustment, correlation weights, MTF)
- Random Forest model integration
- Configuration save/load

#### Individual Chart Analyzer

Analyze specific symbols:

```bash
# Interactive mode
python modules/gemini_chart_analyzer/cli/chart_analyzer_main.py

# Direct analysis
python modules/gemini_chart_analyzer/cli/chart_analyzer_main.py \
  --symbol BTC/USDT \
  --timeframe 1h \
  --prompt-type general
```

### Programmatic API

#### Batch Scanning

```python
from modules.gemini_chart_analyzer.services.batch_scan_service import (
    BatchScanConfig,
    run_batch_scan
)

# Configure batch scan
config = BatchScanConfig(
    timeframe="1h",
    max_symbols=50,
    limit=700,
    cooldown=2.5,
    enable_pre_filter=True,
    pre_filter_mode="voting",
    fast_mode=True
)

# Run scan
results = run_batch_scan(config)

# Access results
print(f"Long signals: {results.long_symbols}")
print(f"Short signals: {results.short_symbols}")
print(f"Report: {results.html_report_path}")
```

#### Individual Chart Analysis

```python
from modules.gemini_chart_analyzer.services.chart_analysis_service import (
    SingleAnalysisConfig,
    run_chart_analysis
)
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager

# Initialize dependencies
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Configure analysis
config = SingleAnalysisConfig(
    symbol="BTC/USDT",
    timeframe="1h",
    limit=500,
    prompt_type="general"
)

# Run analysis
results = run_chart_analysis(config, data_fetcher)

# Access results
print(f"Analysis: {results['analysis']}")
print(f"Chart: {results['chart_path']}")
print(f"Report: {results['html_report_path']}")
```

#### Multi-Timeframe Analysis

```python
config = SingleAnalysisConfig(
    symbol="BTC/USDT",
    timeframes_list=["15m", "1h", "4h"],
    limit=500,
    prompt_type="swing_trading"
)

results = run_chart_analysis(config, data_fetcher)
```

## Module Structure

```text
modules/gemini_chart_analyzer/
├── cli/                              # CLI layer
│   ├── config/                       # Configuration management
│   │   ├── display.py               # Display configuration & formatting
│   │   ├── exporter.py              # Configuration export to JSON
│   │   └── loader.py                # Configuration loading from JSON
│   ├── models/                       # ML model management
│   │   └── random_forest_manager.py # RF model lifecycle management
│   ├── prompts/                      # User interaction prompts
│   │   ├── model_training.py        # Model training prompts
│   │   ├── pre_filter.py            # Pre-filter configuration prompts
│   │   ├── spc.py                   # SPC configuration prompts
│   │   └── timeframe.py             # Timeframe selection prompts
│   ├── runners/                      # Execution orchestration
│   │   └── scanner_runner.py        # Batch scanner execution & display
│   ├── batch_scanner_main.py        # Batch scanner CLI entry point
│   └── chart_analyzer_main.py       # Chart analyzer CLI entry point
│
├── services/                         # Service layer
│   ├── batch_scan_service.py        # Batch market scanning service
│   ├── chart_analysis_service.py    # Individual chart analysis service
│   └── model_training_service.py    # RF model training service
│
├── core/                             # Core layer
│   ├── analyzers/                    # Gemini integration
│   │   ├── components/
│   │   │   ├── analyzer_config.py   # Configuration dataclasses & enums
│   │   │   └── token_limit.py       # Token estimation
│   │   ├── gemini_chart_analyzer.py # Main Gemini analyzer
│   │   └── multi_timeframe_coordinator.py # MTF orchestration
│   ├── generators/                   # Chart generation
│   │   ├── chart_generator.py       # Main chart generator
│   │   └── simple_chart_generator.py # Simplified generator
│   ├── scanners/                     # Market scanning
│   │   └── market_batch_scanner.py  # Batch market scanner
│   ├── reporting/                    # Report generation
│   │   ├── html_report.py           # Legacy HTML report
│   │   └── html_report_generator.py # Modular HTML report generator
│   ├── utils/                        # Utilities
│   │   └── chart_paths.py           # Path management
│   ├── types.py                      # Type definitions
│   └── exceptions.py                 # Custom exceptions
│
└── README.md                         # This file
```

## Configuration

### Display Configuration

Configure CLI output formatting:

```python
from modules.gemini_chart_analyzer.cli.config.display import DisplayConfig

config = DisplayConfig(
    confidence_bar_length=10,      # Bar chart width
    confidence_min=0.0,             # Min confidence value
    confidence_max=1.0,             # Max confidence value
    divider_length=60,              # Divider line length
    symbol_column_width=15,         # Symbol column width
    symbols_per_row_fallback=5,    # Symbols per row (fallback)
    fallback_column_width=12        # Fallback column width
)
```

### Gemini Model Configuration

Configure Gemini models:

```python
from modules.gemini_chart_analyzer.core.analyzers.components.analyzer_config import (
    GeminiModelType,
    ImageValidationConfig
)

# Model selection
primary_model = GeminiModelType.PRO_31_PREVIEW
secondary_model = GeminiModelType.PRO_31_PREVIEW_CUSTOMTOOLS
fallback_models = GeminiModelType.get_fallback_models(primary_model)

# Image validation
image_config = ImageValidationConfig(
    max_file_size_mb=20.0,
    max_width=4096,
    max_height=4096,
    min_width=100,
    min_height=100,
    supported_formats=("PNG", "JPEG", "JPG", "WEBP", "GIF")
)
```

### Batch Scan Configuration

Save/load configurations for repeatable scans:

```python
from modules.gemini_chart_analyzer.cli.config.exporter import export_configuration_to_json
from modules.gemini_chart_analyzer.cli.config.loader import load_configuration_from_json

# Export current configuration
config_data = {
    "analysis_mode": "multi-timeframe",
    "timeframes": ["15m", "1h", "4h"],
    "max_symbols": 50,
    "enable_pre_filter": True,
    "pre_filter_mode": "voting",
    "fast_mode": True
}
export_configuration_to_json(config_data, "my_config.json")

# Load configuration
loaded_config = load_configuration_from_json("my_config.json")
```

## Testing

Run the test suite:

```bash
# All gemini_chart_analyzer tests
pytest tests/gemini_chart_analyzer/ -v

# Specific test files
pytest tests/gemini_chart_analyzer/test_gemini_analyzer_helpers.py -v
pytest tests/gemini_chart_analyzer/test_market_batch_scanner.py -v

# With coverage
pytest tests/gemini_chart_analyzer/ --cov=modules.gemini_chart_analyzer --cov-report=html
```

**Test Coverage:**

- ✅ 42 tests passing
- Core components: GeminiModelType, image validation, token estimation
- Batch scanner: initialization, symbol fetching, batch processing
- Market scanner: full workflow with mocked dependencies
- Exception handling and error cases

## API Reference

### Types ([core/types.py](core/types.py))

#### BatchScanResult

```python
@dataclass
class BatchScanResult:
    long_symbols: List[str]                          # Symbols with LONG signals
    short_symbols: List[str]                         # Symbols with SHORT signals
    none_symbols: List[str]                          # Symbols with no signal
    long_symbols_with_confidence: List[Tuple[str, float]]  # LONG with confidence
    short_symbols_with_confidence: List[Tuple[str, float]] # SHORT with confidence
    none_symbols_with_confidence: List[Tuple[str, float]]  # NONE with confidence
    all_results: Dict[str, Any]                      # All raw results
    summary: Dict[str, Any]                          # Summary statistics
    results_file: str                                # CSV results file path
    html_report_path: Optional[str]                  # HTML report path
    status: str                                      # Scan status
    batches_processed: int                           # Number of batches processed
    total_batches: int                               # Total number of batches
```

#### SignalResult

```python
@dataclass
class SignalResult:
    signal: str        # "LONG", "SHORT", or "NONE"
    confidence: float  # 0.0 to 1.0
```

### Exceptions ([core/exceptions.py](core/exceptions.py))

```python
GeminiAnalyzerError         # Base exception
├── ScanConfigurationError  # Invalid scan configuration
├── DataFetchError          # Data fetching failure
├── ChartGenerationError    # Chart generation failure
├── GeminiAnalysisError     # Gemini analysis failure
└── ReportGenerationError   # Report generation failure
```

### Service Layer APIs

#### [batch_scan_service.py](services/batch_scan_service.py)

```python
@dataclass
class BatchScanConfig:
    timeframe: Optional[str] = None
    timeframes: Optional[List[str]] = None
    max_symbols: Optional[int] = None
    limit: int = 700
    cooldown: float = 2.5
    enable_pre_filter: bool = False
    pre_filter_mode: str = "voting"
    fast_mode: bool = True
    initial_symbols: Optional[List[str]] = None
    spc_config: Optional[Dict[str, Any]] = None
    rf_model_path: Optional[str] = None
    skip_cleanup: bool = False
    output_dir: Optional[str] = None

def run_batch_scan(config: BatchScanConfig) -> BatchScanResult:
    """Execute batch market scan with provided configuration."""
```

#### [chart_analysis_service.py](services/chart_analysis_service.py)

```python
@dataclass
class SingleAnalysisConfig:
    symbol: str
    timeframe: Optional[str] = "1h"
    timeframes_list: Optional[List[str]] = None
    indicators: Optional[Dict[str, Any]] = None
    prompt_type: str = "general"
    custom_prompt: Optional[str] = None
    limit: int = 500
    chart_figsize: Tuple[int, int] = (16, 10)
    chart_dpi: int = 150
    output_dir: Optional[str] = None

def run_chart_analysis(config: SingleAnalysisConfig, data_fetcher: DataFetcher) -> Dict[str, Any]:
    """Execute single or multi-timeframe chart analysis."""
```

---

## Recent Refactoring (January 2026)

### Summary

This module underwent a major refactoring that:

- **Reduced codebase by 83%** (3,726 lines removed, 618 added)
- Introduced **service layer** architecture
- Consolidated configuration into typed dataclasses
- Added proper exception hierarchy
- Modularized CLI components
- Improved type safety across the board

### What Changed

| Aspect | Before | After |
| -------- | -------- | ------- |
| **Architecture** | Monolithic CLI scripts | Layered (CLI → Services → Core) |
| **Configuration** | Scattered across files | Centralized dataclasses |
| **Exceptions** | Generic exceptions | Typed exception hierarchy |
| **Types** | Dict[str, Any] returns | Typed dataclasses (BatchScanResult) |
| **CLI** | Single large files | Modular (prompts, runners, config) |
| **Display** | Hardcoded formatting | Configurable (DisplayConfig) |

### Migration Guide

If you were using the old module structure:

**Old imports:**

```python
from modules.gemini_chart_analyzer.core.analyzers.components.helpers import select_best_model
from modules.gemini_chart_analyzer.core.analyzers.components.model_config import GeminiModelType
from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import SymbolFetchError
```

**New imports:**

```python
from modules.gemini_chart_analyzer.core.analyzers.gemini_chart_analyzer import select_best_model
from modules.gemini_chart_analyzer.core.analyzers.components.analyzer_config import GeminiModelType
from modules.gemini_chart_analyzer.core.exceptions import DataFetchError  # replaces SymbolFetchError
```

**Old batch scanning:**

```python
from modules.gemini_chart_analyzer.core.scanners.market_batch_scanner import MarketBatchScanner

scanner = MarketBatchScanner(cooldown_seconds=2.5)
results = scanner.scan_market(timeframe="1h", max_symbols=50)
# results is Dict[str, Any]
```

**New batch scanning:**

```python
from modules.gemini_chart_analyzer.services.batch_scan_service import BatchScanConfig, run_batch_scan

config = BatchScanConfig(timeframe="1h", max_symbols=50, cooldown=2.5)
results = run_batch_scan(config)
# results is BatchScanResult (typed dataclass)
```

**Benefits:**

- ✅ Type-safe results with IDE autocomplete
- ✅ Clear service layer API
- ✅ Better error messages with custom exceptions
- ✅ Reusable services for web APIs or other interfaces

### Files Deleted

These files were consolidated into new, more focused modules:

- `core/analyzers/components/helpers.py` → Merged into `gemini_chart_analyzer.py`
- `core/analyzers/components/image_config.py` → Merged into `analyzer_config.py`
- `core/analyzers/components/model_config.py` → Merged into `analyzer_config.py`
- `core/analyzers/components/response_parser.py` → Removed (parsing simplified)

### New Files Created

Service layer:

- `services/batch_scan_service.py` - Batch scanning business logic
- `services/chart_analysis_service.py` - Chart analysis business logic
- `services/model_training_service.py` - Model training business logic

Type system:

- `core/types.py` - Typed dataclasses for results
- `core/exceptions.py` - Custom exception hierarchy

CLI modularization:

- `cli/config/display.py` - Display configuration
- `cli/config/loader.py` - Configuration loading
- `cli/config/exporter.py` - Configuration export
- `cli/models/random_forest_manager.py` - RF model management
- `cli/prompts/*.py` - User interaction prompts
- `cli/runners/scanner_runner.py` - Execution orchestration

---

## 🤝 Contributing

When contributing to this module:

1. Follow the layered architecture pattern
2. Add type hints to all public APIs
3. Use dataclasses for configuration objects
4. Raise appropriate custom exceptions
5. Add tests for new functionality
6. Update this README with new features

## 📄 License

Part of the Sovereign-IQ (crypto-probability) project. See project root for license information.

---

**For more information, see:**

- Project README: [../../README.md](../../README.md)
- Claude Coding Guidelines: [../../.claude/CLAUDE.md](../../.claude/CLAUDE.md)
- Testing Guide: [../../tests/docs/test_memory_usage_guide.md](../../tests/docs/test_memory_usage_guide.md)
