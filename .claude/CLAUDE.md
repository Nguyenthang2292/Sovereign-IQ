# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## MCP Integration

**Important**: Always use Context7 MCP when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.

The Context7 MCP provides access to up-to-date library documentation and code examples. Use it proactively for:

- Library/API documentation lookups
- Code generation based on library patterns
- Setup and configuration guidance
- Best practices and examples

## Project Overview

**Sovereign-IQ** (crypto-probability) is a comprehensive cryptocurrency trading analysis system combining Machine Learning, Deep Learning, and Quantitative Strategies. The system features:

- Multi-exchange data fetching (Binance, Kraken, KuCoin, Gate.io, OKX, Bybit, MEXC, Huobi) with intelligent fallback
- 50+ technical indicators and custom trading strategies
- ML models: XGBoost, LSTM, Temporal Fusion Transformer (TFT), Random Forest, Hidden Markov Models
- Web applications with FastAPI backend and Vue.js 3 + TypeScript frontend
- Comprehensive quantitative metrics for pairs trading and portfolio management
- AI-powered chart analysis using Google Gemini

## Development Environment

### Python Backend

- **Python Version**: 3.12.9 (minimum 3.9+, 3.10+ recommended)
- **Package Manager**: pip
- **Dependencies**:
  - `requirements.txt`: Core dependencies (required)
  - `requirements-ml.txt`: Machine Learning packages (optional)
  - `requirements-dev.txt`: Development tools (optional)

### Frontend (Web Apps)

- **Framework**: Vue.js 3 with Composition API
- **Language**: TypeScript (migration completed for both apps)
- **Build Tool**: Vite 7.3+
- **Package Manager**: npm

Both frontend apps now use TypeScript:

- `web/apps/atc_visualizer/frontend`: ATC Visualizer (Port 5174 dev, 8002 backend)
- `web/apps/gemini_analyzer/frontend`: Gemini Analyzer (Port 5173 dev, 8001 backend)

## Common Development Commands

### Python Backend Setup

```bash
# Install dependencies
pip install -r requirements.txt           # Core dependencies
pip install -r requirements-ml.txt        # ML dependencies (optional)
pip install -r requirements-dev.txt       # Development tools (optional)

# Run tests
pytest                                    # All tests with default config
pytest -v                                 # Verbose output
pytest tests/adaptive_trend/              # Specific module tests
pytest -k "test_signal"                   # Pattern matching
pytest --cov=modules --cov-report=html    # Coverage report

# Memory-optimized testing
pytest -c pytest_memory.ini               # Memory profiling enabled
pytest --skip-memory-intensive            # Skip heavy tests
pytest -n 0                               # Single-threaded execution

# Code quality
black modules/                            # Format code
black --check modules/                    # Check formatting without changes
pylint modules/                           # Lint code
flake8 modules/                           # Alternative linter
mypy modules/                             # Type checking (optional)

# Run main analyzers
python main_complex_hybrid.py --timeframe 1h --enable-spc
python main_complex_voting.py --timeframe 1h --enable-spc
python main_lstm.py --symbol BTC/USDT --timeframe 1h --epochs 50

# Module-specific CLI
python modules/xgboost/cli/main.py --symbol BTC/USDT --timeframe 1h
python modules/hmm/cli/main.py --symbol BTC/USDT --timeframe 1h
python modules/pairs_trading/cli/main.py --sort-by quantitative_score
```

### Frontend (Web Apps) Setup

```bash
# Navigate to frontend directory
cd web/apps/atc_visualizer/frontend       # ATC Visualizer
cd web/apps/gemini_analyzer/frontend      # Gemini Analyzer

# Install dependencies
npm install                               # Install packages

# Development
npm run dev                               # Start dev server (Vite HMR)
npm run build                             # Production build with TypeScript check
npm run preview                           # Preview production build

# TypeScript
vue-tsc --noEmit                          # Type check without emitting files

# Testing (Gemini Analyzer only)
npm run test                              # Run Vitest tests
npm run test:ui                           # Vitest UI mode
npm run test:coverage                     # Coverage report
```

### Web Server Management

```bash
# Start all web apps (recommended)
python main.py                            # Installs deps, builds, starts all servers

# Alternative: Start individual apps
python web/scripts/start_app.py gemini_analyzer
python web/scripts/start_app.py atc_visualizer

# Access URLs
# - ATC Visualizer Frontend:  http://localhost:5174
# - ATC Visualizer Backend:   http://localhost:8002/docs (OpenAPI)
# - Gemini Analyzer Frontend: http://localhost:5173
# - Gemini Analyzer Backend:  http://localhost:8001/docs (OpenAPI)
```

## Architecture Overview

### Layered Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│       CLI Scripts    │    Web Apps    │    API Endpoints     │
└──────────────────────┬──────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Core Analyzer Layer                       │
│  hybrid_analyzer.py  │  voting_analyzer.py │ signal_calculators│
└──────────────────────┬──────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  Strategy/Module Layer                       │
│  ATC │ Range Osc │ SPC │ Pairs │ HMM │ XGBoost │ LSTM │ TFT │
└──────────────────────┬──────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                 Common Utilities Layer                       │
│  DataFetcher │ ExchangeManager │ IndicatorEngine │ Metrics   │
└──────────────────────┬──────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Data Sources Layer                         │
│  Binance │ Kraken │ KuCoin │ ... (with smart fallback)      │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### Core Analyzers (`core/`)

Two distinct analysis workflows (see `core/README.md` for detailed comparison):

1. **Hybrid Analyzer** (`hybrid_analyzer.py`): Sequential filtering approach
   - ATC scan → Range Oscillator filter → SPC → Decision Matrix
   - 70-90% fewer calculations through early filtering
   - Optimal for large symbol pools (100+ symbols)
   - Resource-efficient with fallback mechanisms

2. **Voting Analyzer** (`voting_analyzer.py`): Pure voting approach
   - All indicators calculate signals in parallel
   - Weighted voting based on historical accuracy
   - Higher accuracy potential with more resources
   - Optimal for smaller symbol pools (10-50 symbols)

#### Common Infrastructure (`modules/common/`)

Shared utilities used across all modules:

- **Data Layer**:
  - `core/data_fetcher/`: Multi-exchange fetching with caching and smart fallback (modular design)
    - `__init__.py`: Main DataFetcher class with composition pattern
    - `base.py`: Core infrastructure (caching, shutdown handling)
    - `binance_prices.py`: Binance price fetching
    - `binance_futures.py`: Futures positions and balance operations
    - `symbol_discovery.py`: Symbol discovery for spot and futures markets
    - `ohlcv.py`: OHLCV data fetching with exchange fallback
    - `exceptions.py`: Custom exceptions (SymbolFetchError)
  - `core/exchange_manager.py`: Exchange connection management
  - `core/data_fetcher_forex.py`: Forex market data

- **Indicator Layer**:
  - `core/indicator_engine.py`: Technical indicator orchestration with profiles (CORE, XGBOOST, DEEP_LEARNING)
  - `indicators/`: 50+ technical indicator implementations
  - `quantitative_metrics/`: Statistical tests, risk metrics, hedge ratios

#### Trading Strategy Modules (`modules/`)

Each module is self-contained with clear interfaces:

- **adaptive_trend/**: Multi-layer trend classification with 6 MA types and equity-based weighting
- **range_oscillator/**: 8 oscillator-based strategies with weighted voting
- **simplified_percentile_clustering/**: Cluster-based regime detection
- **decision_matrix/**: Weighted voting system combining multiple indicators
- **pairs_trading/**: Mean-reversion and momentum pairs with comprehensive quantitative metrics
- **portfolio/**: Risk calculation (VaR, Beta), correlation analysis, hedge finding
- **position_sizing/**: Kelly Criterion and risk-based position sizing

#### Machine Learning Modules

- **xgboost/**: Multi-class classifier with dynamic volatility-based labeling and hyperparameter optimization
- **lstm/**: CNN-LSTM-Attention architecture with multi-head attention
- **deeplearning/**: Temporal Fusion Transformer with quantile regression
- **random_forest/**: Ensemble learning with feature importance
- **hmm/**: Three HMM strategies (Swings, KAMA, High-Order) with strategy registry

#### Web Applications (`web/`)

- **Shared Components**: `web/shared/components/` - Reusable Vue components with TypeScript declarations
  - All shared components are typed in `index.d.ts`
  - Import using `@shared` alias (configured in vite.config.ts)
  - Components: CustomDropdown, Input, Button, GlassPanel, LoadingSpinner, Checkbox
- **Backend**: FastAPI with async support, OpenAPI documentation
- **Frontend**: Vue 3 + TypeScript + Vite with hot module replacement
  - Both apps use Composition API with `<script setup lang="ts">`
  - Type-safe props with `defineProps<Interface>()`
  - Type-safe emits with `defineEmits<{ (e: 'event', value: Type): void }>()`

### Configuration System

All configuration centralized in `config/` directory:

- `common.py`: Exchanges, timeframes, symbols, defaults
- `config_api.py`: API keys (uses environment variables - see setup/)
- Strategy configs: `decision_matrix.py`, `range_oscillator.py`, `spc.py`, etc.
- ML configs: `xgboost.py`, `lstm.py`, `deep_learning.py`, `hmm.py`

## TypeScript Migration (Completed)

Both frontend applications have been migrated to TypeScript:

### Configuration

- **tsconfig.json**: Strict mode enabled, ES2020 target, bundler module resolution
- **Build Script**: `vue-tsc --noEmit && vite build` (type-check before build)
- **allowJs**: Enabled for gradual migration compatibility

### Key TypeScript Features

- Strict type checking enabled
- No unused locals/parameters enforcement
- Full Vue 3 SFC support with `<script setup lang="ts">`
- Path aliases configured: `@` (src), `@shared` (shared components)

### Migration Status

- ✅ **Complete**: Both apps fully migrated to TypeScript
- ✅ All source files converted from `.js` to `.ts`
- ✅ All Vue components using `<script setup lang="ts">`
- ✅ Shared components have proper type declarations in `web/shared/components/index.d.ts`
- ✅ Build scripts run type-check before production builds

### TypeScript Best Practices

- **Always use TypeScript** for new code in frontend apps
- **Use `<script setup lang="ts">`** in all Vue SFCs
- **Define proper interfaces** for props, emits, and complex data structures
- **Avoid `any` types**: Use proper typing or `unknown` + type guards
- **Const assertions**: Use `as const` for literal type inference (see `i18n/index.ts` for example)
- **Safe ref access pattern**: Extract ref value to variable before accessing to avoid null checks

  ```typescript
  // ✅ Good
  const existingPoller = logPoller.value
  if (existingPoller) {
    existingPoller.stopPolling()
  }

  // ❌ Avoid
  if (logPoller.value) {
    (logPoller.value as any).stopPolling()
  }
  ```

## API Key Management

**Critical**: Never commit API keys. Use environment variables.

### Setup

Run platform-specific setup script:

```bash
# Windows PowerShell
.\setup\setup_api_keys.ps1

# Windows Command Prompt
setup\setup_api_keys.bat

# Linux/Mac
chmod +x setup/setup_api_keys.sh
./setup/setup_api_keys.sh
```

Scripts are idempotent and safe to run multiple times. See:

- `setup/QUICK_START_API_KEYS.md`: Quick start guide
- `setup/SECURITY.md`: Security best practices

## Testing Strategy

### Test Organization

```text
tests/
├── adaptive_trend/        # ATC module tests
├── xgboost/              # XGBoost tests
├── deeplearning/         # TFT tests
├── lstm/                 # LSTM tests
├── hmm/                  # HMM tests
├── pairs_trading/        # Pairs trading tests
├── portfolio/            # Portfolio tests
├── web/                  # Web API tests
├── e2e/                  # End-to-end tests
└── performance/          # Performance benchmarks
```

### Test Infrastructure

- `pytest.ini`: Default config with parallel processing (pytest-xdist)
- `pytest_memory.ini`: Memory-optimized config with profiling
- `conftest.py`: Shared fixtures
- `conftest_optimized.py`: Memory-optimized fixtures (session scope)

### Memory Optimization

The test suite includes 3-phase memory optimization achieving 80-90% RAM reduction:

- Phase 1: Garbage collection + data reduction (50-60% reduction)
- Phase 2: Session fixtures + parallel processing (30-40% additional)
- Phase 3: Lazy loading + monitoring (10-20% additional)

See `tests/docs/test_memory_usage_guide.md` for details.

## Module Development Guidelines

### Adding New Trading Strategies

1. Create module directory in `modules/your_strategy/`
2. Implement core logic following existing patterns
3. Add configuration in `config/your_strategy.py`
4. Create CLI interface in `modules/your_strategy/cli/main.py`
5. Add comprehensive tests in `tests/your_strategy/`
6. Update `core/signal_calculators.py` to integrate signals
7. Document in module README

### Adding New Web Apps

1. Create app directory: `web/apps/your_app/`
2. Set up backend (FastAPI) and frontend (Vue 3 + TypeScript)
3. Configure unique ports for dev server and backend
4. Add to `main.py` process manager
5. Document in `web/docs/`

See `web/docs/ADDING_NEW_APP.md` for detailed instructions.

### Code Style

- **Python**: Follow PEP 8, use `black` for formatting, add type hints
- **TypeScript**:
  - Use strict mode (already configured)
  - Prefer interfaces over types for objects
  - Use literal types with `as const` for better type inference
  - Use `ReturnType<typeof fn>` utility type for cross-platform compatibility
  - Avoid `any` - use proper types or `unknown` with type guards
- **Vue**:
  - Always use `<script setup lang="ts">` with Composition API
  - Define interfaces for all props and emits
  - Use `ref<Type>()` for typed refs
  - Extract ref values before accessing for cleaner null checks
- **Documentation**: Add docstrings to all public functions and classes

### Documentation Organization

This project follows specific conventions for organizing documentation files:

#### Directory Structure

- **Simple modules**: Keep `README.md` in the module root directory
  - Example: `modules/adaptive_trend/README.md`
  - Use for modules with single documentation file

- **Complex modules**: Use a `docs/` subdirectory for multiple or multi-language documentation
  - Example: `modules/common/core/docs/`
  - Use when module has:
    - Multiple documentation files (architecture, guides, API docs, etc.)
    - Multi-language documentation (English, Vietnamese, etc.)
    - Extensive documentation that would clutter the module directory

#### Multi-Language Documentation Pattern

For modules with multi-language documentation:

1. **Create a `docs/` subdirectory** in the module

```tree
   modules/your_module/
   ├── docs/
   │   ├── YourDoc.md          # Index/language selector
   │   ├── YourDoc-en.md       # English version
   │   └── YourDoc-vi.md       # Vietnamese version
   └── your_module.py
   ```

1. **Create an index file** (e.g., `YourDoc.md`) with language switcher

   ```markdown
   # Your Documentation Title

   > **Language / Ngôn ngữ**: [English](YourDoc-en.md) | [Tiếng Việt](YourDoc-vi.md)

   This is the main documentation index. Please select your preferred language:

   - **[English Documentation](YourDoc-en.md)** - Complete documentation in English
   - **[Vietnamese Documentation](YourDoc-vi.md)** - Tài liệu đầy đủ bằng tiếng Việt
   ```

2. **Create language-specific files** with consistent suffixes
   - `-en.md` for English
   - `-vi.md` for Vietnamese (Tiếng Việt)
   - Add other languages as needed (e.g., `-zh.md`, `-ja.md`)

4. **Maintain consistency** between language versions
   - Same section structure and ordering
   - Synchronized updates across all language versions
   - Consider adding version numbers or last-updated timestamps

#### Current Documentation Examples

**Simple modules** (README in root):

- `modules/adaptive_trend/README.md`
- `modules/decision_matrix/README.md`
- `modules/hmm/README.md`
- `modules/xgboost/README.md`

**Complex modules** (docs/ subdirectory):

- `modules/common/core/docs/ExchangeManager*.md` - Multi-language documentation
- `modules/deeplearning/` - Multiple specialized docs (data pipeline, model, training, etc.)

**Web applications**:

- `web/docs/` - Centralized documentation for all web apps
- `web/docs/ARCHITECTURE.md`, `web/docs/ADDING_NEW_APP.md`

#### Best Practices

1. **Keep documentation close to code**: Place docs in the module they document
2. **Use descriptive filenames**: Clearly indicate the topic (e.g., `ExchangeManager.md`, not `docs.md`)
3. **Provide navigation**: Index files should link to all available documentation
4. **Version tracking**: Consider adding version or date in multi-language docs to track synchronization
5. **Reference from README**: Main project README should link to module-specific documentation
6. **Update CLAUDE.md**: When adding new multi-language docs, update this file's "Multi-Language Documentation" section

## Security Best Practices

### Cursor Rules

This project includes `.cursor/rules/snyk_rules.mdc` for security scanning:

- Always run Snyk code scans for new first-party code
- Fix security issues found in newly introduced/modified code
- Rescan after fixes until no new issues remain

### Key Security Practices

- Never commit API keys or credentials
- Use environment variables for sensitive configuration
- Validate all user inputs at API boundaries
- Use secure WebSocket connections for real-time features
- Keep dependencies updated (especially security patches)

## Important Notes

### Workflow Selection

Choose between Hybrid and Voting analyzers based on requirements:

- **Hybrid**: Faster, resource-efficient, sequential filtering (recommended for >100 symbols)
- **Voting**: More comprehensive, parallel evaluation, potentially higher accuracy (best for <50 symbols)

See `core/README.md` for detailed comparison and decision tree.

### Performance Considerations

- Use caching aggressively (data fetcher, correlation analysis)
- Leverage parallel processing where appropriate (ThreadPoolExecutor)
- Monitor memory usage with memory-intensive ML models
- Use session fixtures for tests to reduce overhead

### Common Patterns

#### Python Backend Patterns

- **Data Fetching**: Always use `DataFetcher` with exchange fallback, never direct exchange API calls
- **Indicators**: Use `IndicatorEngine` with appropriate profile (CORE, XGBOOST, DEEP_LEARNING)
- **Signal Calculation**: Use functions in `core/signal_calculators.py` for consistency
- **Error Handling**: Implement robust error handling with fallback mechanisms
- **Configuration**: All config in `config/` directory, support environment variables

#### TypeScript Frontend

- **Const Assertions**: Use for literal type inference

  ```typescript
  const LOCALES = ['en', 'vi'] as const
  type Locale = typeof LOCALES[number]  // 'en' | 'vi'
  ```

- **Safe Ref Access**: Extract before checking

  ```typescript
  const item = ref.value
  if (item) { item.method() }
  ```

- **Typed Classes**: Define proper class with private fields and typed methods

  ```typescript
  export class MyClass {
    private field: string
    constructor(value: string) { this.field = value }
    method(): void { /* ... */ }
  }
  ```

- **Shared Components**: Import from `@shared/components/` - all have proper types

## Documentation References

- `README.md`: Comprehensive project documentation with usage examples
- `core/README.md`: Hybrid vs Voting analyzer comparison
- `modules/*/README.md`: Module-specific documentation
- `web/docs/ARCHITECTURE.md`: Web architecture details
- `docs/api_documentation.md`: REST API documentation
- `docs/openapi.yaml`: OpenAPI specification
- `tests/docs/test_memory_usage_guide.md`: Test memory optimization guide

### Multi-Language Documentation

Some documentation files are available in multiple languages:

- **ExchangeManager**: `modules/common/core/docs/ExchangeManager.md` (index)
  - English: `modules/common/core/docs/ExchangeManager-en.md`
  - Vietnamese: `modules/common/core/docs/ExchangeManager-vi.md`
  
When working with multi-language documentation:

- The main file (without language suffix) serves as an index/entry point
- Language-specific files use `-en.md` (English) and `-vi.md` (Vietnamese) suffixes
- Always maintain consistency between language versions when updating documentation

## Platform Notes

This project is developed on Windows (win32) and uses Windows-compatible command patterns in automation scripts (`main.py`, `web/scripts/start_app.py`). The codebase supports cross-platform development with appropriate platform checks.
