# Gemini Chart Analyzer — Architecture Analysis

> **Generated**: 2026-02-22  
> **Module**: `modules/gemini_chart_analyzer`  
> **Total Files**: ~92 Python files  
> **Lines (approx)**: ~12,000+ LOC

---

## 1. Executive Summary

The `gemini_chart_analyzer` module is a **vision-based AI trading signal engine** that:

1. Fetches OHLCV market data from crypto exchanges
2. Renders candlestick charts with technical indicators as images
3. Sends chart images to **Google Gemini** for visual analysis
4. Parses LONG/SHORT/NONE signals with confidence scores
5. Aggregates multi-timeframe signals into a final recommendation

It supports both **single-symbol deep analysis** and **batch market scanning** (100+ symbols at once).

---

## 2. Layered Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS                                │
│  cli/batch_scanner/main.py  │  cli/chart_analyzer_main.py          │
│  web/apps/gemini_analyzer/  │  main_gemini_chart_*.py (root)       │
└──────────────────────┬─────────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────────┐
│                     SERVICE LAYER                                  │
│  services/batch_scan_service.py     ← BatchScanConfig (Pydantic)   │
│  services/chart_analysis_service.py ← SingleAnalysisConfig         │
│  services/model_training_service.py ← RF model training            │
└──────────────────────┬─────────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────────┐
│                      CORE LAYER                                    │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  analyzers/  │  │  generators/ │  │  scanners/               │  │
│  │              │  │              │  │                          │  │
│  │ GeminiChart  │  │ ChartGen     │  │ MarketBatchScanner       │  │
│  │  Analyzer    │  │ ChartBatch   │  │  ├─ SymbolFetcher        │  │
│  │ GeminiBatch  │  │  Generator   │  │  ├─ DataFetcherAdapter   │  │
│  │  ChartAnlyzr │  │ ChartMultiTF │  │  ├─ ResultManager        │  │
│  │ MultiTF      │  │  BatchGen    │  │  ├─ CleanupManager       │  │
│  │  Coordinator │  │ SimpleChart  │  │  └─ stdin_protection     │  │
│  └──────────────┘  │  Generator   │  │                          │  │
│                    └──────────────┘  │ MarketBatchScannerForex  │  │
│                                      │  (EXPERIMENTAL)          │  │
│  ┌──────────────┐  ┌──────────────┐  └──────────────────────────┘  │
│  │ aggregators/ │  │  prefilter/  │                                │
│  │              │  │              │  ┌──────────────────────────┐  │
│  │ Signal       │  │ workflow.py  │  │     prompts/             │  │
│  │  Aggregator  │  │ stages.py    │  │  batch.txt               │  │
│  │              │  │ sampling/    │  │  detailed.txt            │  │
│  │              │  │  └─ 6 strats │  │  simple.txt              │  │
│  │              │  │ legacy_      │  │  default.txt             │  │
│  │              │  │  voting.py   │  └──────────────────────────┘  │
│  └──────────────┘  └──────────────┘                                │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  reporting/  │  │    utils/    │  │     exceptions.py        │  │
│  │ HTMLReport   │  │ chart_paths  │  │  scanner_types.py        │  │
│  │ generators/  │  │ timeframe    │  │  plotting_utils.py       │  │
│  │  styles.py   │  │  normalize   │  │                          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────────┐
│                  EXTERNAL DEPENDENCIES                             │
│  modules.common (ExchangeManager, DataFetcher, logging, utils)     │
│  modules.random_forest (RF model training/validation)              │
│  config/ (API keys, TIMEFRAME_WEIGHTS, MODELS_DIR)                 │
│  google-genai / google-generativeai (Gemini API)                   │
│  PIL (Image processing)  │  matplotlib (Chart rendering)           │
│  pydantic (Config validation)                                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

### 3.1 Single Analysis Pipeline

```
SingleAnalysisConfig
       │
       ▼
  DataFetcher.fetch_ohlcv()          ← Exchange API (Binance/Kraken/Kucoin)
       │
       ▼
  ChartGenerator.create_chart()      ← matplotlib → PNG image
       │
       ▼
  GeminiChartAnalyzer.analyze_chart() ← Gemini API (vision model)
       │
       ▼
  generate_html_report()             ← HTML output with analysis text
```

### 3.2 Batch Market Scan Pipeline

```
BatchScanConfig
       │
       ▼
  [Optional] Pre-filter Workflow     ← 4-stage filtering (ATC → OSC/SPC → ML → Decision)
       │                               Runs in service layer (batch_scan_service.py)
       ▼
  MarketBatchScanner.scan_market()
       │
       ├─ SymbolFetcher.get_all_symbols()
       │
       ├─ _split_into_batches()      ← 100 symbols/batch (single TF)
       │                               25 symbols/batch (multi TF)
       │
       └─ For each batch:
           ├─ DataFetcherAdapter → fetch OHLCV for batch of symbols
           ├─ ChartBatchGenerator / ChartMultiTFBatchGen → composite PNG
           ├─ GeminiBatchChartAnalyzer.analyze_batch_chart() → JSON parse
           ├─ ResultManager → aggregate, deduplicate
           └─ Cooldown 2.5s between API calls
       │
       ▼
  SignalAggregator.aggregate_signals() ← Weighted multi-TF aggregation
       │
       ▼
  BatchScanResult                    ← {symbol → signal, confidence, TP/SL}
```

### 3.3 Pre-filter Workflow (4-Stage)

```
All Symbols (~400+)
       │
  Stage 0: Sampling (optional)       ← Random / Stratified / Volume-weighted / etc.
       │
  Stage 1: ATC Performance Scan      ← Keep symbols passing ATC filter
       │
  Stage 2: Oscillator + SPC Voting   ← Technical indicator voting system
       │
  Stage 3: ML Models (XGBoost/RF/HMM)← Machine learning ensemble filter
       │
       ▼
  Filtered Symbols (~50-100)          ← Passed to MarketBatchScanner
```

---

## 4. Key Components & Responsibilities

| Component | File | Responsibility | LOC |
|-----------|------|---------------|-----|
| `GeminiChartAnalyzer` | `core/analyzers/gemini_chart_analyzer.py` | Single chart → Gemini API → text analysis. Model selection, retry w/ exponential backoff, fallback models | ~530 |
| `GeminiBatchChartAnalyzer` | `core/analyzers/gemini_batch_chart_analyzer.py` | Batch chart → Gemini API → JSON parsing. Multi-format response extraction | ~630 |
| `MultiTimeframeCoordinator` | `core/analyzers/multi_timeframe_coordinator.py` | Orchestrates deep & batch multi-TF analysis. Validates/normalizes timeframes | ~390 |
| `MarketBatchScanner` | `core/scanners/market_batch_scanner.py` | Full market scan orchestrator. Manages batching, processing, cleanup | ~660 |
| `ChartGenerator` | `core/generators/chart_generator.py` | Candlestick + indicators → PNG. Uses matplotlib | ~240 |
| `ChartBatchGenerator` | `core/generators/chart_batch_generator.py` | Composite grid charts for batch analysis | ~440 |
| `SignalAggregator` | `core/aggregators/signal_aggregator.py` | Weighted multi-TF signal fusion. Configurable confidence threshold | ~190 |
| `batch_scan_service` | `services/batch_scan_service.py` | Service facade. Pre-filter orchestration, config validation (Pydantic) | ~290 |
| `chart_analysis_service` | `services/chart_analysis_service.py` | Service facade for single/multi-TF analysis | ~162 |
| Pre-filter pipeline | `core/prefilter/` | 4-stage sequential filter with 6 sampling strategies | ~500+ |

---

## 5. Key Data Types

```python
# Core scan configuration (after refactoring)
@dataclass
class ScanConfig:
    timeframe: Optional[str] = "1h"
    timeframes: Optional[List[str]] = None
    max_symbols: Optional[int] = None
    limit: int = 500
    cancelled_callback: Optional[Callable] = None
    initial_symbols: Optional[List[str]] = None
    skip_cleanup: bool = False

# Service-level config (Pydantic, nested)
class BatchScanConfig(BaseModel):
    timeframe, timeframes, limit, cooldown
    pre_filter: PreFilterConfig          # Nested
    atc: ATCPerformanceConfig            # Nested
    xgboost: XGBoostConfig               # Nested
    cancelled_callback, rf_model_path...

# Result types
@dataclass
class SignalResult:
    signal: str          # "LONG" | "SHORT" | "NONE"
    confidence: float    # 0.0 - 1.0

@dataclass
class SymbolScanResult:
    symbol: str
    signal_result: SignalResult
    timeframe_breakdown: Dict[str, SignalResult]

@dataclass
class BatchScanResult:
    results: Dict[str, SymbolScanResult]
    summary: Dict[str, int]   # {"LONG": N, "SHORT": N, "NONE": N}
    total_symbols: int
    scan_duration: float
```

---

## 6. External Dependencies

| Dependency | Used By | Purpose |
|-----------|---------|---------|
| `modules.common.core.exchange_manager` | SymbolFetcher, DataFetcherAdapter | Exchange API (ccxt) |
| `modules.common.core.data_fetcher` | Service layer, Scanners | OHLCV data fetching |
| `modules.common.ui.logging` | Nearly all files | Standardized logging |
| `modules.random_forest` | model_training_service, CLI | RF model training/validation |
| `config` (project root) | SignalAggregator, analyzers | API keys, TIMEFRAME_WEIGHTS |
| `google.genai` / `google.generativeai` | GeminiChartAnalyzer | Gemini API (vision models) |
| `matplotlib` + `mplfinance` | Generators | Chart rendering |
| `PIL` (Pillow) | Analyzer | Image loading/validation |
| `pydantic` | Service configs | Input validation |

---

## 7. Architecture Patterns Observed

### ✅ Strengths

| Pattern | Where | Benefit |
|---------|-------|---------|
| **Layered Architecture** | `cli/` → `services/` → `core/` | Clear separation of concerns |
| **Strategy Pattern** | `prefilter/sampling/strategies/` (6 strategies) | Extensible sampling |
| **Facade Pattern** | `batch_scan_service.py`, `chart_analysis_service.py` | Simple API for complex orchestration |
| **Composite Pattern** | `batch_scanner_components/` (5 sub-modules) | Modular scanner decomposition |
| **Lazy Initialization** | `MarketBatchScanner.batch_gemini_analyzer` (property) | Avoids stdin issues on Windows |
| **Retry + Fallback** | `GeminiChartAnalyzer._call_model_with_retries()` | Resilient API calls with model fallback |
| **Exception Hierarchy** | `exceptions.py` (6 typed exceptions) | Typed error handling per domain |
| **Pydantic Config** | `BatchScanConfig`, `PreFilterConfig` | Validated, documented configs with legacy compat |
| **Data Pipeline** | Fetch → Generate → Analyze → Aggregate | Clear unidirectional flow |

### ⚠️ Concerns & Trade-offs

| Concern | Details | Severity |
|---------|---------|----------|
| **God Class: MarketBatchScanner** | 660 LOC, orchestrates entire batch workflow. Even after extracting 5 sub-modules, `_process_batches()` is still ~100 lines | Medium |
| **Deep coupling to `modules.common`** | 48+ files import from `modules.common`. Module cannot be extracted standalone | Medium |
| **Config import from project root** | `from config import TIMEFRAME_WEIGHTS` — hard dependency on global config | Low |
| **No interface/protocol abstractions** | Components coupled via concrete classes. No `AnalyzerProtocol` or `GeneratorProtocol` | Low |
| **Dual API support in GeminiChartAnalyzer** | `__init__` handles both `genai.Client` (new) and `genai.configure` (legacy) in 100+ lines | Low |
| **`batch_scanner_forex.py` (EXPERIMENTAL)** | Extends `MarketBatchScanner` but uses TradingView scraper — dead code risk | Low |
| **Pre-filter in service layer** | Pre-filter was recently moved from `MarketBatchScanner` to `batch_scan_service.py`. This is correct architecturally but creates a large service function (~130 LOC) | Low |

---

## 8. Module Dependency Graph

```
                     ┌─────────────────  ┐
                     │ Entry Points      │
                     │ (CLI / Web / main)│
                     └───────┬─────────  ┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    batch_scan_service  chart_analysis_  model_training_
                         service          service
              │              │              │
              ▼              ▼              ▼
    ┌─────────────────────────────────────────────┐
    │               CORE LAYER                    │
    │                                             │
    │  MarketBatchScanner ←── GeminiBatchAnalyzer │
    │        │                      ↑             │
    │        ├── SymbolFetcher      │             │
    │        ├── DataFetcherAdapter │             │
    │        ├── ResultManager      │             │
    │        ├── CleanupManager     │             │
    │        └── ChartBatchGenerator│             │
    │                               │             │
    │  GeminiChartAnalyzer ─────────┘             │
    │        ↑                                    │
    │  MultiTimeframeCoordinator                  │
    │        ↑                                    │
    │  SignalAggregator                           │
    │                                             │
    │  ChartGenerator ── plotting_utils           │
    │                                             │
    │  prefilter/workflow ── stages ── sampling/  │
    └─────────────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────────────┐
    │           EXTERNAL MODULES                  │
    │  modules.common  │  modules.random_forest   │
    └─────────────────────────────────────────────┘
```

---

## 9. ADR Log (Architecture Decision Records)

### ADR-001: Pre-filter separated from MarketBatchScanner

- **Status**: Accepted (2026-02-22)
- **Context**: `ScanConfig` contained 18+ pre-filter fields that violated SRP. Scanner was doing both filtering and scanning.
- **Decision**: Move pre-filter logic to `batch_scan_service.py`. `ScanConfig` keeps only 7 core fields. Pre-filter config lives in nested Pydantic models (`PreFilterConfig`, `Stage0Config`, `ATCPerformanceConfig`, `XGBoostConfig`).
- **Trade-off**: Service function is now larger (~130 LOC), but scanner is focused and testable.

### ADR-002: Vision-based analysis via Gemini

- **Context**: Need to analyze chart patterns at scale.
- **Decision**: Generate chart images → send to Gemini vision model → parse JSON response.
- **Trade-off**: More expensive (API cost) and slower than pure numerical analysis, but captures visual patterns (support/resistance, chart formations) that numerical indicators miss.

### ADR-003: Batch chart composition

- **Context**: Sending 400+ individual chart images to Gemini is too slow and expensive.
- **Decision**: Compose 100 mini-charts into a single large image per batch. Gemini analyzes all at once and returns JSON.
- **Trade-off**: Reduced API calls (4 vs 400) but reduced per-symbol image resolution.

### ADR-004: 4-stage pre-filtering pipeline

- **Context**: Scanning all 400+ symbols with Gemini is expensive (~$0.01/image).
- **Decision**: Sequential filtering: Sampling → ATC → Oscillators/SPC → ML models.
- **Trade-off**: Aggressive filtering may miss opportunities, but reduces cost by 60-80%.

### ADR-005: Async Batch Processing (Design)

- **Status**: Proposed (2026-02-22)
- **Context**: The batch scan pipeline is currently I/O bound, sequentially fetching data and analyzing batches.
- **Decision**: Implement `asyncio.gather()` for concurrent data fetching and chart generation, while keeping Gemini API calls sequentially rate-limited (or concurrently if rate limits permit).
- **Trade-off**: Increases complexity (async/await propagation) but expected to yield a ~3x speedup on the data formatting phase.
- **Action**: Deferred until rate-limit budgets are confirmed and architecture stabilizes further.

### ADR-006: Result Caching (Design)

- **Status**: Proposed (2026-02-22)
- **Context**: Re-running scans for the same timeframe and timeframe interval incurs redundant Gemini API costs.
- **Decision**: Add a result cache for batch scan Gemini responses only. Single-symbol interactive analysis remains uncached.
- **Cache Key**: `(symbol, timeframe, date, prompt_type)` normalized and hashed for storage.
- **Storage**: `outputs/cache/` using either `diskcache` (preferred for fast local KV) or SQLite as a fallback implementation.
- **TTL**: 1 hour for batch scans; no cache for single analysis.
- **Trade-off**: Risk of stale signals near market regime changes vs. estimated ~60% API cost reduction on repeated scans.
- **Action**: Implement after ADR-005 async batch groundwork is complete to avoid duplicate refactors.

---

## 10. Test Coverage

| Test File | Scope | Tests |
|-----------|-------|-------|
| `tests/test_chart_generator.py` | ChartGenerator | Chart creation, indicators, subplots |
| `tests/test_gemini_chart_analyzer.py` | GeminiChartAnalyzer | Model selection, analyze_chart |
| `tests/test_gemini_batch_chart_analyzer.py` | GeminiBatchChartAnalyzer | Batch analysis, JSON parsing |
| `tests/test_signal_aggregator.py` | SignalAggregator | Weighted aggregation, edge cases |
| `tests/gemini_chart_analyzer/services/test_batch_scan_service.py` | batch_scan_service | Config validation, pre-filter integration |
| `tests/web/test_batch_scanner_api.py` | Web API endpoints | 37 tests for HTTP layer |
| `tests/gemini_chart_analyzer/test_prefilter_*.py` | Pre-filter stages | Stage 1-3, sampling strategies |

**Gap**: No integration tests that run the full pipeline end-to-end with a mock Gemini API.

---

## 11. Recommendations

### Short-term (Quick Wins)

1. **Add `AnalyzerProtocol`**: Define a `Protocol` for `analyze_chart()` interface so generators and analyzers can be swapped without tight coupling.
2. **Extract batch processing**: Move `MarketBatchScanner._process_batches()` into a dedicated `BatchProcessor` class to further reduce scanner complexity.
3. **Integration test**: Create a single end-to-end test with mocked Gemini API that validates the full Fetch→Generate→Analyze→Aggregate pipeline.

### Medium-term (Architecture Evolution)

1. **Config module isolation**: Replace `from config import ...` with dependency injection. Pass `TIMEFRAME_WEIGHTS` as constructor parameter instead of importing global state.
2. **Async batch processing**: The batch scan pipeline is I/O bound (API calls, data fetching). `asyncio` could improve throughput significantly.
3. **Result caching**: Cache Gemini responses per (symbol, timeframe, date) to avoid redundant API calls on re-scans.

### Long-term (Strategic)

1. **Decouple from `modules.common`**: Define internal interfaces for data fetching and exchange management. Use adapters to bridge to `modules.common`. This enables standalone deployment.
2. **Event-driven results**: Replace synchronous batch result collection with an event stream for real-time UI updates during scanning.
