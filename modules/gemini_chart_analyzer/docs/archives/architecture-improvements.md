# Architecture Improvements — gemini_chart_analyzer

## Goal

Implement 6 architecture improvements identified in `ARCHITECTURE.md` §11, then complete a final verification/sync task (Task 7) to close the workstream.

---

## Task 1: Add `AnalyzerProtocol` + `GeneratorProtocol`

Define `Protocol` classes so consumers depend on interfaces, not concrete `GeminiChartAnalyzer`.

- [x] Create `core/protocols.py` with 3 protocols → Verify: `ruff check` passes, file exists

```python
# core/protocols.py
from typing import Protocol, Any, Dict, List, Optional

class ChartAnalyzerProtocol(Protocol):
    def analyze_chart(self, image_path: str, symbol: str, timeframe: str,
                      prompt_type: str = "detailed", custom_prompt: Optional[str] = None) -> str: ...

class BatchChartAnalyzerProtocol(Protocol):
    def analyze_batch_chart(self, image_path: str, batch_id: int,
                            total_batches: int, symbols: List[str]) -> Dict[str, Dict[str, Any]]: ...
    def analyze_multi_tf_batch_chart(self, batch_chart_path: str,
                                     symbols: List[str], normalized_timeframes: List[str]) -> Any: ...

class ChartGeneratorProtocol(Protocol):
    def create_chart(self, df: Any, symbol: str, timeframe: str,
                     indicators: Optional[Dict] = None, output_path: Optional[str] = None,
                     show_volume: bool = True) -> str: ...
```

- [x] Update type hints in `MultiTimeframeCoordinator.__init__` and `MarketBatchScanner.__init__` to accept protocols instead of concrete types → Verify: existing tests still pass
- [x] Add 1 test in `tests/test_protocols.py` that creates a mock implementing `ChartAnalyzerProtocol` and passes it to `chart_analysis_service` → Verify: `pytest tests/gemini_chart_analyzer/test_protocols.py` passes

**Files**: `core/protocols.py` (new), `core/analyzers/multi_timeframe_coordinator.py`, `core/scanners/market_batch_scanner.py`, `tests/test_protocols.py` (new)

---

## Task 2: Extract `BatchProcessor` from `MarketBatchScanner`

Move `_process_batches()`, `_process_single_tf_batch()`, `_process_multi_tf_batch()` (~200 LOC) into a dedicated class.

- [x] Create `core/scanners/batch_processor.py` with class `BatchProcessor` containing the 3 methods extracted from `MarketBatchScanner` → Verify: file exists, `ruff check` passes
- [x] Update `MarketBatchScanner._process_batches` to delegate to `BatchProcessor` instance → Verify: `MarketBatchScanner` drops to ~460 LOC
- [x] Run existing tests → Verify: `pytest tests/gemini_chart_analyzer/ tests/web/test_batch_scanner_api.py -v --tb=short` — 0 failures

**Files**: `core/scanners/batch_processor.py` (new), `core/scanners/market_batch_scanner.py`

---

## Task 3: Integration test with mocked Gemini

Create one end-to-end test that validates Fetch→Generate→Analyze→Aggregate without real API calls.

- [x] Create `tests/test_pipeline_integration.py` that:
  1. Mocks `DataFetcher.fetch_ohlcv_with_fallback_exchange` → returns fixture DataFrame
  2. Mocks `GeminiChartAnalyzer.analyze_chart` → returns canned analysis text
  3. Calls `run_chart_analysis(config, data_fetcher)` from `chart_analysis_service`
  4. Asserts: result contains `symbol`, `analysis`, `chart_path`, `html_report_path`
  → Verify: `pytest tests/gemini_chart_analyzer/test_pipeline_integration.py` passes

- [x] Create `tests/test_batch_pipeline_integration.py` that:
  1. Mocks exchange → returns 5 symbols
  2. Mocks `GeminiBatchChartAnalyzer.analyze_batch_chart` → returns JSON with LONG/SHORT signals
  3. Calls `run_batch_scan(config)`
  4. Asserts: result has `total_symbols=5`, `signals` dict, non-zero `scan_duration`
  → Verify: `pytest tests/gemini_chart_analyzer/test_batch_pipeline_integration.py` passes

**Files**: `tests/test_pipeline_integration.py` (new), `tests/test_batch_pipeline_integration.py` (new)

---

## Task 4: Config module isolation (Dependency Injection)

Replace `from config import TIMEFRAME_WEIGHTS` with constructor parameters. 7 files currently import from global `config`.

- [x] `core/aggregators/signal_aggregator.py` — `SignalAggregator.__init__` already accepts `timeframe_weights` param. Remove `from config import TIMEFRAME_WEIGHTS` line, change default to a hardcoded dict literal `{"15m": 0.1, "1h": 0.2, "4h": 0.3, "1d": 0.4}` → Verify: `pytest tests/gemini_chart_analyzer/test_signal_aggregator.py` passes

- [x] `core/analyzers/multi_timeframe_coordinator.py` — same pattern: remove config import, use default dict in constructor → Verify: existing tests pass

- [x] `core/utils/__init__.py` — wrap `TIMEFRAME_WEIGHTS` access in a function `get_default_timeframe_weights()` that tries `config` import with fallback to hardcoded dict → Verify: import smoke test passes

- [x] `core/prefilter/legacy_voting.py` and `core/prefilter/args_builder.py` — these import many config values. Wrap in lazy accessor functions with fallback defaults → Verify: `ruff check` passes

**Files**: `core/aggregators/signal_aggregator.py`, `core/analyzers/multi_timeframe_coordinator.py`, `core/utils/__init__.py`, `core/prefilter/legacy_voting.py`, `core/prefilter/args_builder.py`

---

## Task 5: Async batch processing (Design-only ADR)

This is a large change. Write an ADR documenting the approach without implementing.

- [x] Add ADR-005 to `docs/ARCHITECTURE.md` → "Async Batch Processing" with:
  - Current bottleneck: sequential `for batch in batches` loop + 2.5s cooldowns
  - Proposed: `asyncio.gather()` for data fetching, sequential Gemini calls (rate limit)
  - Trade-off: complexity vs. ~3x speedup on data fetch phase
  - Decision: Deferred until rate limit budget is confirmed
  → Verify: ADR section added to ARCHITECTURE.md

**Files**: `docs/ARCHITECTURE.md`

---

## Task 6: Result caching layer (Design-only ADR)

- [x] Add ADR-006 to `docs/ARCHITECTURE.md` → "Result Caching" with:
  - Cache key: `(symbol, timeframe, date, prompt_type)`
  - Storage: `diskcache` or SQLite in `outputs/cache/`
  - TTL: 1 hour for batch scans, no cache for single analysis
  - Trade-off: stale signals vs. API cost savings (~60%)
  - Decision: Implement after async batch processing
  → Verify: ADR section added to ARCHITECTURE.md

**Files**: `docs/ARCHITECTURE.md`

---

## Task 7: Final verification + checklist sync

- [x] Run final verification suite and confirm no regressions:
  - ✅ `pytest tests/gemini_chart_analyzer/ tests/web/ -v --tb=short` → `503 passed, 1 warning` (2026-02-22)
  - ⚠️ `ruff check modules/gemini_chart_analyzer` reports pre-existing E501 baseline issues outside this task scope
- [x] Reconcile this tracker with actual implementation status and mark `Done When` checkboxes accordingly

**Files**: `docs/architecture-improvements.md`

---

## Done When

- [x] `core/protocols.py` exists with 3 Protocol classes
- [x] `core/scanners/batch_processor.py` exists, `MarketBatchScanner` ≤ 480 LOC
- [x] 2 integration test files pass with mocked Gemini
- [x] `signal_aggregator.py` and `multi_timeframe_coordinator.py` no longer import from `config`
- [x] ADR-005 and ADR-006 documented in ARCHITECTURE.md
- [x] `pytest tests/gemini_chart_analyzer/ tests/web/ -v --tb=short` → 0 failures
