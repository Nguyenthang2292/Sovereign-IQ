# Changelog

All notable changes to `modules/gemini_chart_analyzer` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this module follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `core/protocols.py` with protocol-based contracts:
  - `ChartAnalyzerProtocol`
  - `BatchChartAnalyzerProtocol`
  - `ChartGeneratorProtocol`
- Added `core/scanners/batch_processor.py` to isolate batch processing responsibilities from `MarketBatchScanner`.
- Added integration tests with mocked Gemini API flow:
  - `tests/gemini_chart_analyzer/test_pipeline_integration.py`
  - `tests/gemini_chart_analyzer/test_batch_pipeline_integration.py`
- Added architecture tracker updates in `docs/archives/architecture-improvements.md` (Task 7 verification + checklist sync).
- Externalized Gemini prompts into dedicated files under `core/prompts/`:
  - `default.txt`
  - `simple.txt`
  - `detailed.txt`
  - `batch.txt`
- Added service-layer pre-filter configuration models to isolate pre-filter concerns from scanner core:
  - `PreFilterConfig`
  - `Stage0Config`
  - `ATCPerformanceConfig`
  - `XGBoostConfig`

### Changed

- Updated `core/scanners/market_batch_scanner.py` to delegate batch logic to `BatchProcessor`.
- Preserved backward compatibility by keeping thin delegation wrappers:
  - `_process_single_tf_batch(...)`
  - `_process_multi_tf_batch(...)`
- Updated `docs/ARCHITECTURE.md` with design ADRs:
  - ADR-005: Async Batch Processing (Design)
  - ADR-006: Result Caching (Design)
- Expanded ADR-006 details to include:
  - Cache key: `(symbol, timeframe, date, prompt_type)`
  - Storage target: `outputs/cache/` (`diskcache` or SQLite)
  - TTL strategy: 1 hour for batch scans, no cache for single analysis
  - Deferred implementation decision after async groundwork
- Refactored `MarketBatchScanner.scan_market(...)` to use `ScanConfig` object input instead of a large flat parameter list.
- Slimmed `ScanConfig` to scanner-only concerns (`timeframe`, `timeframes`, `max_symbols`, `limit`, `cancelled_callback`, `initial_symbols`, `skip_cleanup`) and removed pre-filter specific fields.
- Moved pre-filter orchestration responsibility from scanner core to service layer (`services/batch_scan_service.py`).
- Updated callers (CLI runner + web API) to pass nested pre-filter config via service-layer config instead of scanner-level fields.
- Migrated `BatchScanConfig` and `SingleAnalysisConfig` to `pydantic.BaseModel` validation flow.
- Updated prompt resolution in analyzer to load prompt templates from file-based assets instead of hardcoded inline strings.

### Fixed

- Removed direct `TIMEFRAME_WEIGHTS` config coupling in core analyzer/aggregator paths via default injection strategy and fallback accessors in module utilities.
- Fixed regression after batch refactor where legacy internal method access in tests expected scanner-level methods.
- Hardened exception chaining in `GeminiChartAnalyzer.analyze_chart(...)` by re-raising `GeminiAnalysisError` with `from e`.
- Added API key env fallback (`GEMINI_API_KEY`) before config-file fallback in Gemini analyzer initialization.
- Audited hot-path bare exception handlers and improved handling/logging consistency for cleanup/non-fatal paths.
- Removed `sys.path` manipulation in pre-filter workflow and replaced with safer import/root resolution strategy.
- Fixed path traversal risk in pre-filter config loader by using shared project-root finder helper.
- Removed deprecated delegate functions from CLI chart analyzer and updated callers to use reporting helpers directly.
- Extracted retry magic numbers into analyzer-level constants for clearer behavior and easier tuning.
- Standardized logging imports to `modules.common.ui.logging` across module files.
- Removed stale backup artifact `core/scanners/market_batch_scanner.py.backup` and ignored `*.backup` patterns.
- Removed obsolete `enhance_futures.md` after merge into codex review documentation.

### Verified

- Verification suite completed on 2026-02-22:
  - `pytest tests/gemini_chart_analyzer/ tests/web/ -v --tb=short`
  - Result: `503 passed, 1 warning`
- `ruff check modules/gemini_chart_analyzer` currently reports pre-existing E501 baseline issues outside this change scope.
- API import smoke check verified key module imports succeed:
  - `MarketBatchScanner`
  - `GeminiChartAnalyzer`
  - `SignalAggregator`
- End-to-end smoke flow verified for single-symbol, single-timeframe analysis path.
