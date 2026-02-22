# gemini_chart_analyzer — Fix Tasks (codex_review_2026-02-22)

## Goal

Resolve all Critical + High issues từ code review; nâng module lên production-ready.

---

## Phase 0 — P0: Critical (fix ngay, < 30 phút)

- [x] **C2** — Chain exception trong `analyze_chart()`
  - File: `core/analyzers/gemini_chart_analyzer.py` line 488
  - Đổi `except Exception:` → `except Exception as e:` + `raise GeminiAnalysisError(...) from e`
  - Verify: `grep -n "raise GeminiAnalysisError" gemini_chart_analyzer.py` chứa `from e`

- [x] **H1** — Xóa backup file
  - File: `core/scanners/market_batch_scanner.py.backup`
  - Chạy: `Remove-Item core/scanners/market_batch_scanner.py.backup`
  - Thêm `*.backup` vào `.gitignore` của module
  - Verify: file không còn tồn tại

- [x] **H2** — Thêm env var fallback cho API key
  - File: `core/analyzers/gemini_chart_analyzer.py` line 118-128
  - Thêm `api_key = os.getenv("GEMINI_API_KEY")` trước khi fallback sang `config.config_api`
  - Verify: `GEMINI_API_KEY=test python -c "from modules.gemini_chart_analyzer import GeminiChartAnalyzer"` → không import error (sẽ fail ở key validation, không phải import)

---

## Phase 1 — P1: High (1-2 giờ)

- [x] **H3** — Audit bare `except Exception` handlers
  - Tìm: `grep -rn "except Exception:" modules/gemini_chart_analyzer/ --include="*.py"`
  - Với mỗi handler: nếu re-raise → thêm `from e`; nếu silent cleanup → OK giữ nhưng thêm `log_warn`
  - Ưu tiên trong: `gemini_chart_analyzer.py:149,182`, `cleanup_manager.py:93`
  - Verify: Không còn `except Exception:` (không có `: e`) trong các hot paths

- [x] **H4** — Extract `ScanConfig` dataclass cho `scan_market()`
  - File: `core/scanners/market_batch_scanner.py`
  - Tạo `ScanConfig` dataclass nhóm 22 params của `scan_market()`
  - Signature mới: `scan_market(self, config: ScanConfig) -> BatchScanResult`
  - Update caller trong `services/batch_scan_service.py`
  - Verify: `cargo check` equivalent — import + instantiate không lỗi

- [x] **H5** — Externalize prompts ra file riêng
  - Tạo `core/prompts/` với 3 files: `detailed.txt`, `simple.txt`, `default.txt`, `batch.txt`
  - Cập nhật `_get_prompt()` để load từ file thay vì hardcode
  - Verify: Prompt vẫn hoạt động khi gọi `analyze_chart()`

---

## Phase 2 — P2: Test Coverage (ưu tiên cao nhất về impact)

- [x] **C1a** — Tạo `tests/test_signal_aggregator.py`
  - Test: empty input, NaN confidence, Inf weight, LONG wins, SHORT wins, tie → NONE
  - Verify: `pytest tests/test_signal_aggregator.py -v` → all pass

- [x] **C1b** — Tạo `tests/test_gemini_batch_chart_analyzer.py`
  - Test: `_extract_json_from_text()` với fenced block, bare JSON, malformed text
  - Test: `_parse_json_response()` với valid/invalid/partial symbol lists
  - Mock Gemini API call
  - Verify: `pytest tests/test_gemini_batch_chart_analyzer.py -v` → all pass

- [x] **C1c** — Tạo `tests/test_gemini_chart_analyzer.py`
  - Test: `validate_image()` — file not found, wrong format, too large, valid
  - Test: `select_best_model()` — None input, empty list, valid model list
  - Mock `genai.Client`
  - Verify: `pytest tests/test_gemini_chart_analyzer.py -v` → all pass

- [x] **C1d** — Tạo `tests/test_chart_generator.py`
  - Test: `_add_indicators()` với từng indicator type (MA, RSI, MACD, BB)
  - Test: chart output path generation
  - Verify: `pytest tests/test_chart_generator.py -v` → all pass

---

## Phase 3 — P3: Medium Issues (cleanup)

- [x] **M3** — Refactor `BatchScanConfig` thành nested dataclasses
  - Tách: `Stage0Config`, `PreFilterConfig`, `ATCPerformanceConfig`, `XGBoostConfig`
  - Verify: `from modules.gemini_chart_analyzer import BatchScanConfig` không lỗi

- [x] **M3b** — Migrate `BatchScanConfig` + `SingleAnalysisConfig` sang `pydantic.BaseModel`
  - Install: `pip install pydantic`
  - Đổi `@dataclass` → `class BatchScanConfig(BaseModel)` — tự động type coercion + validation errors
  - Verify: `BatchScanConfig(timeframe="1h")` không lỗi; `BatchScanConfig(limit="not-a-number")` raise `ValidationError`

- [x] **M4** — Bỏ `sys.path` manipulation trong `workflow.py`
  - Dùng relative import hoặc `importlib.util` thay thế
  - Verify: `run_prefilter_worker()` vẫn chạy đúng

- [x] **M5** — Xóa 4 delegate functions trong `chart_analyzer_main.py`
  - Lines 122-174: `format_text_to_html`, `_sanitize_chart_path`, `_find_chart_paths_for_timeframes`, `generate_html_report`
  - Update callers dùng trực tiếp import từ `html_report_generator`
  - Verify: `python cli/chart_analyzer_main.py --help` không lỗi

- [x] **M7** — Fix path traversal trong `loader.py` line 18
  - Dùng `_find_project_root()` helper từ `workflow.py` thay vì `.parent.parent.parent.parent.parent`
  - Verify: Config loader test vẫn tìm đúng root

---

## Phase A — Architectural Refactor: Tách Pre-filter khỏi module

> **Mục tiêu**: `gemini_chart_analyzer` chỉ làm 1 việc: nhận `List[str]` symbols → trả `BatchScanResult`.
> Pre-filter (ATC, XGBoost, RF, SPC, Stage0...) là trách nhiệm của caller, không phải của module này.

- [x] **A1** — Slim down `ScanConfig`: xóa toàn bộ pre-filter fields
  - File: `core/scanners/market_batch_scanner.py` lines 61-77
  - Xóa: `enable_pre_filter`, `pre_filter_*`, `spc_config`, `stage0_*`, `atc_performance`, `approximate_ma_scanner`, `use_atc_*`, `xgboost_lts`, `use_xgboost_performance`
  - Giữ lại: `timeframe`, `timeframes`, `max_symbols`, `limit`, `cancelled_callback`, `initial_symbols`, `skip_cleanup`
  - Verify: `grep -n "atc_performance\|xgboost_lts" market_batch_scanner.py` → 0 kết quả

- [x] **A2** — Xóa `_apply_pre_filter()` và `_run_pre_filter()` khỏi `MarketBatchScanner`
  - File: `core/scanners/market_batch_scanner.py`
  - Xóa: `from modules.gemini_chart_analyzer.core.prefilter.workflow import run_prefilter_worker` (line 38)
  - Xóa: method `_apply_pre_filter()` (lines 356-428) + `_run_pre_filter()` (lines 430-497)
  - Xóa: block `if config.enable_pre_filter` trong `scan_market()` (lines 266-286)
  - Verify: file giảm từ ~860 xóa còn ~600 dòng; `import` thành công

- [x] **A3** — Tạo `PreFilterConfig` dataclass trong service layer
  - File: `services/batch_scan_service.py`
  - Tạo `PreFilterConfig` chứa toàn bộ 13 fields pre-filter hiện tại
  - `BatchScanConfig` chỉ giữ `pre_filter: Optional[PreFilterConfig] = None` thay vì 13 fields rời
  - Verify: `BatchScanConfig(timeframe="1h")` OK; `BatchScanConfig(pre_filter=PreFilterConfig(enabled=True))` OK

- [x] **A4** — Di chuyển pre-filter logic vào `run_batch_scan()`
  - File: `services/batch_scan_service.py`
  - Nếu `config.pre_filter and config.pre_filter.enabled`: gọi `run_prefilter_worker()` → set `scan_config.initial_symbols`
  - `scanner.scan_market(scan_config)` nhận `initial_symbols` đã lọc sẵn, không biết gì về pre-filter
  - Verify: `run_batch_scan(BatchScanConfig(pre_filter=PreFilterConfig(enabled=True)))` → pre-filter chạy bình thường

- [x] **A5** — Update callers: CLI runner + Web API
  - `cli/runners/scanner_runner.py`: xóa `atc_performance`, `xgboost_lts`... khỏi `ScanConfig` instantiation
  - `web/apps/gemini_analyzer/backend/api/batch_scanner.py`: dùng `BatchScanConfig(pre_filter=PreFilterConfig(...))` thay vì flat fields
  - Verify: `pytest tests/web/test_batch_scanner_api.py -v` → pass

- [x] **A6** — Update test mocks
  - Files: `tests/gemini_chart_analyzer/test_market_batch_scanner.py`, `tests/.../test_batch_scan_service.py`
  - Xóa mock patches cho `run_prefilter_worker` trong `market_batch_scanner` (không còn dùng)
  - Verify: `pytest tests/gemini_chart_analyzer/ -v` → pass

---

## Phase 4 — P4: Low / Cleanup (< 30 phút tổng)

- [x] **L1** — `enhance_futures.md` — content merged into `codex_review_2026-02-22.md`, file deleted ✅
- [x] **L4** — Kiểm tra `market_batch_scanner_forex.py` có được import không; nếu không → xóa hoặc mark `# EXPERIMENTAL`
- [x] **L5** — Extract magic numbers `max_retries=3`, `retry_delay=1` thành class constants trong `GeminiChartAnalyzer`
- [x] **L3** — Chuẩn hóa logging import về `modules.common.ui.logging` trong tất cả files

---

## Phase 5 (Final) — Verification

- [x] `pytest tests/ -v --tb=short` → 0 failures
- [x] `ruff check modules/gemini_chart_analyzer/` → 0 errors (hoặc chỉ warnings đã accept)
- [x] `python -c "from modules.gemini_chart_analyzer import MarketBatchScanner, GeminiChartAnalyzer, SignalAggregator"` → import OK
- [x] End-to-end smoke test: analyze 1 symbol, 1 timeframe → nhận về kết quả LONG/SHORT/NONE

---

## Done When

- [x] 0 Critical issues còn lại
- [x] 0 High issues còn lại
- [x] Tối thiểu 4 test files với coverage cho signal aggregator + JSON parsing
- [x] Backup file đã xóa khỏi git history
- [x] `pytest` pass tất cả tests
- [x] `MarketBatchScanner` / `ScanConfig` không còn field nào liên quan đến ATC hoặc XGBoost
- [x] `PreFilterConfig` tách biệt, nằm trong `services/batch_scan_service.py`
