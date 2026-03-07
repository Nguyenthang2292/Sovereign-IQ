# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Reviewed
- **smart_money_concept**: Full codex-review completed (2026-03-02) — `modules/smart_money_concept/docs/2026-03-02-smc-codex-review.md`
  - ✅ **Ship-ready** — 53/53 tests pass (0 warnings, `-W error::FutureWarning` clean)
  - ✅ All 8 fix-plan tasks from PineScript translation audit verified implemented correctly
  - 🟢 8 strengths: clean 3-layer architecture, stateless core, PineScript fidelity, BosChochResult unification, volatility-filtered OBs, rich Pivot model, comprehensive test suite, legacy backward compat
  - 🟡 7 medium issues (non-blocking): ATR recomputation hotpath in OB (M-01), Pyright type warnings (M-02/M-03), inconsistent absolute imports in bos.py (M-04/M-05), bare except in choch_chart (M-06), print() instead of log_warn in equal_hl (M-07)
  - 🔵 6 low issues: scipy stubs noise, incomplete `core/__init__` exports, duplicated chart rendering code, constant duplication across files, fragile sys.path in CLI
  - ⚠️ Test coverage gaps: no dedicated tests for `order_block.py` or `equal_hl.py`

### Fixed
- **smart_money_concept**: Mismatch fixes applied (2026-03-02)
  - ✅ `core/swing.py`: Docstring `external_order` default corrected from 30 → 50 to match actual code
  - ✅ `core/analyzer.py`: Extracted `_merge_structure_events()` helper to eliminate `FutureWarning` from `pd.concat` with empty/all-NA DataFrames
  - ✅ `core/order_block.py`: Removed dead code function `_get_pivot_time_from_event()` that referenced undefined `df` variable (would crash if ever called)

### Fixed
- **xgboost_LTS**: Post-review fixes applied (2026-02-27) — all items verified by `pytest tests/xgboost_LTS`
  - ✅ **F-01 FIXED** (`core/__init__.py`): All three imports now reference `modules.xgboost_LTS`; critical silent-export bug resolved.
  - ✅ **F-04 PARTIALLY FIXED** (`utils/cv_utils.py` + `utils/cv_parallel.py`): Shared `apply_cv_gap()` helper extracted; `cv_parallel.py` migrated. `model.py` and `optimization.py` still have own copies (remaining open).
  - ✅ **F-07 FIXED** (`utils/cache_manager.py`): `max_cache_entries` constructor parameter and `_evict_oldest()` method added to `CacheManager`; default is unlimited (backward-compatible).
  - ✅ **F-09 FIXED** (`utils/utils.py`): `from config import PREDICTION_WINDOWS` moved below the module docstring.
  - ✅ **F-10 FIXED** (`utils/cv_parallel.py`): Redundant `pd.DataFrame(X_test_vals, columns=…)` allocation removed; `X_test` built once and `.values` passed to the eval set.
  - ✅ **F-11 FIXED** (`utils/gpu_utils.py`): `_query_nvidia_smi()` added as shared `@lru_cache(maxsize=1)` helper; both `detect_cuda_available()` and `get_gpu_info()` delegate to it — single subprocess call.
  - ✅ **F-12 FIXED** (`tests/xgboost_LTS/test_optimization_features.py`): `clear_gpu_cache` autouse fixture now also calls `_query_nvidia_smi.cache_clear()` first; companion fix for F-11's new inner LRU cache. 3 GPU tests that previously failed now pass.
  - ✅ **Test fix** (`tests/xgboost_LTS/test_features_comprehensive.py` line 318): `freq="H"` → `freq="h"` (pandas ≥ 2.2 deprecation).
  - ✅ **Full test suite result: 160 passed, 1 skipped, 0 failures** (192 s)

### Reviewed
- **xgboost_LTS**: Full codex-review completed (2026-02-27) — `modules/xgboost_LTS/docs/2026-02-27-xgboost_LTS_codex_review.md`
  - ~~🔴 **Critical (F-01)**~~: ✅ Fixed — see `### Fixed` above.
  - 🟠 **High (F-02)**: `OPTUNA_PARALLEL_TRIALS = True` and `OPTUNA_N_JOBS = -1` are hardcoded locals inside `HyperparameterTuner.optimize()` — not exposed to config or constructor; saturates all CPU cores in a live session.
  - 🟠 **High (F-03)**: No unit tests inside `modules/xgboost_LTS/`; only 4 integration-level test files exist externally. Labeling thresholds, cache hashing, and feature column names have no safety net.
  - 🟡 **Medium (F-04)** *(partial)*: `TARGET_HORIZON` gap-prevention CV logic — `utils/cv_utils.py` extracted; `model.py` and `optimization.py` still have own copies.
  - 🟡 **Medium (F-05)**: `build_model()` defined as inner closure inside `train_and_predict()` — re-created on every call; promote to module scope.
  - 🟡 **Medium (F-06)**: Final full-data fit in `train_and_predict()` calls `fit()` on the same instance used in the last CV fold; XGBoost may append trees — use a fresh instance.
  - ~~🟡 **Medium (F-07)**~~: ✅ Fixed — see `### Fixed` above.
  - 🔵 **Low (F-08)**: `[DEBUG]` comment residue (~8 occurrences) in `optimization.py` — still open.
  - ~~🔵 **Low (F-09, F-10, F-11)**~~: ✅ All fixed — see `### Fixed` above.
  - ✅ **Strengths**: Rust→Numba→Python fallback chain; correct `TimeSeriesSplit` gap with `TARGET_HORIZON`; smart OHLCV-only label cache hash; `ClassDiversityError` typed exception; pickle-safe parallel CV; `num_class` pinned from `len(TARGET_LABELS)`; cross-platform file locking with exponential SQLite retry; float32 overflow guard.

### Added
- **gemini_gann_square**: New module — Gann Square technical analysis + Gemini AI (2026-02-25)
  - `core/swing_detector.py` — Pivot Zigzag algorithm for Swing High/Low detection
  - `core/gann_calculator.py` — 4-zone Gann Square builder with trend-aware signals (LONG/SHORT/SKIP)
  - `core/gann_chart_generator.py` — Candlestick chart with zone overlays, swing markers, current price line
  - `core/gann_signal_engine.py` — Full orchestrator: fetch → detect → calculate → chart → Gemini AI → parse
  - `cli/` — CLI entry point with argparse and interactive menu
  - `prompts/gann_analysis.txt` — Structured Gemini prompt template with placeholder injection
  - 39 unit tests (100% pass) covering calculator zones, trend detection, swing detection, and edge cases
  - Code review: fixed 5 lint issues (unused imports, import ordering, line length)

### Changed
- **binance_client**: Refactored into modular sub-package architecture (2026-02-11)
  - Split monolithic `binance_client.py` (793 lines) into focused modules:
    - `binance/exchange_setup.py` - CCXT exchange initialization
    - `binance/order_execution.py` - Market orders with TP/SL placement
    - `binance/position_management.py` - Position operations
    - `binance/order_management.py` - TP/SL modification and cancellation
    - `binance/client.py` - Main orchestrator with backward compatibility
  - Maintained 100% backward compatibility via legacy import layer
  - All 35 critical tests passing (trailing stop, fresh signal, order executor)
  - Benefits: Better separation of concerns, easier testing, improved maintainability
  - Added comprehensive README documenting new architecture

### Added
- **auto_trade**: Integration tests (Day 3)
  - End-to-end workflow tests (`tests/auto_trade/integration/test_e2e_workflows.py`)
    - Database init/migrate/insert/query full workflow
    - Signal pipeline with mocked components
    - Reconcile workflow with mocked Binance exchange
    - Backup create and verify workflow
  - Performance benchmarks (`tests/auto_trade/integration/test_performance_benchmarks.py`)
    - get_overall_stats with 10k+ orders (< 5s)
    - get_orders_cursor first page (< 1s)
    - Backup creation (< 10s for 10k-order DB)
    - Reconcile with mocked exchange (< 2s)
  - Stress tests (`tests/auto_trade/integration/test_stress.py`)
    - High-volume stats (5k+ orders)
    - Cursor pagination through large dataset
    - Concurrent stats reads
    - Concurrent reconcile calls (serialized via lock)

### Changed
- **auto_trade**: Week 5 Quality & Polish completion
  - Day 3 Integration Testing completed
  - Day 4 Final Review tasks documented

## [3.0.0] - Previous

- See project history for earlier releases.
