# Code Review: `modules/xgboost_LTS`

**Reviewer**: GitHub Copilot (automated)  
**Date**: 2026-02-27  
**Scope**: Full module — all Python source files under `modules/xgboost_LTS/`  
**Method**: Static analysis, structural review, cross-file consistency audit  
**Trigger**: `codex-review @modules/xgboost_LTS`

---

## Executive Summary

`xgboost_LTS` is a production-grade multi-class crypto price-direction ML module. The architecture is well-considered: a clean public API, a Rust → Numba → Python fallback chain for hot paths, content-addressed caching, and correct time-series CV with leakage-prevention gaps. The module is actively used by two dependents (`xgboost_LTS_serverless` via lazy import and `auto_trade` via direct consumption).

**One critical bug exists that silently exports wrong symbols** from `core/__init__.py`. Two high-priority issues reduce operational reliability: hardcoded Optuna parallelism config and the absence of module-level unit tests. Several medium and low issues follow; all are fixable with minimal risk.

**Score: 8.5 / 10** — Critical bug (F-01) and four Low/Medium findings resolved; two High items and two Medium items remain open.

> **Update 2026-02-27 (session 2)**: F-01, F-07, F-09, F-10, F-11 confirmed fixed; F-04 partially fixed (`cv_utils.py` created, two callers updated); GPU test fixture gap (F-12) also fixed. Full test suite: **160 passed, 1 skipped, 0 failures**.

---

## Severity Legend

| Icon | Level  | Meaning                                  |
|------|--------|------------------------------------------|
| 🔴   | Critical | Incorrect behaviour or silent data corruption possible |
| 🟠   | High   | Reliability, configurability, or test coverage gap    |
| 🟡   | Medium | Code quality, DRY, or latent performance issue         |
| 🔵   | Low    | Style, readability, cosmetic                           |
| ✅   | Positive | Noteworthy good practice worth preserving             |

---

## Findings

### ~~🔴~~ ✅ F-01 — `core/__init__.py` imports from the wrong parent module — **FIXED**

**File**: `modules/xgboost_LTS/core/__init__.py`  
**Lines**: All import statements

**Problem**  
Every import in the core sub-package init points to `modules.xgboost` (the legacy module) instead of `modules.xgboost_LTS`:

```python
# Current — WRONG
from modules.xgboost.core.labeling import apply_directional_labels
from modules.xgboost.core.model import ClassDiversityError, predict_next_move, train_and_predict
from modules.xgboost.core.optimization import HyperparameterTuner, StudyManager
```

This is a copy-paste artifact from when `xgboost_LTS` was forked from `xgboost`. Any caller importing via the core sub-package (e.g., `from modules.xgboost_LTS.core import train_and_predict`) silently receives the *old* module's symbols — outdated feature engineering, old hyperparameter defaults, different label logic.

The top-level `modules/xgboost_LTS/__init__.py` imports directly from submodules (bypassing `core/__init__.py`), so the **public API is currently safe**. But:
- Any internal cross-module import that goes through `core/` is broken.
- `xgboost_LTS_serverless` uses the top-level API and is unaffected today — but a refactor could silently break it.

**Fix**

```python
# Correct
from modules.xgboost_LTS.core.labeling import apply_directional_labels
from modules.xgboost_LTS.core.model import ClassDiversityError, predict_next_move, train_and_predict
from modules.xgboost_LTS.core.optimization import HyperparameterTuner, StudyManager
```

**Risk of fix**: Zero. Does not change any executed code path that is currently working.

> ✅ **Fixed**: `core/__init__.py` now correctly imports from `modules.xgboost_LTS` on all three lines. Verified in session 2.

---

### 🟠 F-02 — Optuna parallelism hardcoded inside `optimize()` body

**File**: `modules/xgboost_LTS/core/optimization.py`  
**Lines**: Inside `HyperparameterTuner.optimize()` (approximately lines 440–455)

**Problem**  
Two variables controlling Optuna's parallelism are defined as local variables inside the method body:

```python
OPTUNA_PARALLEL_TRIALS = True   # hardcoded
OPTUNA_N_JOBS = -1              # hardcoded — uses ALL cores
```

These look like config constants (UPPER_CASE) but are not exposed to config, constructor, or call-site override. `OPTUNA_N_JOBS = -1` will saturate all CPU cores, potentially starving the trading engine during a live optimization run.

**Fix**

Option A — pass via constructor (preferred):
```python
class HyperparameterTuner:
    def __init__(self, ..., n_jobs: int = -1, parallel_trials: bool = True):
        self.n_jobs = n_jobs
        self.parallel_trials = parallel_trials
```

Option B — add to the central config module:
```python
# config/model_features.py or similar
OPTUNA_PARALLEL_TRIALS: bool = True
OPTUNA_N_JOBS: int = -1
```

Either approach makes production tuning controllable without code changes.

---

### 🟠 F-03 — No unit tests inside the module

**File**: `modules/xgboost_LTS/` (missing `tests/`)

**Problem**  
No test files exist under `modules/xgboost_LTS/`. All four found test files are integration-level tests living outside the module:

- `tests/auto_trade/core/test_xgboost_filter.py` — tests the `XGBoostFilter` wrapper
- `tests/auto_trade/core/test_xgboost_auto_trainer.py` — tests the auto-trainer orchestrator
- `tests/position_sizing/test_xgboost_batch_class_diversity.py` — tests class diversity edge cases
- `modules/auto_trade/tests/test_xgboost_serverless_filter.py` — tests the serverless filter

None test `labeling.py` thresholds, `cache_manager.py` hash correctness, `features.py` feature names, `optimization.py` study persistence, or `cv_parallel.py` split counts directly. A regression in `apply_directional_labels()` thresholds or `_compute_df_hash()` would have no unit-level safety net.

**Recommended coverage targets**:

| Target | Test | Priority |
|--------|------|----------|
| `apply_directional_labels()` | Label distribution with synthetic OHLCV | High |
| `add_price_derived_features()` | Output column names and dtypes | High |
| `CacheManager._compute_df_hash()` | Same df → same hash; mutated df → different hash | High |
| `get_prediction_window()` | Known timeframes round-trip | Medium |
| `StudyManager` | Save/load round-trip with temp file | Medium |
| `run_parallel_cv()` | Returns correct shape with 2-class fixture | Medium |

---

### 🟡 F-04 — Gap-prevention logic triplicated across three files — ⚠️ PARTIALLY FIXED

**Files**: `core/model.py`, `core/optimization.py`, `utils/cv_parallel.py`

**Problem**  
The identical `TARGET_HORIZON` gap-prevention pattern (skipping the last N rows of training to prevent the test set from peering into the training window) is copy-pasted into all three CV execution paths. Each copy is ~8 lines of `np.searchsorted`, index slicing, and a diversity guard.

A future config change to the gap formula must be applied in three places. One missed update silently produces leaky validation metrics.

**Fix**  
Extract to a shared utility, e.g. `utils/cv_utils.py`:

```python
def apply_cv_gap(X: pd.DataFrame, y: pd.Series, gap: int) -> tuple[pd.DataFrame, pd.Series]:
    """Remove the last `gap` rows from training to prevent target leakage."""
    ...
```

All three callers then replace their copy with a single import.

> ⚠️ **Partially Fixed**: `utils/cv_utils.py` created with `apply_cv_gap()`; `cv_parallel.py` updated to use it. `model.py` and `optimization.py` still contain their own copies — two occurrences remain.

---

### 🟡 F-05 — `build_model()` closure re-created on every `train_and_predict()` call

**File**: `core/model.py`  
**Location**: Inside `train_and_predict()`

**Problem**  
`build_model()` is defined as an inner function inside `train_and_predict()`. Python re-creates the function object and its closure on every call. While the overhead is negligible for the function object itself, it couples the model factory tightly to the outer scope and prevents reuse or testing of the factory in isolation.

**Fix**  
Promote `build_model()` to module-level (it has no dependencies that require the closure):

```python
def build_model(params: dict, num_class: int) -> xgb.XGBClassifier:
    ...

def train_and_predict(...):
    model = build_model(params, num_class)
    ...
```

---

### 🟡 F-06 — Final `train_and_predict()` fit reuses the same model instance

**File**: `core/model.py`  
**Location**: Final fit block after CV loop, approximately lines 290–310

**Problem**  
After CV scoring, the code calls `model.fit(X, y, eval_set=...)` on the exact same `XGBClassifier` instance that was last fit on the train split. XGBoost's `fit()` is not guaranteed to be idempotent when called on an already-fitted instance: depending on the version and `n_estimators`, it may append trees rather than replace them, subtly inflating model capacity.

**Fix**  
Instantiate a fresh model for the final fit:

```python
# After CV loop
final_model = build_model(best_params, num_class)
final_model.fit(X, y, eval_set=[(X_eval, y_eval)], verbose=False)
return final_model, cv_scores, ...
```

---

### ~~🟡~~ ✅ F-07 — `CacheManager` has no eviction policy — **FIXED**

**File**: `utils/cache_manager.py`

**Problem**  
Cache files accumulate indefinitely. In a long-running deployment scanning hundreds of symbols across multiple timeframes, the `artifacts/xgboost/` directory grows without bound. There is no maximum size, LRU eviction, or TTL for model caches (label caches also have no expiry beyond the hash mismatch).

**Recommended additions**:
- `max_cache_entries` param on `CacheManager.__init__()` (default `None` = unlimited)
- `_evict_oldest()` called in `save_model()` when count exceeds limit
- Or at minimum, a documented expected growth rate so operators can schedule `clear_cache()` calls

> ✅ **Fixed**: `max_cache_entries` parameter added to `CacheManager.__init__()` (default `None` = unlimited); `_evict_oldest()` called in `save_model()` when entry count exceeds the limit.

---

### 🔵 F-08 — `[DEBUG]` comment residue in `optimization.py`

**File**: `core/optimization.py`  
**Occurrences**: ~8 throughout the file

**Problem**  
Comments of the form `# [DEBUG] ...` left from development/debugging sessions. Harmless, but signals the file was not cleaned before production promotion.

**Fix**: Remove or convert to normal inline comments.

---

### ~~🔵~~ ✅ F-09 — Import order violation in `utils/utils.py` — **FIXED**

**File**: `utils/utils.py`  
**Lines**: 1–5

**Problem**  
A `from config import PREDICTION_WINDOWS` statement appears before the module docstring. PEP 257 and Python convention require the docstring to be the first statement in a module.

**Fix**: Move the import below the docstring.

> ✅ **Fixed**: `from config import PREDICTION_WINDOWS` now appears after the module docstring in `utils/utils.py`.

---

### ~~🔵~~ ✅ F-10 — Redundant `X_test` DataFrame reconstruction in `cv_parallel.py` — **FIXED**

**File**: `utils/cv_parallel.py`  
**Location**: Inside `_train_cv_fold()`, approximately lines 70–110

**Problem**  
`X_test` is created as a DataFrame from the test slice, then converted to a numpy array for the `eval_set`, then reconstructed again as a DataFrame for prediction. One allocation is avoidable.

**Fix**: Build the DataFrame once, pass `X_test.values` for the eval set, and use `X_test` directly for prediction.

> ✅ **Fixed**: `_train_cv_fold()` now builds `X_test` as a DataFrame once and passes `X_test.values` to the eval set directly.

---

### ~~🔵~~ ✅ F-11 — GPU detection makes two separate `nvidia-smi` subprocess calls — **FIXED**

**File**: `utils/gpu_utils.py`

**Problem**  
`detect_cuda_available()` and `get_gpu_info()` each run a separate `subprocess.run(["nvidia-smi", ...])` call, each cached independently via `@lru_cache`. On first call both hit the subprocess; they cannot share results.

**Fix** (optional — not worth disrupting if GPU detection is infrequent):  
A single `_query_nvidia_smi()` cached function returning raw output; both helpers parse that cached result.

> ✅ **Fixed**: `_query_nvidia_smi()` added as an `@lru_cache(maxsize=1)` function; both `detect_cuda_available()` and `get_gpu_info()` delegate to it. One subprocess call total on first invocation.
>
> ⚠️ **Side-effect fixed (F-12)**: This refactor introduced a third `@lru_cache` layer that the test fixture was not clearing. See **F-12** below.

---

### ~~🔵~~ ✅ F-12 — GPU test fixture did not clear the inner `_query_nvidia_smi` cache — **FIXED**

**File**: `tests/xgboost_LTS/test_optimization_features.py`  
**Introduced by**: F-11 fix (session 2)

**Problem**  
After the F-11 refactor added a third `@lru_cache(maxsize=1)` function (`_query_nvidia_smi`), the `clear_gpu_cache` autouse fixture only cleared the two outer caches:

```python
# Before — incomplete
detect_cuda_available.cache_clear()
get_gpu_info.cache_clear()
```

Because `_query_nvidia_smi`'s cache was never cleared, the real GPU name (`"NVIDIA GeForce RTX 4070 SUPER"`) leaked from the first test into every subsequent test. `subprocess.run` was mocked but its return value was bypassed by the already-cached result — the mock call counter stayed at 0 and the GPU name assertion compared to the wrong value.

This caused three test failures:
- `test_detect_cuda_available_success` — `mock_run.assert_called_once()` failed
- `test_detect_cuda_available_caching` — `mock_run.assert_called_once()` failed  
- `test_get_gpu_info` — expected `"Tesla T4"`, got `"NVIDIA GeForce RTX 4070 SUPER"`

**Fix**

```python
from modules.xgboost_LTS.utils.gpu_utils import detect_cuda_available, get_gpu_info, _query_nvidia_smi

@pytest.fixture(autouse=True)
def clear_gpu_cache():
    _query_nvidia_smi.cache_clear()   # ← added; must be first
    detect_cuda_available.cache_clear()
    get_gpu_info.cache_clear()
```

> ✅ **Fixed**: All 3 GPU tests pass; full suite → **160 passed, 1 skipped, 0 failures** (192 s).

---

## Positive Findings

### ✅ P-01 — Excellent multi-layer performance fallback chain

`features.py` and `labeling.py` implement a clean `Rust → Numba → Pure Python` pattern. If Rust extensions are unavailable (cold environment, missing `.pyd`), the module degrades gracefully with no API change to callers. This is the right approach for a cross-environment codebase.

### ✅ P-02 — Correct time-series leakage prevention

`TimeSeriesSplit` is parameterised with `gap=TARGET_HORIZON` everywhere, and the gap logic is consistently applied in model training, CV scoring, and Optuna objective functions. The label NaN exclusion from training data is also correct. This is the most important correctness property and it is implemented well.

### ✅ P-03 — Content-addressed caching with smart hash scope

`CacheManager` computes the model cache hash over the full input DataFrame and uses only OHLCV columns for the label cache hash. This means adding non-OHLCV features (e.g., a new indicator column) does not invalidate the label cache — a thoughtful optimisation that avoids redundant labeling on feature iteration cycles.

### ✅ P-04 — `ClassDiversityError` enables precise caller-side control

Raising a typed subclass of `ValueError` instead of a generic exception lets callers distinguish "not enough class diversity in this window — skip gracefully" from an unexpected training failure. `xgboost_LTS_serverless` uses this correctly in its error routing.

### ✅ P-05 — Pickle-safe parallel CV

`_train_cv_fold()` in `cv_parallel.py` imports `xgboost` lazily *inside* the worker function, avoids passing non-primitive objects across process boundaries, and filters XGBoost params to primitive types before pickling. This is the correct way to use `ProcessPoolExecutor` with XGBoost and avoids the common "object not serializable" failure mode on Windows.

### ✅ P-06 — `num_class` explicitly set from `len(TARGET_LABELS)`

`core/model.py` always passes `num_class=len(TARGET_LABELS)` to the XGBoost classifier rather than inferring it from training data. This prevents a subtle bug where a fold with missing classes would produce a wrong softmax output shape and corrupt probability vectors.

### ✅ P-07 — Cross-platform file locking with exponential backoff

`optimization.py` implements `file_lock()` using `msvcrt.locking` on Windows and `fcntl.flock` on UNIX, with automatic selection. SQLite `database is locked` errors are caught and retried with exponential backoff up to 30s. This is production-appropriate robustness for a multi-process Optuna study.

### ✅ P-08 — Float32 upgrade with overflow guard

`core/model.py` converts features to `float32` for XGBoost (reducing memory ~50%) but skips the conversion if `max_abs_val > 1e6` to prevent silent overflow. This is a careful implementation of a common optimisation that is often done without the guard.

---

## File-by-File Summary

| File | Lines | Severity | Issues |
|------|-------|----------|--------|
| `__init__.py` | ~30 | — | Clean public API; no issues |
| `core/__init__.py` | ~15 | ✅ | ~~**F-01**~~: Fixed — all imports now reference `modules.xgboost_LTS` |
| `core/model.py` | 533 | 🟡 | **F-05**: `build_model` inner closure; **F-06**: final fit on used instance |
| `core/labeling.py` | 374 | — | Clean; correct `bfill` and fallback chain |
| `core/optimization.py` | 554 | 🟠🟡🔵 | **F-02**: hardcoded Optuna params; **F-04** (shared); **F-08**: debug comments |
| `utils/features.py` | 136 | — | Clean Rust→Python fallback; correct guard |
| `utils/cache_manager.py` | 236 | ✅ | ~~**F-07**~~: Fixed — `max_cache_entries` + `_evict_oldest()` added |
| `utils/cv_parallel.py` | 197 | 🟡 | **F-04** (partial — still needs update in `model.py`/`optimization.py`); ~~**F-10**~~: Fixed |
| `utils/numba_funcs.py` | ~55 | — | Correct `prange` parallelism; O(n) running mean |
| `utils/gpu_utils.py` | ~60 | ✅ | ~~**F-11**~~: Fixed — shared `_query_nvidia_smi()` helper |
| `utils/utils.py` | ~20 | ✅ | ~~**F-09**~~: Fixed — import moved below docstring |
| `rust_extensions/__init__.py` | ~30 | — | Clean re-export shim; `type: ignore` appropriate |

---

## Priority Action Plan

> **Status as of 2026-02-27 session 2**: F-01, F-07, F-09, F-10, F-11, F-12 are ✅ resolved. F-04 is ⚠️ partially resolved. Remaining open: F-02, F-03, F-04 (partial), F-05, F-06, F-08.

### ✅ Completed

- ~~**F-01**~~ — `core/__init__.py` imports corrected  
- ~~**F-04** (partial)~~ — `cv_utils.py` created; `cv_parallel.py` migrated  
- ~~**F-07**~~ — cache eviction policy added  
- ~~**F-09**~~ — import order fixed in `utils/utils.py`  
- ~~**F-10**~~ — redundant `X_test` allocation removed  
- ~~**F-11**~~ — shared `_query_nvidia_smi()` helper added  
- ~~**F-12**~~ — GPU test fixture now clears all three LRU caches  

### Should Fix (next sprint)

1. **F-02** — make Optuna `n_jobs` and `parallel_trials` configurable  
2. **F-03** — add unit tests for `labeling.py`, `cache_manager.py`, `features.py`  
3. **F-04** (remainder) — apply `apply_cv_gap()` in `model.py` and `optimization.py`  

### Nice to Have (backlog)

4. **F-05** — promote `build_model` to module scope  
5. **F-06** — use fresh model instance for final fit  

### Cosmetic (any PR touching the file)

6. **F-08** — remove `[DEBUG]` comments from `optimization.py`  

---

## Dependency Note

`modules/xgboost_LTS_serverless` depends on this module via three lazy imports in `handler.py::_imports()`:

- `apply_directional_labels`
- `train_and_predict`  
- `add_advanced_features`

All three are imported from the top-level `__init__.py`, which bypasses the broken `core/__init__.py`. **F-01 does not affect the serverless module today**, but fixing F-01 is still required to prevent future regressions.

---

*Review generated by automated static analysis. All line numbers are approximate — verify against current source before patching.*
