# SMC Module — Codex Review (2026-03-02)

**Scope**: `modules/smart_money_concept/` — full module (core, models, charts, CLI, tests)
**Verdict**: ✅ **Ship-ready** — 53/53 tests pass (0 warnings), architecture clean, PineScript translation bugs resolved per fix-plan.

---

## Architecture Summary

```
smart_money_concept/
├── models/          ← Pure dataclasses (Pivot, OrderBlock) — zero logic
├── core/            ← Stateless business logic (swing, bos, trend, equal_hl, order_block, analyzer)
├── charts/          ← Plotly renderers only (swing, bos, choch, equal_hl, order_block, renderer)
├── cli.py           ← Interactive CLI entry-point
├── analyzer.py      ← Re-export facade (SMCAnalyzer, SMCState)
└── docs/            ← Decision records, audit trails, PineScript source
```

**Separation of concerns**: Models → Core → Charts layering is strict. No Plotly in core, no business logic in models. ✅

---

## Findings

### 🟢 Strengths

| # | Area | Detail |
|---|------|--------|
| S-01 | **Layered architecture** | Clean 3-layer split: `models` (data) → `core` (logic) → `charts` (rendering). No cross-layer pollution. |
| S-02 | **Stateless core** | All `core/` functions are pure — no global state, no module-level mutables. Thread-safe. |
| S-03 | **PineScript fidelity** | 7 critical/medium translation bugs (audit `2026-03-01`) all verified fixed; close-crossover BOS, trend-state CHoCH, structure-linked OBs, volatility filter all match LuxAlgo intent. |
| S-04 | **BosChochResult unification** | BOS and CHoCH share a single dataclass with `event_type` field — eliminates the old "find swing between two BOS timestamps" hack. |
| S-05 | **Volatility filter on OB** | `_apply_volatility_filter` uses `parsedHigh/parsedLow` swap at `2×ATR` — correct PineScript translation. |
| S-06 | **Pivot model richness** | Full-featured `Pivot` class with distance, percentage, pip-aware comparison, recency check, merge. Well validated in 39 dedicated tests. |
| S-07 | **Test suite** | 53 tests covering Pivot (39), BOS/CHoCH (3), Swing (3), Trend (5), Analyzer E2E (2), plus `_last_break_direction`. All pass with `-W error::FutureWarning`. |
| S-08 | **Legacy compatibility** | `identify_bos()` wrapper preserved for backward compat; `SMC_v3_0_legacy.py` still present for reference but not imported by the new pipeline. |

### 🟡 Medium Issues (Non-blocking)

| # | File | Issue | Recommendation |
|---|------|-------|----------------|
| M-01 | `core/order_block.py:40` | `_apply_volatility_filter()` recomputes full-rolling ATR on **every single bar** inside the structure range loop. For N events × M bars each, this is O(N×M×len(df)). | Pre-compute ATR series once per `identify_order_blocks_from_structure()` call and pass it down. |
| M-02 | `core/order_block.py:164` | `df.loc[ob.end:]` slice using `Optional[datetime]` — Pyright reports `Slice index must be an integer, SupportsIndex or None`. Works at runtime but type-unsafe. | Add explicit `cast` or guard `ob.end is not None`. |
| M-03 | `core/order_block.py:72` | `blocks = []` missing type annotation — Pyright: `Need type annotation for "blocks"`. | `blocks: List[OrderBlock] = []` |
| M-04 | `core/bos.py:11` | Absolute import `from modules.smart_money_concept.core.trend import ...` instead of relative. Inconsistent with rest of core which uses `from ..models` etc. | Change to `from .trend import BEARISH, BULLISH, NEUTRAL` for consistency. |
| M-05 | `core/bos.py:12` | Same pattern: `from modules.smart_money_concept.models import Pivot` as absolute. | Change to `from ..models import Pivot`. |
| M-06 | `charts/choch_chart.py:21` | Bare `except:` in `_get_coords()` swallows all exceptions silently. | Use `except (KeyError, ValueError):` or at minimum `except Exception:`. |
| M-07 | `core/equal_hl.py:34` | Uses `print()` for ATR failure logging. Rest of module uses `log_warn` from `modules.common.ui.logging`. | Replace with `log_warn("Unable to compute ATR, returning empty EqualHLResult.")`. |

### 🔵 Low Issues (Cosmetic / Nits)

| # | File | Issue |
|---|------|-------|
| L-01 | `core/swing.py:11` | Pyright: `scipy.signal` missing stubs. Harmless but noisy. Add `# type: ignore[import-untyped]` or add `scipy-stubs`. |
| L-02 | `core/__init__.py` | Exports `identify_bos`, `classify_swing_types` but not `identify_equal_hl` or `identify_order_blocks_from_structure`. Incomplete public surface for downstream consumers. |
| L-03 | `charts/order_block_chart.py` | Internal vs external block rendering duplicates 90% of code; only color varies. Could refactor to a color lookup dict. |
| L-04 | `charts/bos_chart.py` | Same pattern — internal vs external BOS drawing is ~100% duplicated code. |
| L-05 | `models/order_block.py` | `BULLISH/BEARISH/NEUTRAL` constants duplicated in `core/order_block.py` and `core/trend.py`. Single source of truth should be `models/order_block.py` or a shared `constants.py`. |
| L-06 | `cli.py:10-14` | `try/except ImportError` fallback with `sys.path` manipulation. Works, but fragile for editable installs. Consider using `importlib` or ensuring `pip install -e .` is the standard. |

### ✅ Fix-Plan Task Verification (2026-03-01)

| Task | Status | Verified By |
|------|--------|-------------|
| T1: `external_order=50` | ✅ | `swing.py:26`, `analyzer.py:63`, `test_swing.py:53` |
| T2: `equal_hl` own pivot detect (order=3) | ✅ | `equal_hl.py:24,36` |
| T3: Close-crossover BOS | ✅ | `bos.py:81,92`, `test_bos.py:29` |
| T4: Trend-state CHoCH classification | ✅ | `bos.py:114-170`, `test_bos.py:51` |
| T5: `detect_trend` with `last_structure_break` | ✅ | `trend.py:17-32`, `analyzer.py:87`, `test_trend.py:24` |
| T6: Structure-linked OB + volatility filter | ✅ | `order_block.py:77-160`, mitigation `order_block.py:184-189` |
| T7: Analyzer pipeline reconnection | ✅ | `analyzer.py:67-133`, export 15 elements `analyzer.py:135` |
| T8: Tests pass | ✅ | 53 passed, 0 warnings, `-W error::FutureWarning` clean |

---

## Test Coverage Gaps

| Area | Current Tests | Gap |
|------|--------------|-----|
| `core/order_block.py` | None dedicated | **No unit tests** for OB creation, volatility filter, or mitigation logic |
| `core/equal_hl.py` | None dedicated | **No unit tests** for EQH/EQL detection thresholds |
| `charts/*` | None | Rendering layer untested (acceptable — Plotly visual output) |
| `cli.py` | None | CLI untested (acceptable — thin wrapper) |
| Analyzer `export()` | 1 test (length check) | No assertion on content/order of the 15-element tuple |

**Recommendation**: Add dedicated tests for `order_block.py` and `equal_hl.py` to cover the newly refactored logic.

---

## Summary

The SMC module is **well-architected** after the 2026-03-01 refactor. The PineScript translation audit findings are all correctly resolved. The main technical debt is the ATR recomputation hotpath in `order_block.py` (M-01) and missing test coverage for the OB and EQH/EQL sub-modules. Everything else is cosmetic or low-priority consistency cleanup.

**53 passed | 0 failed | 0 warnings | Ship-ready**
