# SMC Codex Fixes (M01-M07, L01-L06)

## Goal

Resolve medium (M-01 to M-07) and low (L-01 to L-06) issues identified in the 2026-03-02 SMC Codex Review to improve performance, type safety, testability, and code quality.

## Tasks

- [x] Task 1: Fix M-01 (ATR calculation) in `modules/smart_money_concept/core/order_block.py` → Verify: Run `pytest` to ensure no regression after pre-computing ATR outside the structure loop.
- [x] Task 2: Fix M-02 & M-03 (Type safety) in `modules/smart_money_concept/core/order_block.py` → Verify: Run Pyright on `core/order_block.py`; expect 0 type errors for slice indices and list annotations.
- [x] Task 3: Fix M-04 & M-05 (Absolute imports) in `modules/smart_money_concept/core/bos.py` → Verify: Ensure imports are relative (`from .trend import...`, `from ..models import...`) and tests pass.
- [x] Task 4: Fix M-06 & M-07 (Exceptions & Logging) in `charts/choch_chart.py` and `core/equal_hl.py` → Verify: `except (KeyError, ValueError):` used; `print()` replaced with `log_warn` from `modules.common.ui.logging`.
- [x] Task 5: Fix L-01 & L-02 (Exports & Typing) in `core/swing.py` and `core/__init__.py` → Verify: Add `# type: ignore[import-untyped]` for scipy; ensure `identify_equal_hl` and `identify_order_blocks_from_structure` are in `__all__`.
- [x] Task 6: Fix L-05 (Constants duplication) in `models/order_block.py`, `core/order_block.py`, and `core/trend.py` → Verify: Define `BULLISH/BEARISH/NEUTRAL` once and update all imports; run tests.
- [x] Task 7: Fix L-03 & L-04 (Chart DRY refactoring) in `charts/order_block_chart.py` and `charts/bos_chart.py` → Verify: Internal and external drawing logic consolidated via color lookup dictionaries.
- [x] Task 8: Fix L-06 (CLI script path hack) in `modules/smart_money_concept/cli.py` → Verify: Refactor `sys.path` logic for better standard package compatibility.
- [x] Task 9: Final check → Verify: Run `pytest modules/smart_money_concept` and ensure all tests continue to pass (53/53) with zero warnings.

## Done When

- [x] Performance bottleneck for ATR computation in OB logic is removed.
- [x] Pyright type issues and bare exceptions in targeted files are resolved.
- [x] Code duplications in chart rendering are refactored.
- [x] Test suite remains perfectly green.
