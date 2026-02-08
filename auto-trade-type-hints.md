# Auto Trade – Type Hints

## Goal
Add type hints to all public and internal methods/functions in `modules/auto_trade` so the module is type-safe and mypy-clean (or baselined).

## Tasks
- [x] **1. Database** — Add parameter and return types to `database/` (models, mixins, config, utils, backup, migrations, reconcile, queries/*, __init__.py). Use `typing` and existing SQLAlchemy types.  
  → Verify: `mypy modules/auto_trade/database --no-error-summary 2>&1 | head -20` shows no new errors (or only expected).

- [x] **2. Core** — Add type hints to `core/` (atc_scanner, circuit_breaker, gemini_integration, health, persistence_sqlite, scanner_sampling, signal_pipeline, signal_selector, symbol_manager, xgboost_filter, xgboost_per_symbol).  
  → Verify: `mypy modules/auto_trade/core` passes or errors documented.

- [x] **3. Execution** — Add type hints to `execution/` (binance_client, order_*, risk_manager, trailing_stop*, negative_breakeven*).  
  → Verify: `mypy modules/auto_trade/execution` passes.

- [x] **4. Monitoring** — Add type hints to `monitoring/` (account_monitor, alerts, audit, breakeven_manager, events, event_system, lifecycle_handler, logger, metrics, position_monitor, scanner_scheduler).  
  → Verify: `mypy modules/auto_trade/monitoring` passes.

- [x] **5. Strategies** — Add type hints to `strategies/` (gradual_recovery, martingale, recovery_manager).  
  → Verify: `mypy modules/auto_trade/strategies` passes.

- [x] **6. GUI** — Add type hints to `gui/` (components, main_window, utils, dialogs). Prefer `Optional`, `Callable`, and CTk types where available; use `Any` only where necessary.  
  → Verify: `mypy modules/auto_trade/gui` passes or baseline.

- [x] **7. Websocket, backtest, utils, legacy** — Add type hints to `websocket/`, `backtest/`, `utils/`, `legacy/` and root-level `main.py`, `run_gui.py`, `number_utils.py`, `auto_trade_config.py`, `setup_gui.py`.  
  → Verify: `mypy modules/auto_trade` (full package) runs without new regressions.

- [x] **8. Mypy and verification** — Run `mypy modules/auto_trade` (or `python -m mypy modules/auto_trade`). Fix remaining errors or add a `mypy.ini`/`pyproject.toml` baseline and document.  
  → Verify: CI or local mypy run passes; REFACTORING_RECOMMENDATIONS Day 4 "Add type hints to all methods" marked DONE.

## Done When
- [x] All packages under `modules/auto_trade` have type hints on public and internal functions/methods.
- [x] `mypy modules/auto_trade` passes (or baseline documented and deliverables updated).

## Notes
- Use `from __future__ import annotations` in files with forward references if needed.
- Third-party stubs: use `types-PyYAML`, `types-requests` etc. only if already in deps; otherwise use `Any` for external libs.
- REF: `modules/auto_trade/REFACTORING_RECOMMENDATIONS.md` Day 4 (Type Safety).
