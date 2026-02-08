# Auto Trade Module

Automated crypto trading system: signal pipeline, order execution, database, GUI, and monitoring.

## Overview

- **Core**: Signal pipeline (ATC scanner, XGBoost, Gemini), circuit breaker, health checks.
- **Execution**: Binance client, order manager, risk manager, trailing stop, negative breakeven.
- **Database**: SQLAlchemy models, queries, migrations, backup, reconciliation.
- **GUI**: CustomTkinter dashboard (positions, signals, config, database tab), WebSocket data.
- **Strategies**: Gradual recovery, martingale, recovery manager.
- **Monitoring**: Account monitor, alerts, audit, metrics, lifecycle.

## Structure

```
modules/auto_trade/
├── core/           # Signal pipeline, scanner, health, persistence
├── execution/      # Orders, Binance client, risk, trailing stop
├── database/       # Models, queries, migrations, reconcile
├── gui/            # Dashboard, components, config, utils
│   ├── components/ # PositionsFrame, SignalsFrame, ConfigPanel, DatabasePanel, EmptyState, etc.
│   ├── config/     # DatabasePanelConfig (constants)
│   └── main_window/
├── strategies/     # Gradual recovery, martingale
├── monitoring/     # Alerts, audit, metrics
├── backtest/       # AutoTrade backtester adapter
├── docs/           # Architecture, reviews, implementation summaries
└── run_gui.py      # GUI entry point
```

## Recent changes (refactor)

- **Import paths**: All GUI imports use `modules.auto_trade.gui.*` (no bare `gui.*`).
- **Constants**: `gui/config/database_panel_config.py` — `DatabasePanelConfig` for pagination, fonts, layout, reconciliation, cleanup.
- **Database panel**: Split into sections (Orders, Signals, Martingale, Recovery, DataViewer, Stats, Logs, Actions); layout and strings driven by config.
- **Empty state**: Reusable `EmptyState` component for positions/signals when empty; API key masking in config panel.

## Running

- **GUI**: From repo root with venv, `python modules/auto_trade/run_gui.py` or `python run_auto_trade_gui.py`.
- **Tests**: `pytest tests/auto_trade -v`. Coverage (90% target): `pytest tests/auto_trade --cov=modules/auto_trade --cov-report=term-missing --cov-fail-under=90`.

## Docs

- `docs/` — Architecture diagrams (Phase3, Phase6_5), core reviews, database review.
- `docs/GUI_COMPONENTS.md` — GUI components and DatabasePanel sections.
- `docs/GUI_Architecture_Diagram.md` — GUI layout and refactored database tab.
- Submodules: `database/README.md`, `execution/README.md`, `backtest/README.md`.
