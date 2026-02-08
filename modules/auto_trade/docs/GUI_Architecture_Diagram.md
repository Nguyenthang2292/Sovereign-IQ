# GUI Architecture (Auto Trade)

High-level layout after refactor (Database panel split, config constants, absolute imports).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     AUTO-TRADE GUI ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────────┘

  run_gui.py
       │
       │  sys.path += project_root
       │  from modules.auto_trade.gui.main_window import AutoTradeDashboard
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  AutoTradeDashboard (main_window.py)                                         │
│  • Menu, shortcuts, status bar (StatusBar)                                   │
│  • LayoutManager.create_layout() → tabs                                      │
└────────────────────────────────────────┬─────────────────────────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────────┐
│  AccountFrame   │  │  ConfigPanel    │  │  DatabasePanel (tab)            │
│  PositionsFrame │  │  (API keys      │  │  gui/components/                │
│  SignalsFrame   │  │   masked)       │  │  database_panel.py              │
│  StatsFrame     │  │  RecoveryPanel  │  │  + DatabasePanelConfig          │
│  TradeFormFrame │  └─────────────────┘  │  (gui/config/                   │
│  AutoTradeCtrl  │                       │   database_panel_config.py)     │
└─────────────────┘                       └──────────── ─┬──────────────────┘
                                                         │
                    ┌────────────────────────────────────┼────────────────────┐
                    ▼                                    ▼                    ▼
         ┌──────────────────────┐              ┌──────────────────────┐
         │  Left panel (scroll) │              │  Right panel         │
         │  • DataViewerSection │              │  • StatsSection      │
         │  • OrdersSection     │              │  • ActionsSection    │
         │  • SignalsSection    │              │  • LogsSection       │
         │  • MartingaleSection │              └──────────────────────┘
         │  • RecoverySection   │
         └──────────────────────┘

  Shared components (used by multiple panels):
  • EmptyState     — no positions / no signals / empty list
  • LoadingOverlay — long-running ops
  • All imports:   from modules.auto_trade.gui.*
```

## Key integration points

1. **Imports**: All GUI code under `modules/auto_trade` uses `from modules.auto_trade.gui.*` (no bare `gui.*`).
2. **Config**: Database tab layout, pagination, reconciliation, cleanup, and fonts come from `DatabasePanelConfig`.
3. **Sections**: DatabasePanel composes section components; each section uses config where applicable (e.g. `TITLE_FONT`, `DEFAULT_DAYS_TO_KEEP`).
