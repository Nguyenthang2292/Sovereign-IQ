# GUI Architecture & Components (Auto Trade)

Overview of the GUI architecture, layout, and main components after the recent refactoring, which deeply modularized `main_window`, `config_panel`, and `database_panel`.

## High-Level Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                     AUTO-TRADE GUI ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────────┘

  run_auto_trade_gui.py
       │
       │  sys.path += project_root
       │  from modules.auto_trade.gui.main_window import AutoTradeDashboard
       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  AutoTradeDashboard (gui/main_window/main_window.py)                         │
│  • Composed of Multiple Mixins (Lifecycle, Layout, Settings, Scanning,       │
│    Websocket, UI Updates, Risk Manager, Auto Trade)                          │
│  • LayoutManager (gui/main_window/layout.py) → Builds tabs                   │
│  • Menu, Shortcuts, StatusBar                                                │
└────────────────────────────────────────┬─────────────────────────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
┌─────────────────┐  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│  AccountFrame   │  │  ConfigPanel (tab)              │  │  DatabasePanel (tab)            │
│  PositionsFrame │  │  gui/components/config_panel.py │  │  gui/components/database_panel.py
│  SignalsFrame   │  │  + gui/components/              │  │  + gui/components/database/     │
│  StatsFrame     │  │    config_panel_parts/          │  │  + gui/config/                  │
│  TradeFormFrame │  │    - credentials.py             │  │    database_panel_config.py     │
│  AutoTradeCtrl  │  │    - settings_io.py             │  └──────────── ─┬──────────────────┘
│  LogsViewer     │  │    - tab_builders.py            │                 │
│                 │  │    - event_handlers.py          │                 │
└─────────────────┘  └─────────────────────────────────┘                 │
                                                                         │
                    ┌────────────────────────────────────┼───────────────┘
                    ▼                                    ▼
         ┌──────────────────────┐              ┌──────────────────────┐
         │  Left panel (scroll) │              │  Right panel         │
         │  • DataViewerSection │              │  • StatsSection      │
         │  • OrdersSection     │              │  • ActionsSection    │
         │  • SignalsSection    │              │  • LogsSection       │
         │  • MartingaleSection │              └──────────────────────┘
         │  • RecoverySection   │
         └──────────────────────┘
```

## Directory Structure & Modules

The `modules/auto_trade/gui` package is split into detailed functional areas:

- **`main_window/`**: Core logic and UI orchestration for `AutoTradeDashboard`. Divided into multiple mixin classes to prevent a monolithic main class (e.g., `layout.py`, `scanner.py`, `auto_trade.py`, `risk_manager.py`, `websocket_handler.py`, `settings_recovery_mixin.py`, `ui_updates_mixin.py`, etc.).
- **`components/`**: Reusable UI widgets and specific sub-panels.
  - Standard frames: `account_frame.py`, `positions_frame.py`, `signals_frame.py`, `stats_frame.py`, `trade_form.py`, `auto_trade_control.py`, `scanner_control.py`, `logs_viewer.py`.
  - Nested components: `config_panel_parts/` (manages API keys, settings loading/saving, and tabs) and `database/` (handles the sections of the database tab).
  - Shared UI pieces: `empty_state.py`, `loading_overlay.py`, `status_bar.py`.
- **`config/`**: Constants and layout settings (e.g., `database_panel_config.py`).
- **`dialogs/`**: Popup windows (`close_confirmation.py`, `shortcuts_help.py`).
- **`services/`**: GUI-specific services like `database_service.py` for direct SQLite access.
- **`utils/`**: Helper modules to untangle business logic from UI elements. 
  - Ex: `data_service.py`, `websocket_data_service.py` (Real-time updates).
  - Ex: `position_sync_service.py`, `tp_sl_sync.py` (Binance state mapping and TP/SL synchronizations).
  - Ex: `settings_manager.py`, `credential_manager.py` (Secure storage & env validation).
  - Ex: `dry_run_executor.py` (Simulated trade execution DB logic).
  - Ex: `toast.py`, `colors.py`, `formatters.py` (UI helpers).

## Component Integration & Flow

### 1. Main Window and Layout
The `AutoTradeDashboard` inherits from several mixins. Its UI is drawn by `LayoutManager`, which registers sub-frames (like `PositionsFrame`, `SignalsFrame`, `AccountFrame`).
As market data comes in via `WebSocketDataService`, `ui_updates_mixin.py` processes these events and pushes data to `PositionsFrame` or `TradeFormFrame`.

### 2. Config Panel (`gui/components/config_panel.py`)
Because it became too large, `ConfigPanel` orchestrates smaller chunks:
- `tab_builders.py`: Creates the tabs for General, Risk, Auto-Trade, UI.
- `credentials.py`: Secure input masking for API Keys.
- `settings_io.py`: File read/write abstractions.
- `event_handlers.py`: Wires up validation and actions before changes are applied.

### 3. Database Tab (`gui/components/database_panel.py`)
Acts as a multi-pane interface for interacting with the local SQLite `auto_trade.db`:
- **Left sections**: `DataViewerSection` provides a paginated table for all local records. Other sections (`OrdersSection`, `RecoverySection`, etc.) provide quick test functions or specific queries.
- **Right sections**: `StatsSection` summarizes the database. `ActionsSection` initiates reconciliation and cleanups. `LogsSection` tracks execution results of right-panel actions.
- Governed by constants in `DatabasePanelConfig`.

### 4. Utilities & Services Integration
GUI events rarely call directly into core trading modules. Instead, they use intermediate services in `utils/`:
- `WebSocketDataService` listens for Binance socket streams and manages localized callbacks.
- `PositionSyncService` maps live Binance positions to local representations and handles discrepancy resolution.
- `TPSLSync` cleans up orphaned conditional orders on Binance.
- `DryRunExecutor` performs local matching engine calculations for testing order behavior without financial risk.

## Key Design Principles
1. **Separation of Concerns**: UI rendering is separated from internal logic through Mixins (in `main_window/`) and utility services (`utils/`).
2. **Modularity**: Large tabs like Config and Database are broken down into sub-components and orchestrated by a parent frame.
3. **Decoupled Updates**: Core background threads (like websockets or scanner) do not block the Tkinter `mainloop`. Thread-safe queues and callbacks (`tk.after`) are heavily utilized in `ui_updates_mixin.py` and `threading_utils.py`.
