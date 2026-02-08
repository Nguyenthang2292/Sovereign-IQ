# GUI Components (Auto Trade)

Overview of main GUI components and the refactored Database tab.

## Main window and layout

- **AutoTradeDashboard** (`gui/main_window.py`): Main window, menu, shortcuts, status bar.
- **LayoutManager** (`gui/main_window/layout.py`): Builds tabs and places components (Account, Config, Positions, Signals, Stats, Trade form, Database panel).
- **DataService / WebSocketDataService**: Price and position updates.

## Shared components

| Component | Path | Purpose |
|-----------|------|---------|
| **EmptyState** | `gui/components/empty_state.py` | Icon + message + optional hint and action button when a list is empty (e.g. no positions, no signals). |
| **LoadingOverlay** | `gui/components/loading_overlay.py` | Overlay for long-running operations. |
| **StatusBar** | `gui/components/status_bar.py` | Bottom status line. |
| **PositionCard** | `gui/components/positions_frame.py` | Single open position card. |
| **ConfigPanel** | `gui/components/config_panel.py` | API keys (masked), settings, recovery panel. |

## Database tab (refactored)

**DatabasePanel** (`gui/components/database_panel.py`) is a container that composes sections. Layout and constants come from **DatabasePanelConfig** (`gui/config/database_panel_config.py`).

### Sections (left panel)

- **DataViewerSection**: Table selector (Orders, Signals, Martingale Chains, Audit Log), pagination, text view. Uses `DatabasePanelConfig.DEFAULT_PAGE_SIZE`, `TEXTBOX_FONT`, `TITLE_FONT`, `DATA_VIEWER_HEIGHT`.
- **OrdersSection**: Order testing and display.
- **SignalsSection**: Signal testing.
- **MartingaleSection**: Martingale chain testing.
- **RecoverySection**: Gradual recovery testing, uses data viewer.

### Sections (right panel)

- **StatsSection**: Database statistics.
- **ActionsSection**: Reconcile with Binance, cleanup old records. Uses `DatabasePanelConfig.DEFAULT_RECONCILE_HOURS`, `DEFAULT_DAYS_TO_KEEP`, `MAX_RECONCILE_ERRORS_SHOWN`.
- **LogsSection**: Activity log.

### Config constants (`DatabasePanelConfig`)

- Pagination: `DEFAULT_PAGE_SIZE`, `INITIAL_PAGE`.
- Reconciliation: `DEFAULT_RECONCILE_HOURS`, `MAX_RECONCILE_ERRORS_SHOWN`.
- Cleanup: `DEFAULT_DAYS_TO_KEEP`.
- Fonts: `TITLE_FONT`, `TEXTBOX_FONT`, `HEADER_FONT`.
- Layout: `LEFT_PANEL_WEIGHT`, `RIGHT_PANEL_WEIGHT`, `PADX_*`, `PADY_*`.
- Data viewer: `DATA_VIEWER_HEIGHT`, `AVAILABLE_TABLES`, `TABLE_*`.

## Entry point

- `run_gui.py`: Sets `sys.path`, then `from modules.auto_trade.gui.main_window import AutoTradeDashboard`.
