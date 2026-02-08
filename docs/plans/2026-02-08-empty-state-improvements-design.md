# Empty State Improvements – Design

**Date**: 2026-02-08  
**Topic**: GUI empty states (icon + message/hint; CTA on selected screens)  
**Status**: Validated – ready for implementation

---

## 1. Scope and goals

**Scope**  
Improve all user-facing empty states in the auto_trade GUI:

- **Positions** (main window): when there are no open positions.
- **Signals** (main window, Live Signals table): when there are no signals.
- **Recovery** (recovery panel): when there is no active recovery.
- **Data Viewer** (Database tab): when the selected table (Orders, Signals, Martingale Chains, Audit Log) has no rows.

Database tab sections (Orders, Signals, Martingale, Recovery, Stats) do not have their own list widgets; they write results into the Data Viewer or log. Their “empty” case is therefore covered by the Data Viewer empty state when the user selects the corresponding table.

**Goals**  
- Replace plain “No data” / “No open positions” text with a consistent pattern: **icon + short message + optional hint**.  
- On three screens, add a **single primary CTA**: Positions → “Open a trade”, Signals → “Run scanner” (or equivalent), Recovery → “Start recovery”.  
- On the rest (Data Viewer and any future list without a clear action), use **icon + message/hint only**, no button.  
- Keep implementation small: one reusable component, no illustrations (emoji/unicode icon only), no new assets.

**Out of scope**  
- Empty states for log/output text (e.g. “No recovery sessions found” in Data Viewer text); those stay as improved copy only if we touch them.  
- Full illustrations or custom images; we use a single configurable icon (emoji) per context.

---

## 2. Component API and placement

**Component**: Single reusable widget `EmptyState(ctk.CTkFrame)` in a new file `modules/auto_trade/gui/components/empty_state.py`.

**API**  
- `parent`: container frame (e.g. `scroll_frame`, or the Data Viewer content area).  
- `icon`: str, optional (default e.g. `"📭"`) – emoji/unicode shown above the message.  
- `message`: str – main line (e.g. "No open positions").  
- `hint`: str, optional – subtitle/second line (e.g. "Open a trade or wait for a signal").  
- `action_text`: str, optional – label of the primary button.  
- `action_callback`: callable, optional – called when the button is clicked. If either `action_text` or `action_callback` is missing, no button is shown.

Layout: icon (large) → message → hint (if set) → button (if CTA provided). All vertically packed, transparent background so it matches the host panel.

**Placement per screen**  
- **Positions** (`positions_frame.py`): When `update_positions([])` is called, clear `scroll_frame` and pack an `EmptyState(icon="📭", message="No open positions", hint="Open a trade or wait for a signal.", action_text="Open a trade", action_callback=...) in the same scroll frame. Parent must have access to a callback that opens the trade form (e.g. from main window or existing `on_action_callback` wiring).  
- **Signals** (`signals_frame.py`): When the signals table has 0 rows, hide the table (or show a single “empty” row is not used); instead, in the same table_frame (or a wrapper that switches content), show `EmptyState(icon="📡", message="No signals yet", hint="Run the scanner to get live signals.", action_text="Run scanner", action_callback=...) . Callback should start/trigger the scanner (caller passes it in or frame gets it from parent).  
- **Recovery** (`recovery_panel.py`): Where the status currently shows “No active recovery”, replace or wrap that area with `EmptyState(icon="🔄", message="No active recovery", hint="Start a recovery session to begin.", action_text="Start recovery", action_callback=...) so the main action is visible.  
- **Data Viewer** (`data_viewer_section.py`): When `_load_table()` gets 0 rows, do not write "No data found in {table_name}" into the textbox. Instead, hide the textbox and show an `EmptyState` in the same frame (e.g. container that holds either textbox or EmptyState). Use `EmptyState(icon="📂", message="No data in this table", hint="Try another table or run queries that insert data.", action_text=None)` — no CTA.

**Import**: Each of the four call sites imports `EmptyState` from `modules.auto_trade.gui.components.empty_state` (or via `gui.components.empty_state` depending on project root). Use the same font/size as the rest of the app (e.g. existing Arial 14 / Roboto for message, smaller for hint).

---

## 3. Data flow and wiring

**Callbacks – where they come from**

- **Positions**  
  `PositionsFrame` already receives `on_action_callback` from `layout.py` (used for position actions). For the empty-state CTA “Open a trade”, use a **separate** callback that switches to the Trading tab so the user can use the trade form. In `layout.py`, when building the Dashboard right panel, pass e.g. `on_open_trade_callback=lambda: self.parent.tabview.set("Trading")` into `PositionsFrame`. The frame will use this for `EmptyState`’s `action_callback`; if not provided, show EmptyState without a button.

- **Signals**  
  `SignalsFrame` is currently created without a callback. Add an optional `on_run_scanner_callback`. In `layout.py`, pass `on_run_scanner_callback=lambda: (self.parent.tabview.set("Scanner"), self.parent.on_scan_toggle("manual"))` so the empty-state button switches to the Scanner tab and triggers a manual scan. When `on_run_scanner_callback` is None, show EmptyState without a button.

- **Recovery**  
  The recovery panel already has `_on_start_recovery`. In the status area (where “No active recovery” is shown), when there is no active recovery, show `EmptyState` with `action_callback=self._on_start_recovery`. No new callback from outside; the panel owns the action.

- **Data Viewer**  
  No CTA. When `_load_table()` / `refresh()` gets zero rows, do not insert text into the textbox. Instead, show EmptyState and hide the textbox (see below). No callback wiring.

**Data Viewer: show/hide textbox vs EmptyState**

- In `_create_ui()`, create a **content frame** that will hold either the textbox or the EmptyState (not both visible at once). Structure: `frame` (existing) → `header_frame` (unchanged) → `content_frame` (new) → initially only `self.data_viewer` (CTkTextbox) is packed. Pagination stays below `content_frame`.
- In `refresh()`: when `len(data) == 0`, call `self.data_viewer.pack_forget()`, then create and pack `EmptyState(parent=self.content_frame, ...)`. Keep a reference (e.g. `self._empty_state_widget`) so it can be destroyed later. When `len(data) > 0`, destroy `self._empty_state_widget` if present, then `self.data_viewer.pack(...)` again so the textbox is visible and can be updated with the table output. Pagination controls remain visible in both cases; when empty, “Page 1/1” is acceptable.

**Pack/destroy order**

- **Positions / Signals / Recovery**  
  When switching from “has items” to “empty”: destroy or forget the list/cards/table widgets, then pack the EmptyState in the same container. When switching from “empty” to “has items”: destroy the EmptyState widget (if stored in `self._empty_state` or similar), then rebuild the list/table as today. Use a single container (e.g. `scroll_frame` or `table_frame`) so only one of “content” or “empty state” is visible at a time.

- **Data Viewer**  
  As above: empty → pack EmptyState and forget textbox; has data → destroy EmptyState and pack textbox again. Avoid packing both; toggling visibility (pack/pack_forget) or destroy + pack keeps layout simple.

**Summary**

| Screen     | CTA callback source                                      | Empty-state visibility toggle                    |
|-----------|-----------------------------------------------------------|---------------------------------------------------|
| Positions | `layout` → `on_open_trade_callback` → switch to Trading   | Clear scroll_frame; pack EmptyState or cards      |
| Signals   | `layout` → `on_run_scanner_callback` → Scanner + manual   | Hide table; show EmptyState (or reverse)         |
| Recovery  | Panel’s `_on_start_recovery`                             | Replace status label area with EmptyState        |
| Data Viewer | —                                                      | content_frame: textbox pack_forget vs EmptyState |

---

## 4. Error handling and testing

**Error handling**

- **EmptyState component**  
  If `action_callback` is provided and the user clicks the button, call the callback inside a try/except; on exception, log and optionally show a short toast or status message so the app does not crash. Do not let the callback raise uncaught into the GUI loop.

- **Data Viewer**  
  Existing `refresh()` already has a try/except that calls `log_callback(..., "ERROR")` on failure. When adding the empty-state branch (0 rows), keep that same try/except; if the query fails, leave the last successful content or a generic error message in the textbox and do not show EmptyState for the error case. EmptyState is only for “table is empty”, not “query failed”.

- **Positions / Signals / Recovery**  
  No change to existing error handling. If the callback (e.g. open trade, run scanner, start recovery) fails, existing handlers in main_window or recovery_panel apply. EmptyState only triggers the callback; it does not need to interpret errors.

**Testing**

- **Unit tests**  
  - `EmptyState`: with and without `hint`, with and without `action_text`/`action_callback`; assert widget count (no button when callback is None), and that clicking the button invokes the callback (mock callback).  
  - Optional: test that `EmptyState` can be packed and destroyed without error (smoke test).

- **Integration / GUI tests**  
  - Positions: with 0 positions, EmptyState is shown; after CTA or after positions arrive, EmptyState is removed and list is shown.  
  - Data Viewer: select a table with 0 rows → EmptyState visible; select a table with rows or refresh with data → textbox visible, EmptyState gone.  
  - Recovery: when no active recovery, EmptyState (or status area) shows CTA; after start recovery, status updates and EmptyState can be replaced by normal status.  

  Prefer pytest + mock for callbacks; if the project has GUI tests (e.g. that run the app or use a test window), add one to two smoke tests for empty-state visibility. No requirement for full E2E on all four screens.

**Acceptance**

- All four screens show icon + message (+ hint where defined) when the corresponding list/data is empty.  
- Positions, Signals, and Recovery show the CTA button and the button triggers the agreed action (tab switch + optional scan, or start recovery).  
- Data Viewer shows no CTA and does not show raw “No data found in …” in the textbox when using EmptyState.  
- No regressions: existing behaviour when data exists is unchanged.

---

**Design complete.** Ready for implementation; update REFACTORING_RECOMMENDATIONS.md §4 when done.
