# Integrate GannSquare into auto_trade GUI pipeline

## Goal
Add `gemini_gann_square` as an optional final filter layer after `SignalSelector`, iterating ranked candidates top-down until one passes Gann analysis (or skip cycle if all rejected).

---

## Tasks

- [x] **T1** — Add `rank_signals()` to `SignalSelector`
  - File: `modules/auto_trade/core/signal_selector.py`
  - Extract the `candidates` building + sorting logic from `select_best_signal()` into a new public method `rank_signals(xgboost_signals, gemini_signals) -> List[FinalSignal]`
  - Refactor `select_best_signal()` to delegate: `ranked = self.rank_signals(...); return ranked[0] if ranked else None`
  - Verify: existing unit tests still pass, `rank_signals()` returns full sorted list

- [x] **T2** — Create `GannSquareFilter` class
  - File: `modules/auto_trade/core/gann_square_filter.py` (new)
  - `__init__(self, timeframe, limit=200, charts_dir="charts", gemini_api_key=None)`
  - `run(self, ranked_signals: List[FinalSignal]) -> GannAnalysisResult | None`
    - For each signal (high → low score):
      1. Delete all `*.png` in `charts_dir` (keep subdirs intact)
      2. Call `GannSignalEngine().analyze(symbol, timeframe, limit)`
      3. Return immediately if `result.is_tradeable()`
      4. On `SKIP`: log `f"GannSquare: {symbol} SKIP — {result.reasoning[:80]}"` + continue
      5. On exception: log error + continue
    - If all exhausted: log warning, return `None`
  - Verify: instantiate manually, call `run([])` returns `None` without crash

- [x] **T3** — Add `_gann_result_to_final_signal()` converter in `SignalPipeline`
  - File: `modules/auto_trade/core/signal_pipeline.py`
  - Private method `_gann_result_to_final_signal(self, gann: GannAnalysisResult, original: FinalSignal) -> FinalSignal`
    - Uses `gann.entry_price`, `gann.stop_loss`, `gann.take_profit_1` for price levels
    - Copies `original.sources` and adds `gann_confidence: gann.confidence_pct / 100`
    - Re-raises `ValueError` (invalid prices) → caught in Step 7 as rejection
  - Verify: unit test with mock `GannAnalysisResult` → valid `FinalSignal` returned

- [x] **T4** — Wire `GannSquareFilter` into `SignalPipeline`
  - File: `modules/auto_trade/core/signal_pipeline.py`
  - Add `gann_square_filter: GannSquareFilter | None = None` to `__init__` signature
  - In `PipelineConfig` TypedDict add: `enable_gann_square: bool`
  - Replace Step 6 block:
    ```
    Step 6: rank_signals() → List[FinalSignal]
    Step 7 (optional): GannSquareFilter.run(ranked) → GannAnalysisResult | None
      - if None → log_warn + return None
      - if result → _gann_result_to_final_signal(result, ranked[best_idx])
    else: final_signal = ranked[0] if ranked else None
    ```
  - Verify: pipeline runs with `gann_square_filter=None` → unchanged behaviour

- [x] **T5** — Add GUI controls to `ScannerControl`
  - File: `modules/auto_trade/gui/components/scanner_control.py`
  - Inside `_create_configuration()` → **Model Filters** group, after the ATC slider block, add:
    1. `CTkCheckBox` — "Enable Gann Square Filter" → `self.enable_gann_square_var` (BooleanVar, default `False`)
    2. Sub-frame (only shown when checkbox ON, toggled via `.pack()`/`.pack_forget()`):
       - `CTkComboBox` — "Gann TF:" values `["1h","2h","4h","6h","8h","12h","1d"]` → `self.gann_tf_var` (default `"4h"`)
       - `CTkEntry` — "Candles:" → `self.gann_candle_limit_entry` (default `"200"`)
  - `get_config()` must emit keys: `enable_gann_square`, `gann_timeframe`, `gann_candle_limit`
  - `load_config()` must restore these keys from saved settings
  - Verify: toggle checkbox → sub-frame appears/disappears; saved settings round-trip correctly

- [x] **T6** — Construct and inject `GannSquareFilter` in `ScannerManager`
  - File: `modules/auto_trade/gui/main_window/scanner.py`
  - In `_initialize_pipeline()`, after building `signal_selector`, add:
    ```python
    gann_filter = None
    if scanner_config.get("enable_gann_square", False):
        from modules.auto_trade.core.gann_square_filter import GannSquareFilter
        gann_filter = GannSquareFilter(
            timeframe=scanner_config.get("gann_timeframe", "4h"),
            limit=int(scanner_config.get("gann_candle_limit", 200)),
        )
        log_info(f"GannSquareFilter ready (tf={gann_filter.timeframe}, limit={gann_filter.limit})")
    ```
  - Pass `gann_square_filter=gann_filter` to `SignalPipeline(...)` constructor
  - Verify: with toggle OFF → `gann_filter is None`; with toggle ON → `GannSquareFilter` instantiated

- [x] **T7** — End-to-end smoke test
  - Run GUI in DRY_RUN mode, enable Gann Square, trigger Manual Scan
  - Confirm in logs: "GannSquareFilter ready", charts dir cleaned before each call, at least one SKIP or PASS logged
  - Confirm: if all SKIP → "GannSquare rejected all candidates" warning in log, no trade executed
  - Confirm: `charts/` folder contains exactly 1 PNG after cycle ends (last analyzed chart)

---

## Done When
- [x] `scanner_control.py` shows Gann toggle + TF/candle sub-options, persists to settings
- [x] `GannSquareFilter` iterates candidates in order, cleans `charts/` before each, stops on first tradeable
- [x] Pipeline returns `None` (not crash) when all candidates are SKIP
- [x] All existing pipeline behaviour unchanged when `enable_gann_square = False`

---

## Key File Map

| File | Change |
|------|--------|
| `modules/auto_trade/core/signal_selector.py` | Add `rank_signals()`, refactor `select_best_signal()` |
| `modules/auto_trade/core/gann_square_filter.py` | **New file** — `GannSquareFilter` |
| `modules/auto_trade/core/signal_pipeline.py` | Inject filter, wiring Steps 6-7, `_gann_result_to_final_signal()` |
| `modules/auto_trade/gui/components/scanner_control.py` | Gann toggle + sub-controls |
| `modules/auto_trade/gui/main_window/scanner.py` | Construct + inject `GannSquareFilter` |

## Notes
- `GannSignalEngine.analyze()` fetches its own OHLCV data internally — no df injection needed
- `charts/` cleanup: only `*.png` in root of charts dir, not recursive (preserve any subdirs)
- `gann_square_filter` is optional arg in `SignalPipeline.__init__` → zero breaking changes
- `TP` mapped from `gann.take_profit_1` (conservative first target)
