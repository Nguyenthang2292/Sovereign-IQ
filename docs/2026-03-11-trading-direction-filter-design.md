# Trading Direction Filter - Design Document

**Status:** Approved
**Date:** 2026-03-11
**Author:** Design Session with User

---

## Overview

Add a "Trading Direction" filter control to the Auto-Trade GUI that allows users to restrict trading to:
- **Long Only** - Only process LONG signals
- **Short Only** - Only process SHORT signals
- **Both Directions** - Process both LONG and SHORT signals (default)

This feature provides tactical control over trading strategy without code modification, useful for adapting to market conditions or implementing directional bias.

---

## Requirements

### Functional Requirements

1. **GUI Control**
   - Radio button group with 3 mutually exclusive options
   - Located in Auto-Trade Control panel
   - Visual section with emoji icon ("📊 Trading Direction")
   - Default selection: "Both Directions"

2. **Signal Filtering**
   - Filter signals early in pipeline (after ATC scan, before XGBoost/Gemini)
   - Discard unwanted direction signals completely
   - Log filtered signal count for observability

3. **Settings Persistence**
   - Include in settings save/load system
   - Export/import preserves trading direction
   - Validate and sanitize on load (fallback to "BOTH")

4. **Runtime Behavior**
   - Allow changes while auto-trading is running
   - Changes take effect on next scan cycle (no restart required)
   - Does NOT affect existing open positions

### Non-Functional Requirements

- **Performance:** Signal filtering adds <10ms overhead per scan
- **Testing:** Unit tests for filter logic, integration tests for pipeline
- **Maintainability:** Follow existing code patterns in signal_pipeline.py
- **Reliability:** Handle edge cases gracefully (empty lists, invalid settings)

---

## Architecture

### Component Changes

**1. GUI Layer** (`modules/auto_trade/gui/components/auto_trade_control.py`)
- Add framed section "📊 Trading Direction"
- Three radio buttons with icons: ⬆️ Long Only, ⬇️ Short Only, ↕️ Both Directions
- Store selection in `self.trading_direction_var = ctk.StringVar(value="BOTH")`
- Values: `"LONG_ONLY"`, `"SHORT_ONLY"`, `"BOTH"`

**2. Settings Layer** (`modules/auto_trade/gui/services/settings_manager.py`)
- Add `trading_direction` field to settings dict
- Include in `get_settings()` and `load_settings()` methods
- Validate on load, fallback to `"BOTH"` if invalid

**3. Pipeline Layer** (`modules/auto_trade/core/signal_pipeline.py`)
- Add `allowed_directions: List[str]` to `PipelineConfig` TypedDict
- Add `_filter_by_direction()` private method
- Call filter after ATC scan, before XGBoost filter

**4. Main Window Bridge** (`modules/auto_trade/gui/main_window/auto_trade.py`)
- Read `trading_direction` from settings
- Convert to list format:
  - `"LONG_ONLY"` → `["LONG"]`
  - `"SHORT_ONLY"` → `["SHORT"]`
  - `"BOTH"` → `["LONG", "SHORT"]`
- Pass to pipeline config

---

## Data Flow

### Configuration Flow

```
User selects radio button in GUI
    ↓
Auto-Trade Control: trading_direction_var = "LONG_ONLY" | "SHORT_ONLY" | "BOTH"
    ↓
Settings Manager: Persists to settings.json
    ↓
Main Window: Reads setting when starting auto-trade
    ↓
Convert to list: "LONG_ONLY" → ["LONG"]
    ↓
SignalPipeline(config={"allowed_directions": ["LONG"], ...})
```

### Signal Filtering Flow

```
SignalPipeline.run_pipeline()
    ↓
Step 1: Refresh symbols ✓
    ↓
Step 2: ATC Scanner generates signals ✓
    ↓
Step 3: 🆕 Filter by direction
    signals = _filter_by_direction(atc_signals, allowed_directions)
    ↓
Step 4: XGBoost filter ✓ (only processes allowed signals)
    ↓
Step 5: Gemini analysis ✓ (only processes allowed signals)
    ↓
Step 6: Signal selection ✓
```

---

## Implementation Details

### 1. SignalPipeline Filter Method

```python
# modules/auto_trade/core/signal_pipeline.py

class PipelineConfig(TypedDict, total=False):
    max_symbols_to_scan: int
    monitoring_enabled: bool
    max_ai_candidates: int
    xgboost_mode: str
    enable_gann_square: bool
    allowed_directions: List[str]  # 🆕 New field


class SignalPipeline:
    def _filter_by_direction(
        self,
        signals: List[SignalResult],
        allowed_directions: List[str]
    ) -> List[SignalResult]:
        """
        Filter signals by allowed trading directions.

        Args:
            signals: List of signals from ATC scanner
            allowed_directions: List of allowed directions ["LONG", "SHORT"]

        Returns:
            Filtered list of signals matching allowed directions
        """
        if not allowed_directions or set(allowed_directions) == {"LONG", "SHORT"}:
            # No filtering needed - both directions allowed
            return signals

        filtered = [s for s in signals if s.signal_type in allowed_directions]

        if len(filtered) < len(signals):
            removed_count = len(signals) - len(filtered)
            log_info(
                f"Direction filter: Removed {removed_count} signal(s). "
                f"Allowed: {allowed_directions}, Kept: {len(filtered)}"
            )

        return filtered

    def run_pipeline(self) -> Optional[FinalSignal]:
        # ... existing code ...

        # Step 2: ATC scan
        atc_signals = self._atc_scan(symbols)

        # Step 3: 🆕 Filter by direction
        allowed_dirs = self.config.get("allowed_directions", ["LONG", "SHORT"])
        atc_signals = self._filter_by_direction(atc_signals, allowed_dirs)

        # Step 4: XGBoost filter (continues as normal)
        filtered_signals = self._xgboost_filter(atc_signals)

        # ... rest of pipeline ...
```

### 2. GUI Implementation

```python
# modules/auto_trade/gui/components/auto_trade_control.py

class AutoTradeControl(ctk.CTkFrame):
    def __init__(self, parent, ...):
        super().__init__(parent)

        # ... existing code ...

        # Trading Direction Section
        direction_section = ctk.CTkFrame(
            self,
            fg_color=Colors.get_card_bg(),
            corner_radius=8
        )
        direction_section.pack(fill="x", padx=10, pady=(10, 5))

        direction_inner = ctk.CTkFrame(
            direction_section,
            fg_color=Colors.TRANSPARENT
        )
        direction_inner.pack(fill="x", padx=15, pady=12)

        # Section title
        ctk.CTkLabel(
            direction_inner,
            text="📊 Trading Direction",
            font=Fonts.H3,
            anchor="w"
        ).pack(anchor="w", pady=(0, 8))

        # Radio button variable
        self.trading_direction_var = ctk.StringVar(value="BOTH")

        # Radio buttons
        radio_container = ctk.CTkFrame(
            direction_inner,
            fg_color=Colors.TRANSPARENT
        )
        radio_container.pack(fill="x")

        ctk.CTkRadioButton(
            radio_container,
            text="⬆️ Long Only",
            variable=self.trading_direction_var,
            value="LONG_ONLY"
        ).pack(anchor="w", pady=2)

        ctk.CTkRadioButton(
            radio_container,
            text="⬇️ Short Only",
            variable=self.trading_direction_var,
            value="SHORT_ONLY"
        ).pack(anchor="w", pady=2)

        ctk.CTkRadioButton(
            radio_container,
            text="↕️ Both Directions",
            variable=self.trading_direction_var,
            value="BOTH"
        ).pack(anchor="w", pady=2)
```

### 3. Settings Manager Updates

```python
# modules/auto_trade/gui/services/settings_manager.py

VALID_DIRECTIONS = {"LONG_ONLY", "SHORT_ONLY", "BOTH"}

class SettingsManager:
    def get_settings(self) -> Dict:
        """Get current settings including trading direction."""
        settings = {
            # ... existing settings ...
            "trading_direction": self.auto_trade_control.trading_direction_var.get(),
        }
        return settings

    def load_settings(self, settings: Dict):
        """Load settings including trading direction."""
        # ... existing loading logic ...

        # Validate and load trading direction
        direction = settings.get("trading_direction", "BOTH")
        if direction not in VALID_DIRECTIONS:
            log_warn(f"Invalid trading_direction '{direction}', defaulting to BOTH")
            direction = "BOTH"

        self.auto_trade_control.trading_direction_var.set(direction)
```

### 4. Main Window Bridge

```python
# modules/auto_trade/gui/main_window/auto_trade.py

def _start_auto_trade(self):
    """Start auto-trading with direction filter."""
    # ... existing code ...

    # Read trading direction from config
    direction_setting = self.config_panel.get_settings().get("trading_direction", "BOTH")

    # Convert to list format for pipeline
    if direction_setting == "LONG_ONLY":
        allowed_directions = ["LONG"]
    elif direction_setting == "SHORT_ONLY":
        allowed_directions = ["SHORT"]
    else:  # "BOTH"
        allowed_directions = ["LONG", "SHORT"]

    # Pass to pipeline config
    pipeline_config = {
        "allowed_directions": allowed_directions,
        # ... other config ...
    }

    self.pipeline = SignalPipeline(config=pipeline_config, ...)

    # ... rest of startup ...
```

---

## Edge Cases & Error Handling

### Edge Case 1: Empty Signal List After Filtering
- **Scenario:** All ATC signals filtered out (e.g., only SHORT signals but "LONG_ONLY" selected)
- **Handling:** Pipeline continues with empty list, logs info, returns None
- **User sees:** "No signals found" (existing behavior)

### Edge Case 2: Invalid Direction in Settings
- **Scenario:** Corrupted settings file has `trading_direction: "INVALID"`
- **Handling:** Settings manager validates on load, falls back to `"BOTH"`
- **Logged:** Warning message about invalid value

### Edge Case 3: Changing Direction Mid-Scan
- **Scenario:** User changes from "BOTH" to "LONG_ONLY" while pipeline running XGBoost
- **Handling:** Current scan completes with old setting, next scan uses new setting
- **Safe:** Each `run_pipeline()` call reads config at start, no shared mutable state

### Edge Case 4: Existing Open Positions
- **Scenario:** User has open SHORT position, switches to "LONG_ONLY"
- **Handling:** Filter only affects NEW signals, existing positions unaffected
- **Intentional:** Position management is separate from signal generation

### Validation

```python
def validate_trading_direction(value: str) -> str:
    """Validate and sanitize trading direction setting."""
    if value not in VALID_DIRECTIONS:
        return "BOTH"  # Safe default
    return value
```

---

## Testing Strategy

### Unit Tests

**Signal Filtering Logic:**
```python
# tests/test_signal_pipeline.py

def test_filter_by_direction_long_only():
    """Test filtering keeps only LONG signals."""
    pipeline = SignalPipeline(...)
    signals = [
        SignalResult(symbol="BTC/USDT", signal_type="LONG", ...),
        SignalResult(symbol="ETH/USDT", signal_type="SHORT", ...),
        SignalResult(symbol="BNB/USDT", signal_type="LONG", ...),
    ]

    filtered = pipeline._filter_by_direction(signals, ["LONG"])

    assert len(filtered) == 2
    assert all(s.signal_type == "LONG" for s in filtered)

def test_filter_by_direction_both():
    """Test no filtering when both directions allowed."""
    pipeline = SignalPipeline(...)
    signals = [...]  # Mix of LONG and SHORT

    filtered = pipeline._filter_by_direction(signals, ["LONG", "SHORT"])

    assert len(filtered) == len(signals)

def test_filter_by_direction_empty_list():
    """Test filtering handles empty signal list."""
    pipeline = SignalPipeline(...)
    filtered = pipeline._filter_by_direction([], ["LONG"])
    assert filtered == []
```

**Settings Persistence:**
```python
# tests/test_settings_manager.py

def test_trading_direction_saved_and_loaded():
    """Test trading direction persists across save/load."""
    manager = SettingsManager()
    manager.auto_trade_control.trading_direction_var.set("LONG_ONLY")

    settings = manager.get_settings()
    assert settings["trading_direction"] == "LONG_ONLY"

    manager.load_settings(settings)
    assert manager.auto_trade_control.trading_direction_var.get() == "LONG_ONLY"

def test_invalid_direction_defaults_to_both():
    """Test invalid direction value falls back to BOTH."""
    manager = SettingsManager()
    invalid_settings = {"trading_direction": "INVALID_VALUE"}

    manager.load_settings(invalid_settings)

    assert manager.auto_trade_control.trading_direction_var.get() == "BOTH"
```

**Integration Tests:**
```python
def test_pipeline_respects_direction_filter():
    """Test full pipeline filters signals by direction."""
    pipeline = SignalPipeline(
        config={"allowed_directions": ["LONG"], ...}
    )

    with patch.object(pipeline.atc_scanner, 'scan_symbols') as mock_scan:
        mock_scan.return_value = [
            SignalResult(..., signal_type="LONG"),
            SignalResult(..., signal_type="SHORT"),
        ]

        result = pipeline.run_pipeline()

        # XGBoost should only see LONG signal
        filtered_input = pipeline.xgboost_filter.filter_signals.call_args[0][0]
        assert len(filtered_input) == 1
        assert filtered_input[0].signal_type == "LONG"
```

### Manual Testing Checklist

- [ ] Radio buttons are mutually exclusive
- [ ] Default selection is "Both Directions"
- [ ] Setting persists after GUI restart
- [ ] Changing direction while running takes effect on next scan
- [ ] Log messages show filtered signal count
- [ ] Settings export/import includes direction
- [ ] Visual style matches existing UI (card background, fonts, spacing)
- [ ] Section appears in correct location (Auto-Trade Control panel)

---

## Decision Log

### Decision 1: Signal Filtering Strategy
- **Chosen:** Filter signals completely (discard early in pipeline)
- **Alternatives:** Block at execution, Flip direction
- **Rationale:** Saves compute (no XGBoost/Gemini on filtered signals), clear behavior

### Decision 2: Control Type
- **Chosen:** Radio buttons with 3 options
- **Alternatives:** Checkboxes (mutually exclusive or independent)
- **Rationale:** Clearer UI, prevents invalid state, more intuitive

### Decision 3: GUI Placement
- **Chosen:** Auto-Trade Control panel
- **Alternatives:** Risk Settings tab, Signal Filters tab, Scanner Control
- **Rationale:** High visibility, easy access, user preference

### Decision 4: Default State
- **Chosen:** "Both Directions"
- **Alternatives:** "Long Only", Remember last selection
- **Rationale:** Safest default, matches existing behavior

### Decision 5: Persistence
- **Chosen:** Persist in settings
- **Alternatives:** Session-only
- **Rationale:** Consistency, user convenience, standard behavior

### Decision 6: Runtime Editing
- **Chosen:** Allow changes anytime
- **Alternatives:** Require STOP first, Show confirmation
- **Rationale:** Tactical flexibility, immediate effect, user preference

### Decision 7: Visual Style
- **Chosen:** Section with icon/emoji
- **Alternatives:** Simple label, Horizontal layout, Match existing
- **Rationale:** Visual prominence, consistency with other sections

### Decision 8: Architecture
- **Chosen:** Pipeline-level filter (after ATC, before XGBoost)
- **Alternatives:** Signal Selector filter, ATC Scanner config
- **Rationale:** Right abstraction, early filtering, clean separation, testable

---

## Implementation Phases

### Phase 1: Pipeline Filter Logic
1. Add `allowed_directions` to `PipelineConfig`
2. Implement `_filter_by_direction()` method
3. Integrate filter into `run_pipeline()` after ATC scan
4. Add unit tests for filter logic

### Phase 2: GUI Controls
1. Add radio button section to Auto-Trade Control
2. Wire up `trading_direction_var`
3. Test visual appearance and behavior

### Phase 3: Settings Integration
1. Update `get_settings()` and `load_settings()`
2. Add validation for trading direction
3. Test persistence across GUI restarts

### Phase 4: Main Window Bridge
1. Read direction from settings
2. Convert to list format
3. Pass to pipeline config
4. Test end-to-end flow

### Phase 5: Testing & Documentation
1. Add integration tests
2. Manual testing checklist
3. Update user documentation (if exists)

---

## Files to Modify

```
modules/auto_trade/core/signal_pipeline.py       # Filter logic + PipelineConfig
modules/auto_trade/gui/components/auto_trade_control.py  # GUI controls
modules/auto_trade/gui/services/settings_manager.py      # Persistence
modules/auto_trade/gui/main_window/auto_trade.py         # Bridge layer
tests/test_signal_pipeline.py                            # Unit tests
tests/test_settings_manager.py                           # Settings tests
```

---

## Success Criteria

✅ User can select Long Only, Short Only, or Both Directions
✅ Selection persists across GUI restarts
✅ Filter removes unwanted signals before XGBoost/Gemini
✅ Changes take effect on next scan (no restart needed)
✅ All tests pass (unit + integration)
✅ No performance degradation (<10ms overhead)
✅ Existing positions unaffected by direction changes
✅ Settings export/import includes trading direction

---

## Assumptions

- Signal filtering overhead will be negligible (<10ms per scan)
- The filter will be implemented in `SignalPipeline.run_pipeline()` after ATC scan
- Settings store direction as string: `"LONG_ONLY"`, `"SHORT_ONLY"`, or `"BOTH"`
- UI uses section with emoji ("📊 Trading Direction")
- Existing settings save/load infrastructure can accommodate new field
- No confirmation dialog needed when changing direction while running
- ATC scanner continues to generate all signal types (filtering happens in pipeline)

---

## Future Enhancements (Out of Scope)

- Advanced filters (e.g., "Long above resistance, Short below support")
- Per-symbol direction preferences
- Time-based direction switching (e.g., "Long during US hours, Short during Asia")
- Direction analytics (track win rate by direction)
- Integration with sentiment analysis (auto-switch based on market sentiment)

---

## Notes

- This feature does NOT modify signal generation logic (ATC/XGBoost/Gemini unchanged)
- Existing open positions are NOT affected by direction filter changes
- Filter applies to NEW signals only, each scan cycle
- Log messages provide observability into filtered signal counts
- Settings validation ensures robustness against corrupted config files
