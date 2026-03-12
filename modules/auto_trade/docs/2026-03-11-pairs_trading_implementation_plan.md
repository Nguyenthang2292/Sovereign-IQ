# Pairs Trading Integration - Implementation Plan

**Project:** Sovereign-IQ Auto Trade
**Module:** Pairs Trading Integration
**Estimated Duration:** 5-7 days
**Complexity:** Medium-High

---

## Implementation Phases

### Phase 1: Core Infrastructure (Days 1-2)

#### 1.1 CorrelationScanner

**File:** `modules/auto_trade/execution/correlation_scanner.py`

**Tasks:**
- [x] Create `CorrelationScanner` class
- [x] Implement `calculate_correlation()` - reuse `modules/pairs_trading` metrics
- [x] Implement `calculate_hedge_ratio()` with OLS and Kalman methods
- [x] Implement `scan_hedge_candidates()` with filtering and ranking
- [x] Add correlation cache with TTL-based refresh
- [x] Add unit tests (`tests/test_correlation_scanner.py`)

**Dependencies:**
- `modules/pairs_trading` (existing)
- `modules/common/core/data_fetcher.py` (existing)

**Acceptance Criteria:**
- Correlation scanner returns ranked candidates
- Cache reduces redundant API calls
- All unit tests pass ✅ (12/12 tests passing)

---

#### 1.2 PairsCoordinator

**File:** `modules/auto_trade/execution/pairs_coordinator.py`

**Tasks:**
- [x] Create `PairsCoordinator` class
- [x] Implement `should_activate_pairs()` logic
- [x] Implement `determine_regime()` using ADX calculation
- [x] Implement `calculate_position_sizes()` for all 3 regimes:
  - [x] STAT_ARB mode (hedge ratio sizing)
  - [x] MOMENTUM mode (risk-parity sizing)
  - [x] BLENDED mode (50/50 blend)
- [x] Implement `determine_hedge_direction()` per-regime config
- [x] Implement `execute_pair_atomically()` with rollback
- [x] Add unit tests (`tests/test_pairs_coordinator.py`)

**Dependencies:**
- `correlation_scanner.py` (Phase 1.1)
- `modules/auto_trade/execution/order_builder.py` (existing)
- `modules/auto_trade/execution/order_executor.py` (existing)

**Acceptance Criteria:**
- Regime detection works correctly for ADX thresholds
- Position sizing calculations correct for all regimes
- Atomic execution + rollback tested
- All unit tests pass ✅ (18/18 tests passing)

---

### Phase 1: COMPLETED ✅

---

### Phase 2: Database Schema (Day 2)

#### 2.1 Extend Order Model

**File:** `modules/auto_trade/database/repository/dynamodb/orders.py`

**Tasks:**
- [ ] Add new fields to Order model:
  - `pair_id`, `pair_leg`, `pair_correlation`, `pair_hedge_ratio`
  - `pair_regime`, `pair_partner_symbol`, `pair_is_hedged`
- [ ] Create GSI: `pair_id-index`
- [ ] Create GSI: `pair_is_hedged-opened_at-index`
- [ ] Add helper methods:
  - `query_by_pair_id()`
  - `get_partner_leg()`
- [ ] Write migration script (if needed)

**Acceptance Criteria:**
- Orders can be queried by pair_id
- Both legs retrievable via GSI
- Backward compatible (existing orders work)

---

#### 2.2 Create PairMetrics Entity

**File:** `modules/auto_trade/database/repository/dynamodb/pair_metrics.py`

**Tasks:**
- [ ] Create `PairMetrics` model with all fields
- [ ] Create GSI: `status-created_at-index`
- [ ] Implement CRUD operations:
  - `create()`, `get()`, `update()`, `query_by_status()`
- [ ] Add unit tests

**Acceptance Criteria:**
- PairMetrics can be created and queried
- Active pairs retrievable via GSI
- All CRUD operations work

---

### Phase 3: Monitoring & Metrics (Day 3)

#### 3.1 PairsMetricsTracker

**File:** `modules/auto_trade/monitoring/pairs_metrics.py`

**Tasks:**
- [ ] Create `PairsMetricsTracker` class
- [ ] Implement `track_pair_entry()` - initialize tracking
- [ ] Implement `update_pair_metrics()` - real-time PnL updates
- [ ] Implement `calculate_hedge_efficiency()` formula
- [ ] Implement `calculate_pair_sharpe()` formula
- [ ] Implement `check_correlation_drift()` with alert logic
- [ ] Integrate with `EventSystem` for alerts
- [ ] Add background job (runs every 1 hour):
  - Recalculate correlations for active pairs
  - Emit drift alerts if threshold exceeded
- [ ] Add unit tests

**Acceptance Criteria:**
- Metrics calculated correctly
- Correlation drift alerts triggered
- Background job runs on schedule
- All unit tests pass

---

#### 3.2 WebSocket Monitoring Integration

**File:** `modules/auto_trade/execution/pairs_websocket_handler.py` (new)

**Tasks:**
- [ ] Create WebSocket handler for paired orders
- [ ] Listen for order update events
- [ ] Detect when one leg closes (TP or SL)
- [ ] Implement unified SL logic:
  - If either leg hits SL → close partner immediately
- [ ] Implement separate TP logic:
  - If one leg hits TP → keep partner open
- [ ] Update PairMetrics on leg close
- [ ] Add integration tests

**Acceptance Criteria:**
- Unified SL triggers partner close
- Separate TP allows independent wins
- PairMetrics updated on status changes

---

### Phase 4: GUI Integration (Days 4-5)

#### 4.1 Pairs Control Panel

**File:** `modules/auto_trade/gui/components/pairs_control.py`

**Tasks:**
- [ ] Create `PairsControl` class (CTkFrame)
- [ ] Implement status section:
  - Enabled/disabled indicator
  - Active pairs count
  - Last correlation scan timestamp
- [ ] Implement control section:
  - Enable/disable toggle
  - Refresh correlations button
- [ ] Implement active pairs table:
  - Scrollable frame with pair rows
  - Show: pair ID, symbols, correlation, PnL
- [ ] Implement alerts section:
  - Display correlation drift alerts
- [ ] Add update loop (refresh every 5 seconds)
- [ ] Add GUI tests

**Acceptance Criteria:**
- Panel displays correctly
- Toggle enables/disables pairs trading
- Active pairs table updates in real-time
- Alerts section shows drift warnings

---

#### 4.2 Config Panel Extension

**File:** `modules/auto_trade/gui/components/config_panel_parts/pairs_tab.py` (new)

**Tasks:**
- [ ] Create `create_pairs_trading_tab()` function
- [ ] Implement Correlation Parameters section:
  - Min correlation slider (0.50-0.90)
  - Lookback period entry
  - Timeframe dropdown
  - Refresh interval dropdown
- [ ] Implement ADX Regime Thresholds section:
  - ADX Low entry (default: 20)
  - ADX High entry (default: 30)
- [ ] Implement Hedge Direction section:
  - Stat-Arb mode dropdown (opposite/same)
  - Momentum mode dropdown (opposite/same)
  - Blended mode dropdown (opposite/same/correlation_based)
- [ ] Implement Advanced Settings section:
  - Drift alert threshold entry
  - Hedge leverage min/max entries
- [ ] Add settings persistence (save/load from `settings.json`)
- [ ] Add GUI tests

**File:** `modules/auto_trade/gui/components/config_panel.py` (modify)

**Tasks:**
- [ ] Add `self._create_pairs_trading_tab()` call in `__init__`
- [ ] Extend `get_settings()` to include pairs settings
- [ ] Extend `load_settings()` to populate pairs UI

**Acceptance Criteria:**
- Config tab displays with all sections
- Settings save/load correctly from settings.json
- All controls functional

---

#### 4.3 Visual Indicators

**File:** `modules/auto_trade/gui/components/signals_frame.py` (modify)

**Tasks:**
- [ ] Extend `_render_signal_row()`:
  - Check if signal has `pair_id`
  - Show 🔗 icon for hedged signals
  - Display hedge partner symbol

**File:** `modules/auto_trade/gui/components/positions_frame.py` (modify)

**Tasks:**
- [ ] Extend position display:
  - Group paired positions together
  - Show correlation and net PnL
  - Add visual grouping (border, background color)

**Acceptance Criteria:**
- Hedged signals visually distinct
- Paired positions grouped in positions frame

---

#### 4.4 Main Layout Integration

**File:** `modules/auto_trade/gui/main_window/layout.py` (modify)

**Tasks:**
- [ ] Add `PairsControl` panel to main layout
- [ ] Position alongside existing control panels
- [ ] Ensure responsive resizing

**Acceptance Criteria:**
- Pairs Control panel visible in main window
- Layout remains clean and usable

---

### Phase 5: Integration & Testing (Days 5-6)

#### 5.1 OrderExecutor Extension

**File:** `modules/auto_trade/execution/order_executor.py` (modify)

**Tasks:**
- [ ] Add `pairs_coordinator` parameter to `__init__`
- [ ] Modify `execute_from_signal()`:
  - Check if pairs trading enabled
  - If yes, delegate to `pairs_coordinator.execute_pair_atomically()`
  - If no, execute as normal directional trade
- [ ] Add integration tests

**Acceptance Criteria:**
- Pairs execution triggered when enabled
- Directional trade falls back when disabled or no hedge

---

#### 5.2 Settings Manager Extension

**File:** `modules/auto_trade/gui/services/settings_manager.py` (modify)

**Tasks:**
- [ ] Add default pairs settings:
  ```python
  "pairs": {
      "enabled": False,
      "min_correlation": 0.65,
      "lookback": 100,
      "timeframe": "1h",
      "refresh_interval": "2h",
      "adx_low": 20,
      "adx_high": 30,
      "stat_arb_direction": "opposite",
      "momentum_direction": "opposite",
      "blended_direction": "opposite",
      "drift_threshold": 0.15,
      "hedge_leverage_min": 1,
      "hedge_leverage_max": 5
  }
  ```
- [ ] Add validation for pairs settings
- [ ] Ensure backward compatibility

**Acceptance Criteria:**
- Default settings loaded on first run
- Settings validation prevents invalid values

---

#### 5.3 Integration Tests

**File:** `tests/integration/test_pairs_full_flow.py`

**Tasks:**
- [ ] Test: Signal → Hedge Selection → Execution → Database
- [ ] Test: Rollback on hedge failure
- [ ] Test: Unified SL closes both legs
- [ ] Test: Separate TP allows independent closes
- [ ] Test: Correlation drift alert triggered
- [ ] Test: Fallback to directional on no hedge

**Acceptance Criteria:**
- All integration tests pass
- Coverage ≥ 90% for pairs modules

---

### Phase 6: E2E Testing & Documentation (Day 7)

#### 6.1 E2E Tests

**File:** `tests/e2e/test_pairs_e2e.py`

**Tasks:**
- [ ] Test: Enable pairs in GUI → trigger signal → verify execution
- [ ] Test: GUI updates with active pairs
- [ ] Test: Manual close from GUI
- [ ] Test: Configuration changes apply correctly

**Acceptance Criteria:**
- All E2E tests pass on testnet
- GUI responsive and functional

---

#### 6.2 Documentation

**Tasks:**
- [ ] Update `README.md` with pairs trading overview
- [ ] Create user guide: `docs/user_guide/pairs_trading.md`
- [ ] Document configuration parameters
- [ ] Add troubleshooting section
- [ ] Update `CLAUDE.md` with pairs trading entry points

**Acceptance Criteria:**
- Users can enable and configure pairs trading from docs
- All parameters documented

---

#### 6.3 Manual Testing Checklist

**Tasks:**
- [ ] Run through manual testing checklist (see design doc)
- [ ] Test on Binance testnet
- [ ] Verify all error scenarios
- [ ] Stress test with 10+ active pairs

**Acceptance Criteria:**
- All manual test cases pass
- No critical bugs found

---

## File Checklist

### New Files to Create

**Core:**
- [ ] `modules/auto_trade/execution/pairs_coordinator.py`
- [ ] `modules/auto_trade/execution/correlation_scanner.py`
- [ ] `modules/auto_trade/monitoring/pairs_metrics.py`
- [ ] `modules/auto_trade/execution/pairs_websocket_handler.py`

**GUI:**
- [ ] `modules/auto_trade/gui/components/pairs_control.py`
- [ ] `modules/auto_trade/gui/components/config_panel_parts/pairs_tab.py`

**Database:**
- [ ] `modules/auto_trade/database/repository/dynamodb/pair_metrics.py`

**Tests:**
- [ ] `tests/test_correlation_scanner.py`
- [ ] `tests/test_pairs_coordinator.py`
- [ ] `tests/test_pairs_metrics_tracker.py`
- [ ] `tests/integration/test_pairs_full_flow.py`
- [ ] `tests/e2e/test_pairs_e2e.py`

**Documentation:**
- [ ] `docs/user_guide/pairs_trading.md`

### Files to Modify

**Core:**
- [ ] `modules/auto_trade/execution/order_executor.py`
- [ ] `modules/auto_trade/gui/services/settings_manager.py`

**Database:**
- [ ] `modules/auto_trade/database/repository/dynamodb/orders.py`

**GUI:**
- [ ] `modules/auto_trade/gui/components/config_panel.py`
- [ ] `modules/auto_trade/gui/components/signals_frame.py`
- [ ] `modules/auto_trade/gui/components/positions_frame.py`
- [ ] `modules/auto_trade/gui/main_window/layout.py`

**Documentation:**
- [ ] `README.md`
- [ ] `CLAUDE.md`

---

## Development Sequence (Recommended)

### Day 1: Core Infrastructure
1. **Morning:** Create `CorrelationScanner` + unit tests
2. **Afternoon:** Create `PairsCoordinator` (without execution)

### Day 2: Coordinator + Database
1. **Morning:** Complete `PairsCoordinator.execute_pair_atomically()`
2. **Afternoon:** Extend database schema (Order + PairMetrics)

### Day 3: Monitoring
1. **Morning:** Create `PairsMetricsTracker`
2. **Afternoon:** Create `PairsWebSocketHandler` + integration

### Day 4: GUI Foundation
1. **Morning:** Create `PairsControl` panel
2. **Afternoon:** Create Config Panel "Pairs Trading" tab

### Day 5: GUI Integration
1. **Morning:** Integrate PairsControl into main layout
2. **Afternoon:** Add visual indicators (signals, positions)

### Day 6: Integration & Testing
1. **Morning:** Extend OrderExecutor, Settings Manager
2. **Afternoon:** Integration tests

### Day 7: E2E & Documentation
1. **Morning:** E2E tests + manual testing
2. **Afternoon:** Documentation + final review

---

## Risk Mitigation

### High-Risk Areas

1. **Atomic Execution + Rollback**
   - Risk: Rollback might fail, leaving orphaned position
   - Mitigation: Extensive testing, critical alert system, manual intervention protocol

2. **Database Schema Migration**
   - Risk: Breaking existing orders functionality
   - Mitigation: Backward compatibility testing, feature flag, rollback plan

3. **GUI Performance with Many Pairs**
   - Risk: UI lag with 10+ active pairs
   - Mitigation: Efficient rendering, pagination, background updates

4. **Correlation Cache Staleness**
   - Risk: Outdated correlations lead to poor hedges
   - Mitigation: Configurable refresh, manual refresh button, cache TTL

---

## Testing Checklist

- [ ] Unit tests pass (≥ 90% coverage for new modules)
- [ ] Integration tests pass
- [ ] E2E tests pass on testnet
- [ ] Manual testing checklist complete
- [ ] No regression in existing auto-trade functionality
- [ ] Performance acceptable (10+ active pairs)
- [ ] Error scenarios handled gracefully

---

## Deployment Checklist

- [ ] All tests passing
- [ ] Documentation complete
- [ ] Settings schema validated
- [ ] Database indexes created
- [ ] Feature flag enabled (if used)
- [ ] Testnet validation complete
- [ ] Code review approved
- [ ] User acceptance testing (UAT) passed

---

## Post-Deployment

### Monitoring

- Monitor `pairs_execution_failed` events
- Track `pairs_rollback_failed` (critical)
- Monitor correlation drift alert frequency
- Track hedge efficiency distribution

### Metrics to Track

- Pairs execution success rate
- Average hedge efficiency
- Correlation drift frequency
- Pairs vs directional PnL comparison

### Future Enhancements

- Multi-leg strategies (3+ symbols)
- Sector-based pairs (auto-select from sector)
- Machine learning hedge selection
- Dynamic TP/SL based on spread
- Pairs backtesting framework

---

**End of Implementation Plan**
