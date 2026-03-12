# Pairs Trading Integration - Design Document

**Project:** Sovereign-IQ Auto Trade
**Module:** Pairs Trading Integration
**Date:** 2026-03-11
**Status:** Design Complete - Ready for Implementation

---

## Table of Contents

1. [Overview](#overview)
2. [Understanding Summary](#understanding-summary)
3. [Assumptions](#assumptions)
4. [Architecture](#architecture)
5. [Components](#components)
6. [Data Flow](#data-flow)
7. [Database Schema](#database-schema)
8. [GUI Integration](#gui-integration)
9. [Error Handling](#error-handling)
10. [Testing Strategy](#testing-strategy)
11. [Decision Log](#decision-log)
12. [Implementation Plan](#implementation-plan)

---

## Overview

This design document describes the integration of **signal-triggered pairs trading** into the existing `modules/auto_trade` system. The integration enables automated hedged trading where single-symbol signals from the ATC Scanner + XGBoost pipeline trigger simultaneous long/short pairs execution.

### Key Features

- **Signal-triggered pairs**: Existing auto-trade signals trigger hedged pairs instead of directional trades
- **Hybrid adaptive strategy**: Statistical arbitrage in low volatility, momentum in high volatility (ADX-based switching)
- **Atomic execution**: Both legs execute together with rollback protection
- **Adaptive position sizing**: Hedge ratio (stat-arb) or risk-parity (momentum) based sizing
- **Sophisticated exit logic**: Separate TP, unified SL, integrated with adaptive time close
- **Dynamic hedge selection**: Real-time correlation scanning with configurable parameters
- **Comprehensive monitoring**: Hedge efficiency, pair Sharpe ratio, correlation drift alerts

---

## Understanding Summary

**What we're building:**

1. **Signal-triggered pairs trading** integrated into existing auto_trade system
2. When ATC Scanner + XGBoost generates a signal → system opens **hedged pair** (2 legs) instead of directional trade
3. **Hybrid adaptive strategy**: Statistical arbitrage in low volatility, momentum in trending markets, ADX-based regime switching
4. **Adaptive position sizing**: Hedge ratio weighting in stat-arb, risk-parity in momentum, smooth blending in transition
5. **Sophisticated exit logic**: Separate TP (each leg can win), unified SL (either leg stop → close both), integrated with existing adaptive time close
6. **Dynamic hedge selection**: Real-time correlation scan to find best hedge symbol (fully configurable parameters)
7. **Graceful degradation**: If no suitable hedge found → trade directional with warning (backward compatible)

---

## Assumptions

1. **Execution environment**: Binance Futures only (existing BinanceClient)
2. **Account balance**: Sufficient margin to support 2x position size (both legs)
3. **Symbol pool**: Hedge selection scans from existing scanner symbol pool
4. **Database**: DynamoDB schema will be extended to link paired orders (new `pair_id` field)
5. **Correlation calculation**: Uses existing `modules/pairs_trading` metrics (reuse `calculate_correlation`, `calculate_hedge_ratio`, etc.)
6. **TP/SL percentages**: Existing auto_trade config values apply symmetrically to both legs
7. **Risk limits**: Existing RiskManager validates **total exposure** (both legs combined) against account limits
8. **GUI framework**: CustomTkinter (existing framework), no new dependencies
9. **Backwards compatibility**: Non-pairs mode continues to work exactly as before
10. **Testnet support**: Pairs trading works in both testnet and production modes
11. **Config persistence**: Pairs settings stored in same `settings.json`
12. **Independent leverage**: Each leg can have different leverage within configurable min/max range
13. **Metrics tracking**: Pairs-specific metrics (hedge efficiency, pair Sharpe, correlation tracking)
14. **Alerts enabled**: Correlation drift alerts for active pairs
15. **Retention**: Same database retention policy as regular orders (90 days)

---

## Architecture

### High-Level Flow

```
Signal Pipeline (ATC + XGBoost)
         ↓
   PairsCoordinator ←→ CorrelationScanner
         ↓                      ↓
   2x OrderTickets      HedgeCandidate
         ↓
   OrderExecutor (atomic execution)
         ↓
   BinanceClient (both legs)
         ↓
   Database (linked by pair_id)
         ↓
   PairsMetricsTracker
```

### Architectural Approach

**Approach 3: Integrated Pairs Layer**

- Extend existing execution pipeline with pairs capability
- `PairsCoordinator` intercepts signals and orchestrates pairs logic
- Reuses 90% of existing infrastructure (risk management, execution, database, WebSocket)
- New components: `PairsCoordinator`, `CorrelationScanner`, `PairsMetricsTracker`
- Database: Extend `Order` model with pair fields, add `PairMetrics` entity

---

## Components

### 1. PairsCoordinator

**File:** `modules/auto_trade/execution/pairs_coordinator.py`

**Responsibility:** Orchestrate pairs trading logic

**Key Methods:**
- `should_activate_pairs(signal, settings) -> bool`
- `find_hedge_symbol(signal_symbol) -> Optional[HedgeCandidate]`
- `determine_regime(signal_symbol, hedge_symbol) -> Regime`
- `calculate_position_sizes(regime, ...) -> Tuple[float, float]`
- `execute_pair_atomically(signal_ticket, hedge_ticket) -> PairExecutionResult`

### 2. CorrelationScanner

**File:** `modules/auto_trade/execution/correlation_scanner.py`

**Responsibility:** Dynamic hedge symbol selection

**Key Methods:**
- `scan_hedge_candidates(signal_symbol, settings) -> List[HedgeCandidate]`
- `calculate_correlation(symbol1, symbol2, lookback) -> float`
- `calculate_hedge_ratio(symbol1, symbol2, regime) -> float`
- `refresh_correlation_cache() -> None`

**Features:**
- Caches correlations with configurable refresh interval
- Filters by minimum correlation threshold
- Returns ranked candidates by correlation strength
- Reuses `modules/pairs_trading` quantitative metrics

### 3. PairsMetricsTracker

**File:** `modules/auto_trade/monitoring/pairs_metrics.py`

**Responsibility:** Track pairs-specific metrics

**Key Methods:**
- `track_pair_entry(pair_id, correlation, hedge_ratio) -> None`
- `update_pair_metrics(pair_id, signal_pnl, hedge_pnl) -> None`
- `calculate_hedge_efficiency(pair_id) -> float`
- `check_correlation_drift(pair_id) -> Optional[Alert]`

**Metrics Tracked:**
- Hedge efficiency (actual vs theoretical hedge performance)
- Pair Sharpe ratio
- Real-time correlation drift
- Net PnL (signal + hedge)

### 4. PairsControl (GUI)

**File:** `modules/auto_trade/gui/components/pairs_control.py`

**Responsibility:** Dedicated pairs control panel

**Features:**
- Enable/disable pairs trading toggle
- Real-time active pairs summary (symbol, correlation, PnL)
- Correlation drift alerts display
- Manual refresh correlations button

### 5. Config Panel Extension

**File:** `modules/auto_trade/gui/components/config_panel.py` (extended)

**New Tab:** "Pairs Trading"

**Settings:**
- Correlation parameters (min threshold, lookback, timeframe, refresh interval)
- ADX regime thresholds (low, high)
- Hedge direction per-regime (stat-arb, momentum, blended)
- Advanced settings (drift alert threshold, hedge leverage range)

---

## Data Flow

### Execution Sequence

1. **Signal Generation** (Existing Pipeline)
   - ATC Scanner + XGBoost generates signal
   - Signal: `{ symbol: "BTC/USDT", signal_type: "LONG", score: 85 }`

2. **Pairs Activation Check** (PairsCoordinator)
   - Check if pairs trading enabled
   - If YES → proceed to hedge selection
   - If NO → execute as normal directional trade

3. **Hedge Symbol Selection** (CorrelationScanner)
   - Check correlation cache freshness
   - Scan top N symbols for highest correlation
   - Filter by minimum correlation threshold
   - Return best hedge candidate (e.g., "ETH/USDT", corr=0.75)
   - If no suitable hedge → fallback to directional

4. **Regime Detection** (PairsCoordinator)
   - Calculate ADX for both symbols
   - Determine regime:
     - ADX < 20 → STAT_ARB
     - 20 ≤ ADX < 30 → BLENDED
     - ADX ≥ 30 → MOMENTUM

5. **Hedge Direction Determination**
   - Get config for current regime
   - Determine hedge side (opposite or same as signal)

6. **Position Sizing Calculation**
   - STAT_ARB: Use hedge ratio (OLS/Kalman)
   - MOMENTUM: Use risk-parity (ATR-based)
   - BLENDED: 50/50 blend of both methods
   - Apply independent leverage per leg

7. **Order Ticket Creation**
   - Create signal_ticket with pair_id
   - Create hedge_ticket with same pair_id
   - Apply TP/SL from auto_trade config

8. **Atomic Execution** (OrderExecutor)
   - Execute signal leg → wait for fill
   - If signal fails → abort, return error
   - Execute hedge leg → wait for fill
   - If hedge fails → ROLLBACK (close signal leg)
   - If both succeed → set TP/SL, save to database

9. **Post-Execution Monitoring** (PairsMetricsTracker)
   - Track real-time PnL for both legs
   - Calculate hedge efficiency
   - Monitor correlation drift
   - Emit alerts if correlation drops below threshold

### Fallback Flow

If no suitable hedge found:
- Log warning
- Emit event: `pairs_fallback_directional`
- GUI shows: "⚠️ BTC/USDT - Directional (no hedge)"
- Execute as normal single-leg trade

---

## Database Schema

### Extended Order Model

**Table:** `orders` (existing, add fields)

**New Fields:**
```python
pair_id: Optional[str]                # UUID linking both legs
pair_leg: Optional[Literal["SIGNAL", "HEDGE"]]
pair_correlation: Optional[float]     # Correlation at entry
pair_hedge_ratio: Optional[float]     # Hedge ratio used
pair_regime: Optional[Literal["STAT_ARB", "MOMENTUM", "BLENDED"]]
pair_partner_symbol: Optional[str]    # Partner symbol
pair_is_hedged: bool = False          # Quick filter flag
```

**Indexes (GSI):**
- `pair_id-index`: Query both legs by pair_id
- `pair_is_hedged-opened_at-index`: Query all hedged orders

### New PairMetrics Entity

**Table:** `pair_metrics` (new)

**Fields:**
```python
pair_id: str                          # PK
created_at: datetime                  # SK
signal_symbol: str
hedge_symbol: str
entry_correlation: float
entry_hedge_ratio: float
regime: str

# Real-time tracking
current_correlation: Optional[float]
correlation_updated_at: Optional[datetime]

# PnL tracking
signal_pnl: float
hedge_pnl: float
net_pnl: float

# Performance metrics
hedge_efficiency: Optional[float]
pair_sharpe: Optional[float]

# Status
status: Literal["ACTIVE", "PARTIAL_CLOSED", "FULLY_CLOSED"]
signal_leg_status: str
hedge_leg_status: str

# Alerts
correlation_drift_alerted: bool
last_alert_at: Optional[datetime]
```

**Index:**
- `status-created_at-index`: Query active/closed pairs

---

## GUI Integration

### 1. New Pairs Control Panel

**Location:** Top-level panel in main layout (alongside Scanner Control, Auto-Trade Control)

**Layout:**
- Status indicator (enabled/disabled, active pairs count)
- Enable/disable toggle
- Refresh correlations button
- Active pairs table (pair ID, symbols, correlation, PnL)
- Correlation drift alerts section

### 2. Config Panel Extension

**New Tab:** "Pairs Trading"

**Sections:**
1. **Correlation Parameters**
   - Min correlation threshold (slider: 0.50 - 0.90)
   - Lookback period (entry: 50-500 candles)
   - Timeframe dropdown (15m, 30m, 1h, 4h)
   - Refresh interval (1h, 2h, 4h, 12h)

2. **ADX Regime Thresholds**
   - ADX Low (stat-arb threshold, default: 20)
   - ADX High (momentum threshold, default: 30)

3. **Hedge Direction**
   - Stat-Arb Mode (opposite/same)
   - Momentum Mode (opposite/same)
   - Blended Mode (opposite/same/correlation_based)

4. **Advanced Settings**
   - Correlation drift alert threshold (default: 0.15)
   - Hedge leverage range (min-max)

### 3. Visual Indicators

**Signals Frame:**
- Hedged signals marked with 🔗 icon
- Show hedge partner symbol in signal text

**Positions Frame:**
- Group paired positions together
- Show correlation and net PnL for pairs

---

## Error Handling

### 1. Execution Failures

**Signal Leg Fails:**
- Abort execution (don't execute hedge)
- Log error and emit event
- GUI shows: "❌ BTC/USDT pair execution failed (signal leg)"

**Hedge Leg Fails After Signal Succeeds:**
- **ROLLBACK**: Close signal leg immediately at market
- Log rollback attempt
- If rollback succeeds → clean abort
- If rollback fails → **CRITICAL ALERT** (manual intervention required)

**TP/SL Placement Fails:**
- Don't abort trade (positions already open)
- Queue for `EnsureTPSLJob` to retry
- GUI shows: "⚠️ TP/SL placement pending retry"

### 2. Partial Fills

- Accept partial fills ≥ 95%
- Adjust partner leg size proportionally to actual fill
- Log adjustment for transparency

### 3. Correlation Drift (Active Pairs)

- Recalculate correlation every 1 hour for active pairs
- If drift > threshold (e.g., 0.15):
  - Emit alert to GUI
  - Mark pair as alerted (don't spam)
  - User decides whether to close manually

### 4. Insufficient Margin

- Validate total margin requirement before execution
- Include 5% buffer for fees/slippage
- If insufficient → reject pair, show error in GUI

### 5. Unified SL Triggered

- If either leg hits SL → close partner leg immediately at market
- Emit event: `pairs_unified_sl_triggered`
- GUI shows: "🛑 BTC/ETH pair closed (unified SL)"

### 6. Exchange API Errors

- Retry with exponential backoff (max 3 attempts)
- Handle rate limits gracefully (backoff)
- Don't retry permanent errors (invalid order)

---

## Testing Strategy

### Unit Tests (70%)

**Test: PairsCoordinator**
- Pairs activation logic
- Regime determination (ADX thresholds)
- Position sizing calculations (stat-arb, momentum, blended)
- Hedge direction logic

**Test: CorrelationScanner**
- Correlation calculation
- Hedge candidate ranking
- Cache refresh logic
- Hedge ratio calculation (OLS, Kalman)

**Test: PairsMetricsTracker**
- Hedge efficiency calculation
- Correlation drift detection
- Alert triggering logic

### Integration Tests (25%)

**Test: Full Pairs Execution Flow**
- Signal → hedge selection → atomic execution → database save
- Verify both orders created with correct pair_id linkage
- Verify PairMetrics entry created

**Test: Rollback on Hedge Failure**
- Signal succeeds, hedge fails
- Verify signal leg closed (rollback)
- Verify database state consistent

### E2E Tests (5%)

**Test: GUI → Execution → Database**
- Enable pairs trading in GUI
- Trigger signal from scanner
- Verify pair executed and GUI updated

### Test Scenarios

Critical scenarios to cover:
- Happy paths (stat-arb, momentum, blended regimes)
- Error cases (execution failures, insufficient margin, partial fills)
- Exit scenarios (signal TP first, hedge SL unified close, correlation drift)
- Configuration edge cases (exact thresholds, extreme values)

---

## Decision Log

### Decision 1: Execution Model
**Chosen:** Atomic Execution
**Why:** Prevents naked exposure, true pairs trading, cleaner state management

### Decision 2: Strategy Type
**Chosen:** Hybrid Adaptive Strategy
**Why:** Adapts to market conditions (stat-arb in sideways, momentum in trends)

### Decision 3: Position Sizing
**Chosen:** Adaptive Hybrid Sizing
**Why:** Sizing method matches regime (hedge ratio for stat-arb, risk-parity for momentum)

### Decision 4: Exit Logic
**Chosen:** Separate TP + Unified SL + Adaptive Time Close
**Why:** Asymmetric risk/reward, leverage existing feature, intuitive logic

### Decision 5: TP/SL Configuration
**Chosen:** Symmetric TP/SL
**Why:** Reuses existing config, simple, user-familiar

### Decision 6: Signal Source
**Chosen:** Existing ATC Scanner + XGBoost
**Why:** Proven signal quality, natural integration, simpler UX

### Decision 7: Hedge Selection
**Chosen:** Dynamic Correlation Lookup
**Why:** Adaptive to market, no manual maintenance, fully automated

### Decision 8: Correlation Parameters
**Chosen:** Custom Configurable
**Why:** Maximum flexibility, user can optimize, future-proof

### Decision 9: ADX Regime Switching
**Chosen:** Configurable Multi-Threshold
**Why:** Smooth transitions, reduces whipsaw, configurable

### Decision 10: Fallback Behavior
**Chosen:** Trade Directional with Warning
**Why:** Backward compatible, don't miss opportunities, user awareness

### Decision 11: Hedge Direction
**Chosen:** User Configurable Per-Regime
**Why:** Maximum control, educational, flexible

### Decision 12: GUI Integration
**Chosen:** New Top-Level Pairs Control Panel
**Why:** High visibility, dedicated control surface, real-time monitoring

### Decision 13: Database Schema
**Chosen:** Extend Order model + new PairMetrics
**Why:** Unified queries, first-class support, rich analytics

### Decision 14: Architectural Approach
**Chosen:** Integrated Pairs Layer
**Why:** Code reuse, maintainability, natural UX, extensibility

### Decision 15: Leverage Handling
**Chosen:** Independent leverage per leg
**Why:** Risk granularity, flexibility, safety (range limits)

*(Full decision log with 19 decisions documented in design process)*

---

## Implementation Plan

See [Implementation Plan](#implementation-plan-section) below for file-by-file breakdown and development sequence.

---

## Appendix

### Key Technologies

- **Python**: 3.12+
- **GUI**: CustomTkinter
- **Database**: DynamoDB (AWS)
- **Exchange**: Binance Futures (ccxt)
- **Metrics**: `modules/pairs_trading` (correlation, cointegration, hedge ratios)
- **Existing Infrastructure**: OrderExecutor, OrderManager, RiskManager, EventSystem

### References

- `modules/pairs_trading/` - Existing pairs analysis module
- `modules/auto_trade/` - Auto trade system
- `modules/auto_trade/execution/order_executor.py` - Order execution
- `modules/auto_trade/database/config.py` - Database configuration
- `CLAUDE.md` - Project overview and conventions

---

**End of Design Document**
