# Task: Gradual Recovery Strategy Implementation

## Overview

Implement a new recovery system that recovers initial losses through a series of small winning trades instead of one-shot Martingale doubling.

**Status:** 🟢 Completed  
**Priority:** HIGH  
**Estimated Time:** 2-3 days  
**Target File:** `modules/auto_trade/strategies/gradual_recovery.py`

---

## Requirements

### Core Concept

- **Input:** Initial loss amount (e.g., -$500)
- **Goal:** Recover to breakeven ($0 total loss) through multiple small profit trades
- **Strategy:** Adjust position sizing and leverage progressively based on recovery progress
- **Exit:** Stop when total_loss <= 0 OR safety limits reached

### Key Features

1. ✅ Track cumulative loss from initial failure
2. ✅ Accept small profits (5%, 10%, etc.) and accumulate them
3. ✅ Dynamically adjust margin/leverage based on:
   - Remaining loss to recover
   - Current win streak
   - Risk tolerance
4. ✅ Provide recovery roadmap (estimated trades needed)
5. ✅ Safety limits to prevent infinite loops

---

## Implementation Tasks

### Phase 1: Core Data Structures

**File:** `modules/auto_trade/strategies/gradual_recovery.py`

- [x] **Task 1.1:** Create `RecoveryState` dataclass

  ```python
  @dataclass
  class RecoveryState:
      initial_loss: float           # Original loss to recover
      remaining_loss: float          # Current loss left
      total_profit_accumulated: float
      recovery_percentage: float     # 0-100%
      trades_count: int              # Number of recovery trades
      win_streak: int                # Current winning streak
      is_complete: bool
      estimated_trades_remaining: int
  ```

- [x] **Task 1.2:** Create `RecoveryConfig` TypedDict

  ```python
  class RecoveryConfig(TypedDict, total=False):
      target_profit_per_trade: float  # Target % profit per trade (default: 5%)
      max_recovery_trades: int        # Max trades before giving up (default: 20)
      max_total_loss: float           # Stop-loss for the entire recovery (default: 2x initial_loss)
      margin_scaling_mode: str        # "fixed", "progressive", "adaptive"
      leverage_scaling_mode: str      # "fixed", "progressive", "adaptive"
      min_leverage: int               # Minimum leverage (default: 2x)
      max_leverage: int               # Maximum leverage (default: 10x)
      enable_streak_bonus: bool       # Increase margin on win streaks (default: False)
  ```

### Phase 2: Core Recovery Logic

**File:** `modules/auto_trade/strategies/gradual_recovery.py`

- [x] **Task 2.1:** Implement `GradualRecoveryStrategy` class initialization
   - Accept `initial_loss`, `config`, `database` (optional)
   - Initialize state variables
   - Validate config parameters

- [x] **Task 2.2:** Implement `record_profit(profit_amount: float)`
   - Subtract profit from `remaining_loss`
   - Increment `trades_count`
   - Update `win_streak`
   - Check if recovery is complete (`remaining_loss <= 0`)
   - Log progress percentage
   - Persist to database if available

- [x] **Task 2.3:** Implement `record_loss(loss_amount: float)`
   - Add to `remaining_loss` (setback during recovery)
   - Reset `win_streak` to 0
   - Log warning about setback
   - Check if `max_total_loss` exceeded

- [x] **Task 2.4:** Implement `calculate_next_position_size() -> float`
   - Calculate recommended margin based on:
     - `remaining_loss`
     - `margin_scaling_mode` (fixed/progressive/adaptive)
     - Current `win_streak` (if `enable_streak_bonus=True`)
   - Return suggested position size in USDT

- [x] **Task 2.5:** Implement `calculate_next_leverage() -> int`
   - Calculate recommended leverage based on:
     - `leverage_scaling_mode`
     - Recovery progress (%)
     - Risk tolerance
   - Clamp between `min_leverage` and `max_leverage`
   - Return leverage value

- [x] **Task 2.6:** Implement `estimate_remaining_trades() -> int`
   - Formula: `remaining_loss / (avg_profit_per_trade)`
   - Use configurable `target_profit_per_trade`
   - Account for historical win rate if available

- [x] **Task 2.7:** Implement `should_stop() -> bool`
   - Check if `trades_count >= max_recovery_trades`
   - Check if `remaining_loss >= max_total_loss`
   - Return `True` if any safety limit reached

- [x] **Task 2.8:** Implement `get_state() -> RecoveryState`
   - Return current recovery state
   - Calculate `recovery_percentage`
   - Calculate `estimated_trades_remaining`

- [x] **Task 2.9:** Implement `reset()`
   - Reset all state variables to initial
   - Clear database records if available

### Phase 3: Scaling Strategies

**File:** `modules/auto_trade/strategies/gradual_recovery.py`

- [x] **Task 3.1:** Implement `_calculate_fixed_margin() -> float`
   - Return constant margin regardless of progress

- [x] **Task 3.2:** Implement `_calculate_progressive_margin() -> float`
   - Increase margin gradually as recovery progresses
   - Formula: `base_margin * (1 + recovery_percentage * scaling_factor)`
   - Example: 20% recovered → 1.2x margin

- [x] **Task 3.3:** Implement `_calculate_adaptive_margin() -> float`
   - Adjust based on:
     - Win streak (increase if on streak)
     - Remaining loss (larger margin if close to completion)
     - Recent volatility (reduce if market unstable)

- [x] **Task 3.4:** Implement `_calculate_fixed_leverage() -> int`
   - Return constant leverage

- [x] **Task 3.5:** Implement `_calculate_progressive_leverage() -> int`
   - Start low, increase gradually
   - Example: 50% recovered → increase from 2x to 5x

- [x] **Task 3.6:** Implement `_calculate_adaptive_leverage() -> int`
   - Similar to adaptive margin
   - Account for market conditions

### Phase 4: Integration & Utilities

**File:** `modules/auto_trade/strategies/gradual_recovery.py`

- [x] **Task 4.1:** Add database persistence methods
   - `_persist_state()` - Save to DB
   - `_load_state()` - Restore from DB
   - `_clear_state()` - Delete from DB

- [x] **Task 4.2:** Add logging & monitoring
   - Log every profit/loss event
   - Log milestone achievements (25%, 50%, 75%, 100%)
   - Emit warnings when approaching limits

- [x] **Task 4.3:** Add properties for easy access

   ```python
   @property
   def is_active(self) -> bool
   
   @property
   def recovery_percentage(self) -> float
   
   @property
   def progress_bar(self) -> str  # Visual progress "█████░░░░░ 50%"
   ```

- [x] **Task 4.4:** Create helper function `create_recovery_plan()`
   - Generate detailed recovery roadmap
   - Return dict with:
     - Estimated trades needed
     - Suggested margin per trade
     - Suggested leverage progression
     - Risk assessment

### Phase 5: Testing & Validation

**File:** `modules/auto_trade/tests/test_gradual_recovery.py`

- [x] **Task 5.1:** Write unit tests for profit recording
   - Test single profit reduces `remaining_loss`
   - Test multiple profits accumulate correctly
   - Test completion detection

- [x] **Task 5.2:** Write unit tests for loss recording
   - Test setback increases `remaining_loss`
   - Test win streak resets
   - Test safety limit triggers

- [x] **Task 5.3:** Write unit tests for position sizing
   - Test fixed mode returns constant values
   - Test progressive mode scales correctly
   - Test adaptive mode responds to streaks

- [x] **Task 5.4:** Write integration test scenarios
   - Scenario 1: Perfect recovery (10 wins, 5% each)
   - Scenario 2: Setback recovery (8 wins, 2 losses)
   - Scenario 3: Failed recovery (hit max_trades)
   - Scenario 4: Exceeded max_total_loss

- [x] **Task 5.5:** Write performance/stress tests
   - Test with very large loss amounts
   - Test with many trades (100+)
   - Test database persistence under load

### Phase 6: Documentation

**File:** `modules/auto_trade/docs/GRADUAL_RECOVERY_GUIDE.md`

- [x] **Task 6.1:** Write comprehensive usage guide
   - Installation & setup
   - Configuration options
   - Example scenarios
   - Best practices

- [x] **Task 6.2:** Document scaling strategies
   - When to use fixed/progressive/adaptive
   - Risk-reward tradeoffs
   - Example configurations for different risk profiles

- [x] **Task 6.3:** Create comparison with Martingale
   - Side-by-side feature comparison
   - Risk analysis
   - When to use which strategy

- [x] **Task 6.4:** Add inline code documentation
   - Docstrings for all public methods
   - Type hints for all parameters
   - Example code snippets

### Phase 7: GUI Integration

**File:** `modules/auto_trade/gui/components/recovery_panel.py`

- [x] **Task 7.1:** Create RecoveryPanel widget
   - Display current recovery state
   - Show progress bar
   - Show estimated trades remaining
   - Show margin/leverage recommendations

- [x] **Task 7.2:** Add recovery configuration tab
   - Input fields for all `RecoveryConfig` options
   - Presets (Conservative/Moderate/Aggressive)
   - Live calculation preview

- [x] **Task 7.3:** Add recovery history chart
   - Line chart showing `remaining_loss` over time
   - Highlight wins/losses
   - Show milestone markers

- [x] **Task 7.4:** Add recovery alerts
   - Notify when 50% recovered
   - Notify when 100% recovered
   - Alert when approaching safety limits

---

## Example Usage

```python
from modules.auto_trade.strategies.gradual_recovery import (
    GradualRecoveryStrategy,
    RecoveryConfig,
)

# Initialize with $500 loss to recover
config: RecoveryConfig = {
    "target_profit_per_trade": 5.0,  # Target 5% profit per trade
    "max_recovery_trades": 20,
    "margin_scaling_mode": "progressive",
    "leverage_scaling_mode": "fixed",
    "min_leverage": 2,
    "max_leverage": 10,
}

recovery = GradualRecoveryStrategy(
    initial_loss=500.0,
    config=config,
    database=db,  # Optional
)

# After each trade
if trade_result > 0:
    recovery.record_profit(trade_result)  # e.g., +$25
else:
    recovery.record_loss(abs(trade_result))  # e.g., -$10

# Get recommendations for next trade
next_margin = recovery.calculate_next_position_size()
next_leverage = recovery.calculate_next_leverage()

# Check progress
state = recovery.get_state()
print(f"Recovery: {state.recovery_percentage:.1f}%")
print(f"Remaining: ${state.remaining_loss:.2f}")
print(f"Est. trades: {state.estimated_trades_remaining}")

# Safety check
if recovery.should_stop():
    print("Recovery limit reached. Stopping.")
    recovery.reset()
```

---

## Success Criteria

- [x] All unit tests pass with >90% coverage
- [x] Integration tests demonstrate successful recovery scenarios
- [x] GUI panel displays accurate real-time data
- [x] Documentation is clear and comprehensive
- [x] Performance: Can handle 100+ trades without lag
- [x] Database persistence works correctly
- [x] Safety limits prevent runaway losses

---

## Risk Considerations

### Advantages over Martingale

- ✅ Lower risk per trade (no exponential growth)
- ✅ More sustainable for gradual recovery
- ✅ Psychologically easier (small wins accumulate)
- ✅ Less vulnerable to liquidation

### Disadvantages

- ⚠️ Requires more winning trades
- ⚠️ Longer time to full recovery
- ⚠️ Vulnerable to losing streaks during recovery
- ⚠️ May accumulate small losses if win rate too low

### Mitigation Strategies

1. Set strict `max_recovery_trades` limit
2. Implement `max_total_loss` stop-loss
3. Use conservative `target_profit_per_trade` (3-5%)
4. Monitor win rate and adjust strategy if <60%
5. Combine with signal quality filters

---

## Timeline

- **Week 1:** Phase 1-3 (Core implementation)
- **Week 2:** Phase 4-5 (Integration & testing)
- **Week 3:** Phase 6-7 (Documentation & GUI)

---

## Notes

- This strategy is complementary to Martingale, not a replacement
- Consider allowing users to switch between strategies based on market conditions
- May want to implement hybrid mode (start with Gradual, escalate to Martingale if stalled)
- Database schema will need new tables for recovery tracking

---

**Created:** 2026-02-05  
**Status:** Ready for implementation  
**Assignee:** TBD
