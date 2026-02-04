# Gradual Recovery Strategy Guide

## Overview

The Gradual Recovery Strategy is a risk-averse approach to recovering from trading losses. Instead of using the aggressive Martingale doubling technique, this strategy recovers losses through a series of small, controlled winning trades.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Options](#configuration-options)
- [Scaling Strategies](#scaling-strategies)
- [Example Scenarios](#example-scenarios)
- [Best Practices](#best-practices)

---

## Installation

The Gradual Recovery module is included in the auto_trade package.

```python
from modules.auto_trade.strategies.gradual_recovery import (
    GradualRecoveryStrategy,
    RecoveryConfig,
    create_recovery_plan,
)
```

---

## Quick Start

### Basic Usage

```python
from modules.auto_trade.strategies.gradual_recovery import GradualRecoveryStrategy

# Initialize with $500 loss to recover
config = {
    'target_profit_per_trade': 5.0,  # Target 5% profit per trade
    'max_recovery_trades': 20,
}

recovery = GradualRecoveryStrategy(
    initial_loss=500.0,
    config=config,
)

# After each winning trade
recovery.record_profit(25.0)  # Record a $25 profit

# After a losing trade during recovery
recovery.record_loss(10.0)  # Record a $10 setback

# Get current state
state = recovery.get_state()
print(f"Recovery: {state.recovery_percentage:.1f}%")
print(f"Remaining: ${state.remaining_loss:.2f}")

# Get recommendations for next trade
next_margin = recovery.calculate_next_position_size()
next_leverage = recovery.calculate_next_leverage()
```

### Creating a Recovery Plan

```python
from modules.auto_trade.strategies.gradual_recovery import create_recovery_plan

plan = create_recovery_plan(
    initial_loss=500,
    config={
        'target_profit_per_trade': 5.0,
        'max_recovery_trades': 20,
    }
)

print(f"Estimated trades needed: {plan['estimated_trades_needed']}")
print(f"Suggested margin: ${plan['suggested_margin_per_trade']:.2f}")
print(f"Risk level: {plan['risk_assessment']}")
```

---

## Configuration Options

### Required Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `initial_loss` | `float` | Total loss amount to recover in USDT | Required |

### Optional Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `target_profit_per_trade` | `float` | Target % profit per trade (5 = 5%) | 5.0 |
| `max_recovery_trades` | `int` | Maximum trades before giving up | 20 |
| `max_total_loss` | `float` | Stop-loss for entire recovery (2x initial_loss) | 2x initial_loss |
| `margin_scaling_mode` | `str` | How to adjust margin size | 'fixed' |
| `leverage_scaling_mode` | `str` | How to adjust leverage | 'fixed' |
| `min_leverage` | `int` | Minimum leverage | 2 |
| `max_leverage` | `int` | Maximum leverage | 10 |
| `enable_streak_bonus` | `bool` | Increase margin on win streaks | False |

---

## Scaling Strategies

### Margin Scaling Modes

#### 1. Fixed Mode (`"fixed"`)

Uses constant margin regardless of progress.

```python
config = {'margin_scaling_mode': 'fixed'}
# Margin stays at initial_loss / 10
```

**Best for:** Conservative traders, predictable recovery timeline.

#### 2. Progressive Mode (`"progressive"`)

Increases margin gradually as recovery progresses.

```python
config = {'margin_scaling_mode': 'progressive'}
# Formula: base_margin * (1 + recovery_percentage * 0.5)
# Example: 20% recovered → 1.1x margin
```

**Best for:** Balanced risk-reward, moderate aggression.

#### 3. Adaptive Mode (`"adaptive"`)

Adjusts based on:
- Win streak (increase if on streak)
- Recovery progress (larger margin if close to completion)
- Recent volatility (reduce if market unstable)

```python
config = {'margin_scaling_mode': 'adaptive'}
```

**Best for:** Experienced traders, dynamic market conditions.

### Leverage Scaling Modes

#### 1. Fixed Leverage (`"fixed"`)

Constant leverage throughout recovery.

```python
config = {'leverage_scaling_mode': 'fixed', 'min_leverage': 3}
```

**Best for:** Conservative, predictable risk.

#### 2. Progressive Leverage (`"progressive"`)

Starts low, increases gradually with progress.

```python
config = {'leverage_scaling_mode': 'progressive', 'min_leverage': 2, 'max_leverage': 10}
# Starts at 2x, gradually increases to 10x
```

**Best for:** Moderate aggression.

#### 3. Adaptive Leverage (`"adaptive"`)

Adjusts based on win streak and progress.

```python
config = {'leverage_scaling_mode': 'adaptive', 'enable_streak_bonus': True}
```

**Best for:** Dynamic adjustment, experienced traders.

---

## Example Scenarios

### Scenario 1: Conservative Recovery

```python
config = {
    'target_profit_per_trade': 3.0,  # Small, steady profits
    'max_recovery_trades': 33,  # Allow many trades
    'margin_scaling_mode': 'fixed',
    'leverage_scaling_mode': 'fixed',
    'min_leverage': 2,
    'max_leverage': 3,
}

recovery = GradualRecoveryStrategy(initial_loss=500, config=config)
```

**Profile:** Low risk, slow recovery, high probability of success.

### Scenario 2: Moderate Recovery

```python
config = {
    'target_profit_per_trade': 5.0,
    'max_recovery_trades': 20,
    'margin_scaling_mode': 'progressive',
    'leverage_scaling_mode': 'progressive',
    'min_leverage': 3,
    'max_leverage': 7,
}

recovery = GradualRecoveryStrategy(initial_loss=500, config=config)
```

**Profile:** Balanced risk-reward, standard approach.

### Scenario 3: Aggressive Recovery

```python
config = {
    'target_profit_per_trade': 8.0,  # Higher profit target
    'max_recovery_trades': 13,  # Fewer trades
    'margin_scaling_mode': 'adaptive',
    'leverage_scaling_mode': 'adaptive',
    'min_leverage': 5,
    'max_leverage': 15,
    'enable_streak_bonus': True,
}

recovery = GradualRecoveryStrategy(initial_loss=500, config=config)
```

**Profile:** Higher risk, faster recovery, requires good win rate.

---

## Best Practices

### 1. Choose Appropriate Risk Level

- **Beginners:** Use fixed scaling with low leverage (2-3x)
- **Intermediate:** Progressive scaling, moderate leverage (3-7x)
- **Advanced:** Adaptive scaling, higher leverage (5-15x)

### 2. Set Realistic Targets

- **Conservative:** 3-4% profit per trade
- **Moderate:** 5-7% profit per trade
- **Aggressive:** 8-10% profit per trade

### 3. Monitor Win Rate

- **Excellent (70%+):** Can use aggressive settings
- **Good (60-70%):** Use moderate settings
- **Poor (<60%):** Use conservative settings or reconsider recovery

### 4. Use Safety Limits

Always set:
- `max_recovery_trades`: Prevents infinite recovery attempts
- `max_total_loss`: Limits total exposure during recovery

### 5. Track Progress

```python
# Check progress regularly
state = recovery.get_state()
print(recovery.progress_bar)  # Visual progress bar
print(f"Estimated trades: {state.estimated_trades_remaining}")

# Safety check
if recovery.should_stop():
    print("Recovery limit reached. Stopping.")
    recovery.reset()
```

### 6. Combine with Quality Signals

Don't use recovery blindly. Combine with:
- Technical analysis signals
- Market sentiment analysis
- Risk management rules

---

## API Reference

### Class: `GradualRecoveryStrategy`

#### Constructor

```python
GradualRecoveryStrategy(
    initial_loss: float,
    config: RecoveryConfig,
    database: Optional[object] = None
)
```

#### Methods

##### `record_profit(profit_amount: float)`
Record a winning trade during recovery.

##### `record_loss(loss_amount: float)`
Record a losing trade during recovery.

##### `calculate_next_position_size() -> float`
Get recommended margin for next trade.

##### `calculate_next_leverage() -> int`
Get recommended leverage for next trade.

##### `estimate_remaining_trades() -> int`
Estimate trades needed to complete recovery.

##### `should_stop() -> bool`
Check if safety limits have been reached.

##### `get_state() -> RecoveryState`
Get current recovery state object.

##### `reset()`
Reset recovery to initial state.

#### Properties

##### `is_active -> bool`
True if recovery is in progress.

##### `recovery_percentage -> float`
Current recovery progress (0-100%).

##### `progress_bar -> str`
Visual progress bar (e.g., "█████░░░░░ 50%").

---

## Comparison with Martingale

| Feature | Gradual Recovery | Martingale |
|---------|------------------|------------|
| Risk per trade | Low | High (exponential) |
| Time to recover | Slower | Faster (if successful) |
| Probability of liquidation | Low | High |
| Psychological stress | Low | High |
| Capital required | Moderate | High |
| Suitable for | Most traders | High-risk traders only |

---

## Troubleshooting

### Recovery not progressing

1. Check win rate - if below 60%, consider stopping
2. Review signal quality
3. Adjust to more conservative settings

### Hitting max trades limit

1. Increase `max_recovery_trades` slightly
2. Lower `target_profit_per_trade`
3. Improve trade entry quality

### Too many setbacks

1. Market may be against you - consider pausing
2. Reduce leverage
3. Switch to fixed mode for stability

---

## Support

For issues or questions, please refer to the main project documentation or open an issue in the repository.
