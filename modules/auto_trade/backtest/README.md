# Auto-Trade Backtesting Module

Integration of the existing `modules/backtester` with auto-trade system specific requirements.

## Overview

This module adapts the existing `FullBacktester` to test auto-trade strategies with:
- 50% Stop Loss / 5% Take Profit
- 95% Balance risk per trade
- 2x Leverage (scales with Martingale)
- Break-Even protection at 30% drawdown
- Optional Martingale loss recovery strategy

## Structure

```
modules/auto_trade/backtest/
├── __init__.py           # Module initialization
├── adapter.py            # AutoTradeBacktester adapter class
├── strategy_simulator.py # Full strategy simulation
└── README.md            # This file
```

## Quick Start

### Basic Backtest (Without Martingale)

```python
from modules.auto_trade.backtest import AutoTradeBacktester
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager

# Initialize
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Create backtester
backtester = AutoTradeBacktester(
    data_fetcher=data_fetcher,
    stop_loss_pct=0.50,     # 50% stop loss
    take_profit_pct=0.05,   # 5% take profit
    risk_per_trade=0.95,    # 95% balance per trade
    leverage=2,             # 2x leverage
    enable_breakeven=True,  # Enable BE protection
    enable_martingale=False # Disable Martingale
)

# Run backtest
result = backtester.backtest_strategy(
    symbol="BTC/USDT",
    timeframe="1h",
    lookback=288,  # 12 days
    initial_capital=10000.0
)

# Display results
print(f"Win rate: {result['metrics']['win_rate']*100:.2f}%")
print(f"Total return: {result['metrics']['total_return']*100:.2f}%")
print(f"BE moves: {result['metrics']['breakeven_moves']}")
```

### Martingale Backtest

```python
# Create backtester with Martingale enabled
backtester = AutoTradeBacktester(
    data_fetcher=data_fetcher,
    enable_martingale=True,      # Enable Martingale
    martingale_max_steps=4,      # Max 4 steps
    martingale_max_leverage=16   # Max 16x leverage
)

# Run backtest
result = backtester.backtest_strategy(
    symbol="BTC/USDT",
    timeframe="1h",
    lookback=288,
    initial_capital=10000.0
)

# Validate Martingale safety
safety = backtester.validate_martingale_safety(result['trades'])
print(f"Martingale safe: {safety['safe']}")
print(f"Max consecutive losses: {safety['max_consecutive_losses']}")
print(f"Max leverage used: {safety['max_leverage_used']}x")
```

## Features

### 1. AutoTradeBacktester

Adapter class that wraps `FullBacktester` with auto-trade parameters.

**Key Parameters:**
- `stop_loss_pct`: Stop loss percentage (default: 0.50 = 50%)
- `take_profit_pct`: Take profit percentage (default: 0.05 = 5%)
- `risk_per_trade`: Risk percentage per trade (default: 0.95 = 95%)
- `leverage`: Initial leverage (default: 2)
- `enable_breakeven`: Enable break-even protection (default: True)
- `breakeven_drawdown_pct`: Drawdown threshold for BE (default: 0.30 = 30%)
- `enable_martingale`: Enable Martingale strategy (default: False)
- `martingale_max_steps`: Maximum Martingale steps (default: 4)
- `martingale_max_leverage`: Maximum leverage (default: 16)

**Methods:**
- `backtest_strategy()`: Run backtest with auto-trade parameters
- `validate_martingale_safety()`: Validate Martingale safety metrics

### 2. Break-Even Protection

Simulates moving take profit to entry price when drawdown reaches threshold (30% by default).

**How it works:**
1. Monitors drawdown from entry price
2. When drawdown >= 30%, marks position as "BE moved"
3. If position would have closed with loss, converts to break-even (0% PnL)
4. Tracks BE moves in metrics

**Metrics:**
- `breakeven_moves`: Number of positions where BE was triggered

### 3. Martingale Strategy

Simulates doubling leverage after each loss to recover.

**How it works:**
1. Start with 2x leverage
2. After a loss: double leverage (2x → 4x → 8x → 16x)
3. Maximum 4 steps (configurable)
4. After profit: reset to 2x leverage
5. Tracks total loss to recover

**Safety Features:**
- Maximum steps limit (default: 4)
- Maximum leverage limit (default: 16x)
- Safety validation method
- Disabled by default for safety

**Metrics:**
- `martingale_trades`: Number of trades in Martingale chain
- `max_martingale_step`: Highest Martingale step reached

### 4. Safety Validation

Validates that Martingale strategy didn't exceed safety limits.

```python
safety = backtester.validate_martingale_safety(trades)

# Returns:
{
    "safe": bool,                      # Overall safety
    "max_consecutive_losses": int,     # Longest losing streak
    "max_leverage_used": int,          # Highest leverage used
    "exceeded_max_steps": bool,        # Exceeded step limit?
    "exceeded_max_leverage": bool      # Exceeded leverage limit?
}
```

## Integration with Existing Backtester

This module leverages the existing `modules/backtester/core/backtester.py`:

1. **Uses FullBacktester** as the base engine
2. **Applies auto-trade parameters** (SL, TP, leverage, risk)
3. **Post-processes trades** to add BE and Martingale simulation
4. **Recalculates metrics** with auto-trade adjustments
5. **Adds auto-trade specific metrics** (BE moves, Martingale stats)

**Compatibility:**
- All existing backtester features are preserved
- Uses `single_signal` mode (highest confidence signal)
- Maintains compatibility with existing infrastructure

## Testing

Run comprehensive tests:

```bash
# Run Phase 6.5 test script
python -m modules.auto_trade.test_backtest_phase6
```

This will run:
1. Basic backtest without Martingale
2. Martingale backtest with safety validation
3. Display comprehensive results

## Metrics Reference

### Standard Metrics (from FullBacktester)

- `win_rate`: Percentage of winning trades
- `num_trades`: Total number of trades
- `total_return`: Total return percentage
- `sharpe_ratio`: Risk-adjusted return
- `max_drawdown`: Maximum drawdown
- `profit_factor`: Total profit / Total loss
- `avg_win`: Average winning trade percentage
- `avg_loss`: Average losing trade percentage

### Auto-Trade Specific Metrics

- `leverage_used`: Leverage used (2x initial)
- `breakeven_moves`: Number of BE protections triggered
- `martingale_trades`: Trades in Martingale chain
- `max_martingale_step`: Highest Martingale step (0-4)

## Safety Recommendations

### Break-Even Protection

✅ **Safe to enable by default**
- Limits downside risk
- Protects capital during drawdowns
- Minimal negative impact on winning trades

### Martingale Strategy

⚠️ **Use with extreme caution**
- **Disabled by default** for safety
- Can lead to catastrophic losses if limits exceeded
- **Always validate safety** after backtest
- Recommended settings:
  - Max steps: 3-4
  - Max leverage: 16x
  - Only enable after thorough testing

**When to use Martingale:**
- High win rate strategies (>60%)
- Sufficient capital buffer
- Strict stop-loss discipline
- Close monitoring

**When NOT to use Martingale:**
- Low capital (<$5000)
- High volatility markets
- Low win rate strategies
- Unattended trading

## Future Enhancements

- [ ] Integration with live signal pipeline
- [ ] Historical signal replay
- [ ] Multi-symbol backtesting
- [ ] Portfolio-level metrics
- [ ] Advanced Martingale variants (Fibonacci, D'Alembert)
- [ ] Risk-adjusted position sizing
- [ ] Correlation-based diversification

## Dependencies

- `modules/backtester`: Base backtesting engine
- `modules/common`: Data fetching and utilities
- `modules/auto_trade/core`: Signal pipeline (for strategy simulator)

## License

Same as main project.
