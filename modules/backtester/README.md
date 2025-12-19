# Backtester Module

Module mô phỏng backtesting cho trading strategies với entry/exit rules và tính toán performance metrics.

## Tổng quan

Module này cung cấp `FullBacktester` class để:
- Simulate trading với entry/exit rules dựa trên signals
- Track trades và PnL
- Tính toán performance metrics (win rate, Sharpe ratio, max drawdown, profit factor, etc.)

## Cấu trúc Module

```
modules/backtester/
├── __init__.py
├── core/
│   ├── __init__.py
│   └── backtester.py    # FullBacktester class
└── README.md
```

## Sử dụng

### Python API

```python
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.backtester import FullBacktester

# Initialize components
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Create backtester
backtester = FullBacktester(
    data_fetcher=data_fetcher,
    stop_loss_pct=0.02,      # 2% stop loss
    take_profit_pct=0.04,    # 4% take profit
    trailing_stop_pct=0.015, # 1.5% trailing stop
    max_hold_periods=100,    # Max 100 periods
)

# Run backtest
result = backtester.backtest(
    symbol="BTC/USDT",
    timeframe="1h",
    lookback=2160,  # 90 days * 24 hours
    signal_type="LONG",
    initial_capital=10000.0,
)

# Access results
trades = result['trades']
equity_curve = result['equity_curve']
metrics = result['metrics']

print(f"Win rate: {metrics['win_rate']*100:.2f}%")
print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max drawdown: {metrics['max_drawdown']*100:.2f}%")
print(f"Number of trades: {metrics['num_trades']}")
```

## Features

### Entry/Exit Rules

- **Entry**: Dựa trên signals (1 cho LONG, -1 cho SHORT, 0 cho no signal)
- **Exit Conditions**:
  - Stop Loss: 2% default
  - Take Profit: 4% default
  - Trailing Stop: 1.5% default
  - Max Hold Periods: 100 periods default

### Performance Metrics

- **Win Rate**: Tỷ lệ trades thắng
- **Average Win/Loss**: Lợi nhuận/lỗ trung bình
- **Total Return**: Tổng lợi nhuận
- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Drawdown tối đa
- **Profit Factor**: Tỷ lệ tổng lợi nhuận / tổng lỗ
- **Number of Trades**: Số lượng trades

## Configuration

Default parameters có thể được override khi khởi tạo:

```python
backtester = FullBacktester(
    data_fetcher=data_fetcher,
    stop_loss_pct=0.02,        # Custom stop loss
    take_profit_pct=0.05,      # Custom take profit
    trailing_stop_pct=0.02,    # Custom trailing stop
    max_hold_periods=50,       # Custom max hold
)
```

## Integration

Module này được sử dụng bởi:
- `modules.position_sizing`: Position sizing calculation
- Có thể được sử dụng độc lập cho bất kỳ backtesting nào

## Dependencies

- `pandas`, `numpy`: Data manipulation
- `modules.common.core.data_fetcher`: Fetch OHLCV data
- `modules.common.quantitative_metrics.risk`: Sharpe ratio, max drawdown calculation
- `config.position_sizing`: Default backtest parameters

## License

Same as main project.

