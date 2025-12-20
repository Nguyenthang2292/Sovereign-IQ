# Backtester Module

Module mô phỏng backtesting cho trading strategies với entry/exit rules và tính toán performance metrics.

## Tổng quan

Module này cung cấp `FullBacktester` class để:
- Simulate trading với entry/exit rules dựa trên signals
- Track trades và PnL
- Tính toán performance metrics (win rate, Sharpe ratio, max drawdown, profit factor, etc.)
- Hỗ trợ nhiều signal calculation modes (majority vote hoặc single signal)
- Tối ưu hóa performance với Architecture 5 - Hybrid Approach

## Cấu trúc Module

```
modules/backtester/
├── __init__.py
├── main.py                    # CLI entry point
├── core/
│   ├── __init__.py
│   ├── backtester.py          # FullBacktester class
│   ├── signal_calculator.py   # Signal calculation (sequential & parallel)
│   ├── trade_simulator.py    # Trade simulation logic
│   ├── equity_curve.py        # Equity curve calculation
│   ├── metrics.py             # Performance metrics calculation
│   ├── exit_conditions.py    # Exit condition checks (JIT-compiled)
│   ├── parallel_workers.py   # Multiprocessing worker functions
│   └── shared_memory_utils.py # Shared memory utilities for parallel processing
└── README.md
```

## Sử dụng

### Python API

#### Basic Usage

```python
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.backtester import FullBacktester

# Initialize components
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Create backtester with default settings (majority vote mode)
backtester = FullBacktester(
    data_fetcher=data_fetcher,
    stop_loss_pct=0.02,      # 2% stop loss
    take_profit_pct=0.04,    # 4% take profit
    trailing_stop_pct=0.015, # 1.5% trailing stop
    max_hold_periods=100,    # Max 100 periods
    risk_per_trade=0.01,     # 1% risk per trade
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
total_time = result['total_time']

print(f"Win rate: {metrics['win_rate']*100:.2f}%")
print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max drawdown: {metrics['max_drawdown']*100:.2f}%")
print(f"Number of trades: {metrics['num_trades']}")
print(f"Total time: {total_time:.2f} seconds")
```

#### Single Signal Mode (Highest Confidence)

```python
# Create backtester with single signal mode
backtester = FullBacktester(
    data_fetcher=data_fetcher,
    signal_mode='single_signal',  # Use highest confidence instead of majority vote
)

# Run backtest - signal_type is ignored in single_signal mode
result = backtester.backtest(
    symbol="BTC/USDT",
    timeframe="1h",
    lookback=2160,
    signal_type="LONG",  # Ignored in single_signal mode
    initial_capital=10000.0,
)
```

#### Custom Indicators

```python
# Create backtester with custom enabled indicators
backtester = FullBacktester(
    data_fetcher=data_fetcher,
    enabled_indicators=['range_oscillator', 'spc', 'xgboost'],  # Only use these indicators
    use_confidence_weighting=True,  # Weight votes by confidence
    min_indicators_agreement=2,     # Require at least 2 indicators to agree
)
```

### CLI Usage

```bash
# Run with default settings (single signal mode, precomputed calculation)
python -m modules.backtester.main

# Run with single signal mode
python -m modules.backtester.main --signal-mode single_signal

# Run with majority vote mode (explicit)
python -m modules.backtester.main --signal-mode majority_vote

# Run with incremental calculation mode (Position-Aware Skipping)
python -m modules.backtester.main --signal-calculation-mode incremental

# Combine signal mode and calculation mode
python -m modules.backtester.main --signal-mode single_signal --signal-calculation-mode incremental
```

## Features

### Signal Calculation Modes

#### 1. Majority Vote Mode (Default)

- **Mode**: `signal_mode='majority_vote'`
- **Logic**: Kết hợp signals từ nhiều indicators bằng majority vote
- **Requirements**: 
  - Cần ít nhất `min_indicators_agreement` indicators đồng ý (default: 3)
  - Signal phải khớp với `signal_type` (LONG hoặc SHORT)
- **Use Case**: Khi muốn đảm bảo có sự đồng thuận từ nhiều indicators

```python
backtester = FullBacktester(
    data_fetcher=data_fetcher,
    signal_mode='majority_vote',
    min_indicators_agreement=3,  # At least 3 indicators must agree
)
```

#### 2. Single Signal Mode (Highest Confidence)

- **Mode**: `signal_mode='single_signal'`
- **Logic**: Chọn signal có confidence cao nhất từ tất cả indicators
- **Requirements**: 
  - Không cần majority vote
  - Không filter theo `signal_type` (có thể là LONG hoặc SHORT)
  - Nếu có nhiều signals cùng confidence, ưu tiên LONG trước SHORT
- **Use Case**: Khi muốn trade bất kỳ signal nào có confidence cao nhất

```python
backtester = FullBacktester(
    data_fetcher=data_fetcher,
    signal_mode='single_signal',
)
```

#### Signal Calculation Approaches

##### 1. Precomputed Mode (Default)

- **Mode**: `signal_calculation_mode='precomputed'`
- **Logic**: Tính tất cả signals cho toàn bộ DataFrame trước, sau đó simulate trades
- **Lợi ích**: 
  - Dễ debug và analyze signals
  - Có thể xem tất cả signals trước khi simulate
  - Phù hợp cho analysis và research
- **Use Case**: Khi cần analyze signals hoặc khi positions thường được close nhanh

```python
backtester = FullBacktester(
    data_fetcher=data_fetcher,
    signal_calculation_mode='precomputed',  # Default
)
```

##### 2. Incremental Mode (Position-Aware Skipping)

- **Mode**: `signal_calculation_mode='incremental'`
- **Logic**: Tính signals từng period và **skip signal calculation khi position đang mở**
- **Lợi ích**: 
  - **Tiết kiệm thời gian**: Không tính signals không cần thiết khi position đang mở
  - **Kết hợp signal calculation và trade simulation**: Trong cùng một loop
  - **Walk-forward semantics**: Vẫn maintain walk-forward semantics (chỉ dùng data đến period hiện tại)
- **Use Case**: Khi positions thường được hold lâu, giúp tiết kiệm computation time đáng kể

```python
backtester = FullBacktester(
    data_fetcher=data_fetcher,
    signal_calculation_mode='incremental',  # Skip when position open
)
```

**How it works**:
1. Loop qua từng period
2. Nếu có position đang mở: Skip signal calculation, chỉ check exit conditions
3. Nếu không có position: Calculate signal cho period này (rolling window)
4. Nếu có signal: Enter position
5. Khi position close: Resume signal calculation cho period tiếp theo

**Performance**: Có thể tiết kiệm 30-50% thời gian khi positions được hold lâu (>10 periods).

### Entry/Exit Rules

- **Entry**: Dựa trên signals (1 cho LONG, -1 cho SHORT, 0 cho no signal)
- **Exit Conditions**:
  - **Stop Loss**: 2% default (configurable)
  - **Take Profit**: 4% default (configurable)
  - **Trailing Stop**: 1.5% default (configurable)
  - **Max Hold Periods**: 100 periods default (configurable)

### Performance Metrics

- **Win Rate**: Tỷ lệ trades thắng
- **Average Win/Loss**: Lợi nhuận/lỗ trung bình
- **Total Return**: Tổng lợi nhuận
- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Drawdown tối đa
- **Profit Factor**: Tỷ lệ tổng lợi nhuận / tổng lỗ
- **Number of Trades**: Số lượng trades

### Architecture 5 - Hybrid Approach (Performance Optimizations)

Module sử dụng Architecture 5 - Hybrid Approach để tối ưu hóa performance:

#### 1. Vectorized Indicator Pre-computation

- **Tính năng**: Tính toán tất cả indicators một lần cho toàn bộ DataFrame
- **Lợi ích**: Nhanh hơn 10-15x so với tính toán từng period
- **Implementation**: Sử dụng vectorized operations từ NumPy/Pandas
- **Status**: Luôn được bật trong implementation mới

```python
# Automatically enabled - no configuration needed
# All indicators are pre-computed once before signal calculation
```

#### 2. Incremental Signal Calculation

- **Tính năng**: Tính signals từ pre-computed data một cách incremental
- **Lợi ích**: Duy trì walk-forward semantics nhưng nhanh hơn nhiều
- **Implementation**: Sử dụng DataFrame views thay vì copy để giảm memory overhead
- **Status**: Luôn được bật trong implementation mới

#### 3. Shared Memory for Parallel Processing

- **Tính năng**: Sử dụng shared memory để truyền DataFrame giữa các processes
- **Lợi ích**: Giảm memory overhead 50-70% so với pickle serialization
- **Requirements**: Python 3.8+ với `multiprocessing.shared_memory`
- **Fallback**: Tự động fallback về pickle nếu shared memory không available

```python
# Automatically enabled if available
# Falls back to pickle if shared memory is not supported
```

#### 4. Parallel Processing

- **Tính năng**: Xử lý signals song song với multiprocessing
- **Lợi ích**: Tăng tốc đáng kể cho datasets lớn (>100 periods)
- **Configuration**: Điều khiển bởi `ENABLE_PARALLEL_PROCESSING` trong config
- **Auto-detection**: Chỉ sử dụng parallel cho datasets >100 periods

```python
# Configure in config/position_sizing.py
ENABLE_PARALLEL_PROCESSING = True  # Enable parallel processing
NUM_WORKERS = None  # Auto-detect CPU count
BATCH_SIZE = None   # Auto-optimize batch size
```

#### 5. Multithreading for Indicators

- **Tính năng**: Tính toán indicators song song với ThreadPoolExecutor
- **Lợi ích**: Tận dụng I/O-bound operations (API calls, file reads)
- **Configuration**: Điều khiển bởi `ENABLE_MULTITHREADING` trong config

```python
# Configure in config/position_sizing.py
ENABLE_MULTITHREADING = True  # Enable multithreading for indicators
```

#### 6. GPU Acceleration

- **Tính năng**: Sử dụng GPU cho ML models (XGBoost)
- **Lợi ích**: Tăng tốc prediction cho ML models
- **Configuration**: Điều khiển bởi `USE_GPU` trong config
- **Requirements**: CUDA-enabled GPU và XGBoost với GPU support

```python
# Configure in config/position_sizing.py
USE_GPU = True  # Enable GPU acceleration if available
```

### Performance Characteristics

- **Small datasets (<100 periods)**: Sequential processing (optimized)
- **Large datasets (>100 periods)**: Parallel processing với shared memory
- **Expected speedup**: 10-15x cho datasets lớn (>1000 periods)
- **Memory reduction**: 50-70% với shared memory so với pickle

## Configuration

### FullBacktester Parameters

```python
backtester = FullBacktester(
    data_fetcher=data_fetcher,
    
    # Risk Management
    stop_loss_pct=0.02,        # Stop loss percentage (default: 2%)
    take_profit_pct=0.04,      # Take profit percentage (default: 4%)
    trailing_stop_pct=0.015,   # Trailing stop percentage (default: 1.5%)
    max_hold_periods=100,      # Maximum periods to hold (default: 100)
    risk_per_trade=0.01,       # Risk per trade for equity curve (default: 1%)
    
    # Signal Calculation
    signal_mode='majority_vote',        # 'majority_vote' or 'single_signal' (default: 'majority_vote')
    signal_calculation_mode='precomputed',  # 'precomputed' or 'incremental' (default: 'precomputed')
    enabled_indicators=None,            # List of enabled indicators (default: all)
    use_confidence_weighting=True,      # Weight votes by confidence (default: True)
    min_indicators_agreement=3,        # Minimum indicators that must agree (default: 3)
)
```

### Config File Settings

Các settings trong `config/position_sizing.py`:

```python
# Parallel Processing
ENABLE_PARALLEL_PROCESSING = True  # Enable parallel processing
NUM_WORKERS = None                  # Number of workers (None = auto-detect)
BATCH_SIZE = None                   # Batch size (None = auto-optimize)
OPTIMIZE_BATCH_SIZE = True          # Auto-optimize batch size

# Multithreading
ENABLE_MULTITHREADING = True        # Enable multithreading for indicators

# GPU
USE_GPU = False                     # Enable GPU acceleration

# Performance Logging
LOG_PERFORMANCE_METRICS = False     # Log performance metrics
ENABLE_PERFORMANCE_PROFILING = False # Enable performance profiling

# Caching
CLEAR_CACHE_ON_COMPLETE = False     # Clear cache after backtest completes
```

## Integration

### Position Sizing

Module này được sử dụng bởi `modules.position_sizing` để tính toán position sizing:

```python
from modules.position_sizing.core.position_sizer import PositionSizer

position_sizer = PositionSizer(data_fetcher=data_fetcher)

# Position sizing automatically uses cumulative performance from equity curve
result = position_sizer.calculate_position_size(
    symbol="BTC/USDT",
    account_balance=10000.0,
    signal_type="LONG",
)

# Result includes cumulative_performance_multiplier
print(f"Position size: {result['position_size_usdt']:.2f} USDT")
print(f"Cumulative performance multiplier: {result['cumulative_performance_multiplier']:.3f}")
```

### Standalone Usage

Module có thể được sử dụng độc lập cho bất kỳ backtesting nào:

```python
# Use with custom signals
import pandas as pd

# Create custom signals
signals = pd.Series([1, 0, -1, 0, 1], index=df.index)

# Use with FullBacktester (requires modification to accept custom signals)
# Or use trade_simulator directly
from modules.backtester.core.trade_simulator import simulate_trades

trades = simulate_trades(
    df=df,
    signals=signals,
    signal_type="LONG",
    initial_capital=10000.0,
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
    trailing_stop_pct=0.015,
    max_hold_periods=100,
)
```

## Dependencies

### Core Dependencies

- `pandas`, `numpy`: Data manipulation và vectorized operations
- `modules.common.core.data_fetcher`: Fetch OHLCV data
- `modules.common.quantitative_metrics.risk`: Sharpe ratio, max drawdown calculation
- `config.position_sizing`: Default backtest parameters

### Optional Dependencies

- `numba`: JIT compilation cho exit conditions (optional, có fallback)
- `multiprocessing.shared_memory`: Shared memory cho parallel processing (Python 3.8+)

### Module Dependencies

- `modules.position_sizing.core.hybrid_signal_calculator`: Signal calculation
- `modules.common.core.indicator_engine`: Indicator calculation
- `modules.range_oscillator`: Range Oscillator signals
- `modules.simplified_percentile_clustering`: SPC signals
- `modules.xgboost`: XGBoost signals
- `modules.hmm`: HMM signals
- `modules.random_forest`: Random Forest signals

## Examples

### Example 1: Basic Backtest

```python
from modules.backtester import FullBacktester
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager

# Initialize
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)
backtester = FullBacktester(data_fetcher=data_fetcher)

# Run backtest
result = backtester.backtest(
    symbol="BTC/USDT",
    timeframe="1h",
    lookback=2160,
    signal_type="LONG",
    initial_capital=10000.0,
)

# Print results
print(f"Win rate: {result['metrics']['win_rate']*100:.2f}%")
print(f"Total return: {result['metrics']['total_return']*100:.2f}%")
print(f"Number of trades: {result['metrics']['num_trades']}")
```

### Example 2: Single Signal Mode

```python
# Create backtester with single signal mode
backtester = FullBacktester(
    data_fetcher=data_fetcher,
    signal_mode='single_signal',
)

# Run backtest
result = backtester.backtest(
    symbol="ETH/USDT",
    timeframe="4h",
    lookback=500,
    signal_type="LONG",  # Ignored in single_signal mode
    initial_capital=5000.0,
)
```

### Example 3: Custom Configuration

```python
# Create backtester with custom settings
backtester = FullBacktester(
    data_fetcher=data_fetcher,
    stop_loss_pct=0.03,      # 3% stop loss
    take_profit_pct=0.06,    # 6% take profit
    trailing_stop_pct=0.02,  # 2% trailing stop
    max_hold_periods=50,     # Max 50 periods
    risk_per_trade=0.02,     # 2% risk per trade
    enabled_indicators=['range_oscillator', 'spc'],  # Only use these
    min_indicators_agreement=2,  # Require 2 indicators to agree
)
```

## Testing

Run tests với pytest:

```bash
# Run all backtester tests
pytest tests/backtester/ -v

# Run specific test file
pytest tests/backtester/test_single_signal_backtester.py -v

# Run with coverage
pytest tests/backtester/ --cov=modules.backtester --cov-report=html
```

## Performance Tips

1. **Use parallel processing** cho datasets lớn (>100 periods)
2. **Enable shared memory** nếu có Python 3.8+ để giảm memory usage
3. **Disable caching** nếu memory bị hạn chế (`CLEAR_CACHE_ON_COMPLETE = True`)
4. **Adjust batch size** nếu parallel processing chậm (tăng `BATCH_SIZE`)
5. **Use single_signal mode** nếu muốn nhiều signals hơn (ít strict hơn majority vote)

## Troubleshooting

### Issue: Backtest quá chậm

**Solutions**:
- Enable parallel processing: `ENABLE_PARALLEL_PROCESSING = True`
- Giảm lookback period nếu không cần thiết
- Disable indicators không cần thiết: `enabled_indicators=['range_oscillator']`

### Issue: Memory errors

**Solutions**:
- Enable shared memory (Python 3.8+)
- Clear cache after completion: `CLEAR_CACHE_ON_COMPLETE = True`
- Giảm batch size: `BATCH_SIZE = 10`

### Issue: Không có signals

**Solutions**:
- Giảm `min_indicators_agreement` (ví dụ: từ 3 xuống 2)
- Thử `signal_mode='single_signal'` thay vì `majority_vote`
- Kiểm tra indicators có được enable không

## License

Same as main project.
