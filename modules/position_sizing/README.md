# Position Sizing Module

Module tính toán position sizing tối ưu sử dụng **Bayesian Kelly Criterion** và **Regime Switching** dựa trên kết quả từ `main_hybrid.py` hoặc `main_voting.py`.

## Tổng quan

Module này cung cấp:
- **Bayesian Kelly Criterion**: Tính toán position size tối ưu dựa trên historical performance với confidence intervals
- **Regime Switching**: Điều chỉnh position size dựa trên market regime (BULLISH/NEUTRAL/BEARISH) được detect bởi HMM
- **Full Backtest**: Simulate trading với entry/exit rules để tính performance metrics

## Workflow

```
1. Load Symbols từ main_hybrid.py hoặc main_voting.py
2. Input Account Balance
3. For each Symbol:
   a. Detect Current Regime via HMM
   b. Load Historical Data
   c. Run Full Backtest
   d. Calculate Performance Metrics
   e. Calculate Bayesian Kelly Fraction
   f. Adjust for Regime
   g. Calculate Position Size
4. Display Results
```

## Cài đặt và Sử dụng

### Command Line

```bash
# Sử dụng kết quả từ hybrid analyzer
python main_position_sizing.py --source hybrid --account-balance 10000

# Sử dụng kết quả từ voting analyzer
python main_position_sizing.py --source voting --account-balance 10000

# Load từ file CSV/JSON
python main_position_sizing.py --symbols-file results.csv --account-balance 10000

# Manual input symbols
python main_position_sizing.py --symbols "BTC/USDT,ETH/USDT" --account-balance 10000

# Với custom parameters
python main_position_sizing.py \
    --source hybrid \
    --account-balance 10000 \
    --timeframe 1h \
    --lookback-days 90 \
    --max-position-size 0.1 \
    --output results.csv
```

### Interactive Menu

Chạy không có arguments để vào interactive menu:

```bash
python main_position_sizing.py
```

Menu sẽ hỏi:
1. Symbol source (hybrid/voting/file/manual)
2. Account balance
3. Backtest settings (timeframe, lookback days)
4. Position size constraints

### Python API

```python
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager
from modules.position_sizing.core.position_sizer import PositionSizer

# Initialize components
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Create position sizer
position_sizer = PositionSizer(
    data_fetcher=data_fetcher,
    timeframe="1h",
    lookback_days=90,
)

# Calculate position size for a single symbol
result = position_sizer.calculate_position_size(
    symbol="BTC/USDT",
    account_balance=10000.0,
    signal_type="LONG",
)

print(f"Position size: {result['position_size_usdt']:.2f} USDT")
print(f"Kelly fraction: {result['kelly_fraction']:.4f}")
print(f"Regime: {result['regime']}")

# Calculate portfolio allocation
symbols = [
    {'symbol': 'BTC/USDT', 'signal': 1},
    {'symbol': 'ETH/USDT', 'signal': 1},
]

results_df = position_sizer.calculate_portfolio_allocation(
    symbols=symbols,
    account_balance=10000.0,
)
```

## Cấu trúc Module

```
modules/position_sizing/
├── __init__.py
├── cli/
│   ├── __init__.py
│   ├── main.py              # Entry point CLI
│   ├── argument_parser.py   # Parse arguments và interactive prompts
│   └── display.py           # Display results
├── core/
│   ├── __init__.py
│   ├── kelly_calculator.py  # Bayesian Kelly Criterion calculation
│   ├── regime_detector.py   # Regime detection sử dụng HMM
│   └── position_sizer.py   # Main orchestrator
├── utils/
│   ├── __init__.py
│   ├── data_loader.py       # Load symbols từ hybrid/voting results
│   └── metrics.py           # Performance metrics helpers
└── README.md

Note: FullBacktester is now in modules/backtester module (separate module)
```

## Components

### 1. RegimeDetector

Sử dụng HMM module để detect current market regime:
- **BULLISH**: Upward trending market (tăng position size)
- **NEUTRAL**: Sideways/consolidating market (không điều chỉnh)
- **BEARISH**: Downward trending market (giảm position size)

### 2. FullBacktester (from modules.backtester)

Simulate trading với:
- Entry/exit rules dựa trên signals
- Stop loss (2%), Take profit (4%), Trailing stop (1.5%)
- Track trades và PnL
- Calculate performance metrics (win rate, Sharpe ratio, max drawdown)

**Note**: FullBacktester is now a separate module at `modules/backtester` and can be used independently.

### 3. BayesianKellyCalculator

Tính toán Kelly fraction với:
- Beta distribution prior cho win rate
- Posterior distribution với historical data
- Confidence intervals (95% default)
- Fractional Kelly (25% of full Kelly default) để reduce risk

### 4. PositionSizer

Main orchestrator kết hợp:
- Regime detection
- Backtesting
- Kelly calculation
- Regime adjustments
- Portfolio-level constraints

## Configuration

Các parameters có thể cấu hình trong `config/position_sizing.py`:

- `DEFAULT_LOOKBACK_DAYS = 90`: Số ngày lookback cho backtesting
- `DEFAULT_TIMEFRAME = "1h"`: Timeframe cho backtesting
- `DEFAULT_FRACTIONAL_KELLY = 0.25`: Sử dụng 25% của full Kelly
- `DEFAULT_MAX_POSITION_SIZE = 0.1`: Tối đa 10% account per symbol
- `DEFAULT_MIN_POSITION_SIZE = 0.01`: Tối thiểu 1% account per symbol
- `DEFAULT_MAX_PORTFOLIO_EXPOSURE = 0.5`: Tối đa 50% account cho toàn bộ portfolio
- `REGIME_MULTIPLIERS`: Điều chỉnh position size theo regime:
  - `BULLISH: 1.2` (tăng 20%)
  - `NEUTRAL: 1.0` (không điều chỉnh)
  - `BEARISH: 0.8` (giảm 20%)

## Output Format

Kết quả được hiển thị dưới dạng table với các columns:

- **Symbol**: Trading pair
- **Signal**: LONG hoặc SHORT
- **Regime**: BULLISH, NEUTRAL, hoặc BEARISH
- **Position Size (USDT)**: Số tiền USDT nên đầu tư
- **Position Size (%)**: Phần trăm của account balance
- **Kelly Fraction**: Kelly fraction trước khi điều chỉnh regime
- **Adjusted Kelly**: Kelly fraction sau khi điều chỉnh regime

Performance Metrics:
- **Win Rate**: Tỷ lệ thắng
- **Avg Win**: Lợi nhuận trung bình khi thắng
- **Avg Loss**: Lỗ trung bình khi thua
- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Drawdown tối đa
- **Trades**: Số lượng trades trong backtest

## Integration với Hybrid/Voting Analyzers

Module này được thiết kế để hoạt động với kết quả từ:
- `main_hybrid.py`: Sequential filtering + voting approach
- `main_voting.py`: Pure voting system approach

Có thể load symbols từ:
1. DataFrame trực tiếp từ analyzer results
2. CSV/JSON file được export từ analyzer
3. Manual input qua command line

## Lưu ý

- Position sizing chỉ là gợi ý, không phải lời khuyên đầu tư
- Luôn sử dụng risk management phù hợp
- Backtest results không đảm bảo future performance
- Regime detection có thể không chính xác 100%
- Kelly Criterion giả định win rate và win/loss ratio ổn định

## Dependencies

- `pandas`, `numpy`: Data manipulation
- `scipy`: Beta distribution cho Bayesian Kelly
- HMM module: Regime detection
- DataFetcher: Fetch historical OHLCV data
- Risk metrics module: Sharpe ratio, max drawdown calculation

## Testing

Run tests:

```bash
python -m pytest tests/position_sizing/ -v
```

## License

Same as main project.

