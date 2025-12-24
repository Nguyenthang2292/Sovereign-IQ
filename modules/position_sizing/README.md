# Position Sizing Module

Module tính toán position sizing tối ưu sử dụng **Bayesian Kelly Criterion** dựa trên kết quả từ `main_hybrid.py` hoặc `main_voting.py`.

## Tổng quan

Module này cung cấp:
- **Bayesian Kelly Criterion**: Tính toán position size tối ưu dựa trên historical performance với confidence intervals
- **Full Backtest**: Simulate trading với entry/exit rules để tính performance metrics
- **Cumulative Performance Adjustment**: Điều chỉnh position size dựa trên cumulative performance của strategy

## Workflow

```
1. Load Symbols từ main_hybrid.py hoặc main_voting.py
2. Input Account Balance
3. For each Symbol:
   a. Load Historical Data
   b. Run Full Backtest
   c. Calculate Performance Metrics
   d. Calculate Bayesian Kelly Fraction
   e. Adjust for Cumulative Performance
   f. Calculate Position Size
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
print(f"Adjusted Kelly: {result['adjusted_kelly_fraction']:.4f}")

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
│   ├── position_sizer.py   # Main orchestrator
│   ├── hybrid_signal_calculator.py  # Hybrid signal calculation
│   ├── cache_manager.py     # Cache management mixin
│   ├── signal_combiner.py   # Signal combination mixin
│   ├── indicator_calculators.py  # Individual indicator calculators mixin
│   └── batch_calculators.py  # Batch/vectorized calculators mixin
├── utils/
│   ├── __init__.py
│   ├── data_loader.py       # Load symbols từ hybrid/voting results
│   └── metrics.py           # Performance metrics helpers
└── README.md

**Lưu ý**: `FullBacktester` được cung cấp bởi module `modules/backtester` (xem phần Components bên dưới).
```

## Components

### 1. FullBacktester (Optional External Dependency)

Module này sử dụng `FullBacktester` từ `modules/backtester` để thực hiện backtesting và tính toán performance metrics. `FullBacktester` là một module độc lập và có thể được sử dụng riêng biệt.

**Import và Sử dụng cơ bản**:

```python
from modules.backtester import FullBacktester
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager

# Initialize components
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Create backtester instance
backtester = FullBacktester(
    data_fetcher=data_fetcher,
    signal_mode='majority_vote',  # hoặc 'single_signal'
    signal_calculation_mode='precomputed',  # hoặc 'incremental'
)

# Run backtest
result = backtester.backtest(
    symbol="BTC/USDT",
    timeframe="1h",
    lookback=2160,
    signal_type="LONG",
    initial_capital=10000.0,
)
```

**Cấu hình tối thiểu**:
- `data_fetcher`: DataFetcher instance để fetch historical data
- `signal_mode`: 'majority_vote' (default) hoặc 'single_signal'
- `signal_calculation_mode`: 'precomputed' (default) hoặc 'incremental'

**Lưu ý về Module Boundaries**:
- `FullBacktester` được cung cấp bởi `modules/backtester` và không phải là phần của module `position_sizing`
- Module `position_sizing` sử dụng `FullBacktester` như một dependency để tính toán performance metrics
- Để xem chi tiết về API, configuration options, và các ví dụ đầy đủ, vui lòng tham khảo [modules/backtester/README.md](../backtester/README.md)

**Cài đặt/Import**:
- `FullBacktester` được import từ `modules.backtester`
- Không cần cài đặt thêm dependencies nếu đã cài đặt đầy đủ dependencies của project
- Đảm bảo module `backtester` có sẵn trong project structure

### 2. BayesianKellyCalculator

Tính toán Kelly fraction với:
- Beta distribution prior cho win rate
- Posterior distribution với historical data
- Confidence intervals (95% default)
- Fractional Kelly (25% of full Kelly default) để reduce risk

### 3. PositionSizer

Main orchestrator kết hợp:
- Backtesting
- Kelly calculation
- Cumulative performance adjustments
- Portfolio-level constraints

### 4. HybridSignalCalculator

Tính toán hybrid signals bằng cách kết hợp nhiều indicators:
- Range Oscillator
- SPC (Simplified Percentile Clustering)
- XGBoost
- HMM
- Random Forest

Sử dụng majority vote hoặc weighted voting để kết hợp signals.

## Configuration

Các parameters có thể cấu hình trong `config/position_sizing.py`:

- `DEFAULT_LOOKBACK_DAYS = 15`: Số ngày lookback cho backtesting (default: 15)
- `DEFAULT_TIMEFRAME = "1h"`: Timeframe cho backtesting
- `DEFAULT_FRACTIONAL_KELLY = 0.25`: Sử dụng 25% của full Kelly
- `DEFAULT_MAX_POSITION_SIZE = 1.0`: Tối đa 100% account per symbol
- `DEFAULT_MIN_POSITION_SIZE = 0.3`: Tối thiểu 30% account per symbol (nếu signal hợp lệ)
- `DEFAULT_MAX_PORTFOLIO_EXPOSURE = 0.5`: Tối đa 50% account cho toàn bộ portfolio

### Conflict Resolution giữa Min Position Size và Max Portfolio Exposure

**Quan trọng**: `DEFAULT_MIN_POSITION_SIZE` là một **soft suggestion**, không phải hard floor. Nó có thể bị override để đảm bảo tổng exposure của portfolio không vượt quá `DEFAULT_MAX_PORTFOLIO_EXPOSURE`.

**Thuật toán giải quyết xung đột** (thứ tự ưu tiên):

1. **Bước 1 - Áp dụng min/max per symbol**: Mỗi position size được tính toán từ Kelly fraction, sau đó được clamp trong khoảng `[min_position_size, max_position_size]` (trừ khi Kelly fraction = 0, lúc đó position size = 0).

2. **Bước 2 - Kiểm tra tổng exposure**: Nếu tổng exposure của tất cả positions vượt quá `max_portfolio_exposure`, hệ thống sẽ **normalize tỷ lệ** tất cả positions bằng cách nhân với hệ số `normalization_factor = max_portfolio_exposure / total_exposure`.

3. **Kết quả**: Sau khi normalize, các position size có thể **nhỏ hơn** `min_position_size` để đảm bảo tổng exposure ≤ `max_portfolio_exposure`. Điều này có nghĩa là `max_portfolio_exposure` có **ưu tiên cao hơn** `min_position_size`.

**Lưu ý về giá trị mặc định**: `DEFAULT_MIN_POSITION_SIZE = 0.3` (30%) là một giá trị **khá cao và hiếm khi được sử dụng** trong thực tế. Với `DEFAULT_MAX_PORTFOLIO_EXPOSURE = 0.5` (50%), giá trị này chỉ cho phép tối đa 1 position (30% < 50%). Nếu có 2 positions, mỗi position tối thiểu 30% sẽ tạo ra tổng 60%, vượt quá giới hạn 50%, và sau khi normalize sẽ vi phạm minimum 30%.

**Khuyến nghị**: 
- Đối với portfolio đa dạng (nhiều symbols), nên sử dụng `DEFAULT_MIN_POSITION_SIZE = 0.05` đến `0.1` (5-10% per symbol).
- Giá trị 0.3 chỉ phù hợp cho trường hợp **single-position strategy** hoặc khi muốn tập trung vào một vài positions có confidence cao.
- Giá trị này có thể được cấu hình trong `config/position_sizing.py` tùy theo chiến lược trading.

**Cumulative Performance Adjustment**: Position size được điều chỉnh tự động dựa trên cumulative performance của strategy:

**Công thức điều chỉnh**:

```text
multiplier = clamp(1 + k * clamp(cumulativePerformance, -0.5, 0.5), 0.5, 1.5)
```

**Tham số**:
- `k`: Hệ số nhạy cảm (sensitivity factor), mặc định `k = 1.0`
- `cumulativePerformance`: Hiệu suất tích lũy, được tính bằng `(final_equity - initial_capital) / initial_capital`
  - **Giá trị mặc định**: Trên lần chạy backtest đầu tiên, `cumulativePerformance` mặc định là `0.0` (không có điều chỉnh) trừ khi người dùng cung cấp giá trị `initialPerformance` (hoặc `bootstrap`) để khởi tạo metric.
  - Đơn vị: Tỷ lệ phần trăm dưới dạng số thập phân (ví dụ: 0.1 = +10%, -0.1 = -10%)
  - Được clamp trong khoảng `[-0.5, 0.5]` trước khi áp dụng công thức (giới hạn tác động tối đa)
  - **Khởi tạo với giá trị tùy chỉnh**: Để seed metric với giá trị khác 0 (bootstrapping), người dùng có thể cung cấp tham số `initialPerformance` (hoặc key cấu hình tương ứng) khi khởi tạo position sizer. Giá trị này sẽ được sử dụng làm điểm bắt đầu cho cumulative performance và vẫn sẽ được clamp trong khoảng `[-0.5, 0.5]` trước khi áp dụng vào công thức điều chỉnh.
- `multiplier`: Hệ số nhân cuối cùng, được clamp trong khoảng `[0.5, 1.5]`

**Ví dụ tính toán**:
- **Performance dương**: Nếu `cumulativePerformance = 0.3` (30% lợi nhuận):
  - `multiplier = clamp(1 + 1.0 * clamp(0.3, -0.5, 0.5), 0.5, 1.5) = clamp(1 + 0.3, 0.5, 1.5) = 1.3x`
- **Performance âm**: Nếu `cumulativePerformance = -0.2` (-20% lỗ):
  - `multiplier = clamp(1 + 1.0 * clamp(-0.2, -0.5, 0.5), 0.5, 1.5) = clamp(1 - 0.2, 0.5, 1.5) = 0.8x`

**Edge cases và hành vi clamping**:
- **Clamping `cumulativePerformance`**: Giá trị được giới hạn trong `[-0.5, 0.5]` để tránh điều chỉnh quá mức:
  - Performance > 50%: Chỉ áp dụng tối đa 0.5 → multiplier tối đa 1.5x
  - Performance < -50%: Chỉ áp dụng tối thiểu -0.5 → multiplier tối thiểu 0.5x
- **Clamping `multiplier` cuối cùng**: Đảm bảo hệ số nhân luôn nằm trong `[0.5, 1.5]`:
  - Nếu tính toán ra < 0.5: Clamp về 0.5x (giảm tối đa 50%)
  - Nếu tính toán ra > 1.5: Clamp về 1.5x (tăng tối đa 50%)
- **Reset conditions**: Multiplier được tính lại mỗi lần chạy backtest dựa trên equity curve mới nhất, không có cơ chế reset tự động (mỗi backtest là độc lập)

## Output Format

Kết quả được hiển thị dưới dạng table với các columns:

- **Symbol**: Trading pair
- **Signal**: LONG hoặc SHORT
- **Position Size (USDT)**: Số tiền USDT nên đầu tư
- **Position Size (%)**: Phần trăm của account balance
- **Kelly Fraction**: Kelly fraction trước khi điều chỉnh
- **Adjusted Kelly**: Kelly fraction sau khi điều chỉnh cumulative performance
- **Cumulative Performance Multiplier**: Hệ số điều chỉnh dựa trên performance

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

### Phương thức Load dữ liệu

**Khuyến nghị**: Sử dụng **DataFrame trực tiếp** cho in-process pipelines (truyền DataFrame từ analyzer results). Sử dụng **CSV/JSON** cho offline interchange hoặc khi cần lưu trữ kết quả để sử dụng sau.

Có thể load symbols từ:
1. **DataFrame trực tiếp** (khuyến nghị cho in-process pipelines): Truyền DataFrame từ analyzer results
2. **CSV/JSON file** (khuyến nghị cho offline interchange): Load từ file được export từ analyzer
3. **Manual input**: Nhập symbols qua command line

### Cấu trúc dữ liệu yêu cầu

#### Cột/Field bắt buộc

- **`symbol`** (string, required): Trading pair symbol, format `BASE/QUOTE` (ví dụ: `BTC/USDT`, `ETH/USDT`)

#### Cột/Field tùy chọn (có giá trị mặc định)

- **`signal`** (int/float, default: `0`): Trading signal
  - `1` hoặc `"LONG"` hoặc `"BUY"`: Long position
  - `-1` hoặc `"SHORT"` hoặc `"SELL"`: Short position
  - `0`: No signal
- **`price`** (float, default: `0.0`): Current price của symbol
- **`exchange`** (string, default: `"binance"`): Exchange name (lowercase)

#### Cột/Field tùy chọn (được preserve nếu có)

Các field sau sẽ được preserve nếu có trong dữ liệu input:

**Danh sách fields và kiểu dữ liệu mong đợi:**
- `trend` (string): Xu hướng thị trường, ví dụ: "UP", "DOWN", "SIDEWAYS"
- `weighted_score` (float): Điểm số có trọng số, giá trị trong khoảng [0.0, 1.0] hoặc tương đương
- `cumulative_vote` (int): Tổng số vote tích lũy từ các indicators
- `voting_breakdown` (JSON/object): Chi tiết breakdown của voting, format dictionary/JSON object
- `osc_signal` (int hoặc enum): Range Oscillator signal, giá trị: -1 (SHORT), 0 (NEUTRAL), 1 (LONG)
- `osc_confidence` (float): Độ tin cậy của Range Oscillator, giá trị trong khoảng [0.0, 1.0]
- `spc_signal` (int hoặc enum): SPC signal, giá trị: -1 (SHORT), 0 (NEUTRAL), 1 (LONG)
- `spc_confidence` (float): Độ tin cậy của SPC, giá trị trong khoảng [0.0, 1.0]
- `xgboost_signal` (int hoặc enum): XGBoost signal, giá trị: -1 (SHORT), 0 (NEUTRAL), 1 (LONG)
- `xgboost_confidence` (float): Độ tin cậy của XGBoost, giá trị trong khoảng [0.0, 1.0]
- `hmm_signal` (int hoặc enum): HMM signal, giá trị: -1 (SHORT), 0 (NEUTRAL), 1 (LONG)
- `hmm_confidence` (float): Độ tin cậy của HMM, giá trị trong khoảng [0.0, 1.0]
- `random_forest_signal` (int hoặc enum): Random Forest signal, giá trị: -1 (SHORT), 0 (NEUTRAL), 1 (LONG)
- `random_forest_confidence` (float): Độ tin cậy của Random Forest, giá trị trong khoảng [0.0, 1.0]
- `source` (string): Nguồn dữ liệu, ví dụ: "hybrid", "voting", "manual"

**Chính sách giải quyết xung đột (Conflict Resolution Policy):**

**Quy tắc ưu tiên**: Input values được preserve nguyên vẹn và **sẽ không được recalculate hoặc validate** bởi hệ thống. Nếu input values conflict với system-calculated values (ví dụ: khi hệ thống tính toán lại signals trong quá trình backtesting), **input values sẽ được ưu tiên và giữ nguyên**.

**Hành vi cụ thể:**
- Các fields được preserve sẽ được copy trực tiếp từ input data vào output results mà không qua bất kỳ validation hoặc transformation nào
- Nếu input data chứa các fields này, chúng sẽ được giữ nguyên trong output DataFrame
- Nếu input data không chứa các fields này, chúng sẽ không được tạo tự động (không có giá trị mặc định)
- Hệ thống không kiểm tra tính nhất quán giữa preserved fields và system-calculated values (ví dụ: không kiểm tra xem `osc_signal` có khớp với signal được tính toán bởi Range Oscillator hay không)

**Validation và Formatting:**
- **Không có validation tự động**: Các preserved fields không được validate về format, range, hoặc type. Hệ thống giả định input data đã được validate ở upstream (từ hybrid/voting analyzers)
- **Type coercion**: Nếu có type mismatch (ví dụ: string thay vì int cho signal fields), pandas sẽ tự động xử lý theo khả năng của nó, nhưng không có explicit conversion
- **Format expectations**: 
  - Signal fields (`*_signal`) nên là integer (-1, 0, 1) hoặc có thể là string ("LONG", "SHORT", "NEUTRAL")
  - Confidence fields (`*_confidence`) nên là float trong [0.0, 1.0]
  - `voting_breakdown` nên là JSON-serializable object/dictionary

**Xử lý conflicts và logging:**
- **Không có explicit conflict detection**: Hệ thống không tự động phát hiện hoặc cảnh báo về conflicts giữa input values và system-calculated values
- **Logging**: Nếu có lỗi xảy ra trong quá trình preserve fields (ví dụ: type conversion errors), lỗi sẽ được log ở mức WARNING và field đó có thể bị skip hoặc giữ nguyên giá trị gốc
- **Error handling**: Nếu có lỗi nghiêm trọng khi xử lý preserved fields (ví dụ: JSON parsing error cho `voting_breakdown`), lỗi sẽ được log và field đó sẽ được set thành `None` hoặc bị skip, nhưng không làm gián đoạn quá trình tính toán position sizing

### Xử lý lỗi và Validation

Module tự động validate dữ liệu khi load:

1. **Validation khi load DataFrame/CSV** (Mandatory Column Check):
   - Cột `symbol` là **bắt buộc** (required column)
   - Nếu thiếu cột `symbol`, hệ thống sẽ **raise `ValueError`** và **abort** ngay lập tức với thông báo: `"DataFrame missing required columns: ['symbol']"`
   - Các cột khác sẽ được set giá trị mặc định nếu thiếu

2. **Normalization sau khi load** (Post-Load Symbol Normalization):
   - Sau khi dữ liệu đã được load thành công, hệ thống sẽ tự động normalize các giá trị trong cột `symbol`
   - Nếu một giá trị `symbol` thiếu ký tự `/` (ví dụ: `"BTC"`), hệ thống sẽ **tự động append** `"/USDT"` vào cuối (ví dụ: `"BTC"` → `"BTC/USDT"`)
   - Các symbol đã có format đúng (ví dụ: `"BTC/USDT"`, `"ETH/BUSD"`) sẽ được giữ nguyên
   - Validate và normalize `signal` (convert string thành int nếu cần)
   - Symbols không hợp lệ sau khi normalize sẽ được skip với warning log

3. **Error handling**:
   - `FileNotFoundError`: Khi file không tồn tại
   - `ValueError`: Khi thiếu required columns hoặc format không hợp lệ
   - Tất cả errors đều được log và abort process

### CSV/JSON Loading Behavior

**CSV Loading**:
- Sử dụng `pandas.read_csv()` để load, **preserve tất cả columns** từ file
- Sau đó được validate và normalize qua `load_symbols_from_dataframe()`
- Tất cả columns trong CSV sẽ được preserve, chỉ required columns được validate

**JSON Loading**:
- Hỗ trợ nhiều format:
  - List of dictionaries: `[{symbol: "BTC/USDT", ...}, ...]`
  - Dictionary với key `symbols`: `{symbols: [{symbol: "BTC/USDT", ...}, ...]}`
  - Dictionary với key `data`: `{data: [{symbol: "BTC/USDT", ...}, ...]}`
- Sau khi parse, được convert thành DataFrame và xử lý tương tự CSV
- **Preserve tất cả fields** có trong JSON

### Ví dụ Schema

#### CSV Schema

```csv
symbol,signal,price,exchange,trend,weighted_score,osc_signal,spc_signal
BTC/USDT,1,45000.0,binance,UP,0.85,1,1
ETH/USDT,1,2500.0,binance,UP,0.72,1,0
SOL/USDT,-1,100.0,binance,DOWN,0.65,-1,-1
```

**Column types**:
- `symbol`: string (required)
- `signal`: int/float (optional, default: 0)
- `price`: float (optional, default: 0.0)
- `exchange`: string (optional, default: "binance")
- Các columns khác: any type (optional, preserved if present)

#### JSON Schema

**Format 1 - List of objects** (khuyến nghị):
```json
[
  {
    "symbol": "BTC/USDT",
    "signal": 1,
    "price": 45000.0,
    "exchange": "binance",
    "trend": "UP",
    "weighted_score": 0.85,
    "osc_signal": 1,
    "spc_signal": 1
  },
  {
    "symbol": "ETH/USDT",
    "signal": 1,
    "price": 2500.0,
    "exchange": "binance",
    "trend": "UP",
    "weighted_score": 0.72
  }
]
```

**Format 2 - Object with symbols key**:
```json
{
  "symbols": [
    {
      "symbol": "BTC/USDT",
      "signal": 1,
      "price": 45000.0,
      "exchange": "binance",
      "trend": "UP",
      "weighted_score": 0.85,
      "cumulative_vote": 3,
      "voting_breakdown": {
        "osc": 1,
        "spc": 1,
        "xgboost": 1
      },
      "osc_signal": 1,
      "osc_confidence": 0.92,
      "spc_signal": 1,
      "spc_confidence": 0.88,
      "xgboost_signal": 1,
      "xgboost_confidence": 0.85,
      "hmm_signal": 0,
      "hmm_confidence": 0.65,
      "random_forest_signal": 1,
      "random_forest_confidence": 0.78,
      "source": "hybrid"
    },
    {
      "symbol": "ETH/USDT",
      "signal": 1,
      "price": 2500.0,
      "exchange": "binance",
      "trend": "UP",
      "weighted_score": 0.72
    }
  ]
}
```

**Format 3 - Object with data key**:
```json
{
  "data": [
    {
      "symbol": "BTC/USDT",
      "signal": 1,
      "price": 45000.0,
      "exchange": "binance",
      "trend": "UP",
      "weighted_score": 0.85,
      "cumulative_vote": 3,
      "voting_breakdown": {
        "osc": 1,
        "spc": 1,
        "xgboost": 1
      },
      "osc_signal": 1,
      "osc_confidence": 0.92,
      "spc_signal": 1,
      "spc_confidence": 0.88,
      "xgboost_signal": 1,
      "xgboost_confidence": 0.85,
      "hmm_signal": 0,
      "hmm_confidence": 0.65,
      "random_forest_signal": 1,
      "random_forest_confidence": 0.78,
      "source": "voting"
    },
    {
      "symbol": "ETH/USDT",
      "signal": 1,
      "price": 2500.0,
      "exchange": "binance",
      "trend": "UP",
      "weighted_score": 0.72
    }
  ]
}
```

**Lưu ý về các format JSON:**
- Tất cả ba format (Format 1, Format 2, Format 3) đều **hỗ trợ đầy đủ** tất cả các optional fields được liệt kê trong phần "Cột/Field tùy chọn (được preserve nếu có)"
- Các optional fields có thể được bao gồm hoặc bỏ qua tùy ý trong bất kỳ format nào
- Hệ thống sẽ preserve tất cả các fields có trong JSON, bất kể format nào được sử dụng
- Format 1 (List of objects) là format được khuyến nghị vì đơn giản và dễ đọc nhất

**Field types**:
- `symbol`: string (required)
- `signal`: number hoặc string (optional, default: 0)
- `price`: number (optional, default: 0.0)
- `exchange`: string (optional, default: "binance")
- Các fields khác: any type (optional, preserved if present)

## Lưu ý

- Position sizing chỉ là gợi ý, không phải lời khuyên đầu tư
- Luôn sử dụng risk management phù hợp
- Backtest results không đảm bảo future performance
- Kelly Criterion giả định win rate và win/loss ratio ổn định
- Cumulative performance adjustment dựa trên historical data, không đảm bảo future performance

## Dependencies

- `pandas`, `numpy`: Data manipulation
- `scipy`: Beta distribution cho Bayesian Kelly
- DataFetcher: Fetch historical OHLCV data
- Risk metrics module: Sharpe ratio, max drawdown calculation
- Multiple indicator modules: Range Oscillator, SPC, XGBoost, HMM, Random Forest (cho hybrid signals)

## Testing

Run tests:

```bash
python -m pytest tests/position_sizing/ -v
```

## License

Same as main project.

