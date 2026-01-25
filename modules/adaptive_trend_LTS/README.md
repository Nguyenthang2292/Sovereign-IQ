# Adaptive Trend Classification LTS (ATC LTS)

**Long-term support version with Rust-accelerated backend, GPU/CPU optimization, and automatic memory management**

Module Adaptive Trend Classification LTS là phiên bản ổn định của ATC với:

- **Rust backend**: Equity, KAMA, MAs (EMA/WMA/DEMA/LSMA/HMA), signal persistence chạy trên Rust khi đã build; fallback Numba nếu chưa build.
- **Parallel computing**: Multi-processing + multi-threading với auto-detection CPU/RAM
- **GPU acceleration**: Tự động detect và sử dụng GPU (CUDA/OpenCL) nếu có
- **Memory management**: Automatic cleanup, monitoring và prevention memory leaks
- **Numba JIT**: Fallback cho MA calculations khi Rust chưa có
- **Caching**: Intelligent caching cho MA results
- **NumPy optimization**: Pre-allocated arrays và NumPy operations thay vì Pandas

Module cung cấp hệ thống phân tích xu hướng thích ứng sử dụng nhiều loại Moving Averages với adaptive weighting dựa trên equity curves.

## Tổng quan

ATC là một hệ thống phân loại xu hướng thích ứng sử dụng:

- **6 loại Moving Averages**: EMA, HMA, WMA, DEMA, LSMA, KAMA
- **2-layer architecture**:
  - Layer 1: Tính signals cho từng MA type dựa trên equity curves
  - Layer 2: Tính weights và kết hợp tất cả để tạo Average_Signal
- **Adaptive weighting**: Sử dụng equity curves để tự động điều chỉnh trọng số của từng MA
- **Robustness modes**: "Narrow", "Medium", "Wide" để điều chỉnh độ nhạy

## Cấu trúc Module

```text
adaptive_trend_LTS/
├── __init__.py              # Module exports
├── README.md                # Tài liệu này
├── core/
│   ├── rust_backend.py      # Rust extension wrapper (equity, KAMA, MAs, persistence)
│   ├── compute_atc_signals/ # ATC signals (Rust-accelerated when built)
│   ├── compute_moving_averages/  # MA với Rust hoặc Numba fallback
│   ├── compute_equity/      # Equity curves
│   ├── process_layer1/      # Layer 1 processing
│   ├── signal_detection/    # Signal detection
│   ├── scanner/             # Multi-symbol scanning
│   └── ...
├── rust_extensions/         # Rust crate (PyO3); xem rust_extensions/README.md
├── cli/                     # CLI (argument_parser, display, main, ...)
├── docs/                    # Tài liệu chi tiết (setting_guides, phase tasks, ...)
└── utils/                   # config, cache_manager, rate_of_change, ...
```

**Tài liệu:** Tham khảo đầy đủ parameters, presets và troubleshooting: [docs/setting_guides.md](docs/setting_guides.md).

## Cách hoạt động

### Layer 1: Individual MA Signals

Với mỗi loại MA (EMA, HMA, WMA, DEMA, LSMA, KAMA):

1. Tính toán 9 MAs với các độ dài khác nhau (base length ± offsets dựa trên robustness)
2. Tính signals cho từng MA dựa trên price/MA crossovers
3. Tính equity curves cho từng signal sử dụng exponential growth
4. Weighted average của 9 signals dựa trên equity curves → Layer 1 signal cho MA type đó

### Layer 2: Combined Signal

1. Tính weights cho từng MA type dựa trên Layer 1 signals
2. Weighted average của tất cả Layer 1 signals → **Average_Signal** (final output)

### Equity Curves

Equity curves mô phỏng performance của trading strategy:

- Sử dụng exponential growth factor (La) và decay rate (De)
- Equity cao hơn → weight cao hơn → MA đó có ảnh hưởng lớn hơn
- Adaptive: Tự động điều chỉnh weights dựa trên performance

## Setup (Rust extensions)

Để bật Rust backend (khuyến nghị):

```bash
cd modules/adaptive_trend_LTS/rust_extensions
maturin develop --release
```

Hoặc từ thư mục gốc project: `.\build_rust.bat` (Windows) / `.\build_rust.ps1`.

**Yêu cầu:** [Rust](https://rustup.rs/), [maturin](https://www.maturin.rs/) (`pip install maturin`). Chi tiết và xử lý lỗi: [docs/phase3_task.md#prerequisites--setup](docs/phase3_task.md#prerequisites--setup).

## Sử dụng

### Ví dụ cơ bản

Các ví dụ dưới dùng `modules.adaptive_trend_enhance`; có thể thay bằng `modules.adaptive_trend_LTS` (cùng API, dùng Rust backend khi đã build).

```python
import pandas as pd
from modules.adaptive_trend_enhance import compute_atc_signals, ATCConfig

# Chuẩn bị dữ liệu
prices = pd.Series([...])  # Close prices

# Cấu hình
config = ATCConfig(
    ema_len=28,
    hma_len=28,
    wma_len=28,
    dema_len=28,
    lsma_len=28,
    kama_len=28,
    robustness="Medium",  # "Narrow", "Medium", "Wide"
    lambda_param=0.02,     # Growth rate
    decay=0.03,            # Decay rate
    cutout=0,              # Bars to skip
)

# Tính toán ATC signals
results = compute_atc_signals(
    prices=prices,
    ema_len=config.ema_len,
    hull_len=config.hma_len,
    wma_len=config.wma_len,
    dema_len=config.dema_len,
    lsma_len=config.lsma_len,
    kama_len=config.kama_len,
    robustness=config.robustness,
    La=config.lambda_param,
    De=config.decay,
    cutout=config.cutout,
)

# Kết quả
average_signal = results["Average_Signal"]  # Final combined signal
ema_signal = results["EMA_Signal"]         # Layer 1: EMA signal
hma_signal = results["HMA_Signal"]         # Layer 1: HMA signal
# ... các signals khác
```

### Phân tích một symbol

```python
from modules.adaptive_trend_enhance import analyze_symbol, ATCConfig
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager

# Khởi tạo
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Cấu hình
config = ATCConfig(
    timeframe="15m",
    limit=1500,
    ema_len=28,
    # ... các parameters khác
)

# Phân tích
result = analyze_symbol(
    symbol="BTC/USDT",
    data_fetcher=data_fetcher,
    config=config,
)

if result:
    print(f"Symbol: {result['symbol']}")
    print(f"Current Price: {result['current_price']}")
    print(f"ATC Results: {result['atc_results']}")
```

### Scan nhiều symbols

```python
from modules.adaptive_trend_enhance import scan_all_symbols, ATCConfig
from modules.common.core.data_fetcher import DataFetcher
from modules.common.core.exchange_manager import ExchangeManager

# Khởi tạo
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Cấu hình
config = ATCConfig(
    timeframe="15m",
    limit=1500,
    # ... các parameters khác
)

# Scan
results, short_signals = scan_all_symbols(
    data_fetcher=data_fetcher,
    atc_config=config,
    min_signal=0.5,  # Minimum signal strength
)

# Kết quả
for _, result in results.iterrows():
    print(f"{result['symbol']}: Signal = {result['signal']}")
```

### Sử dụng CLI

```bash
# Phân tích một symbol
python -m modules.adaptive_trend_enhance.cli.main BTC/USDT

# Scan tất cả futures symbols
python -m modules.adaptive_trend_enhance.cli.main --auto

# Interactive mode
python -m modules.adaptive_trend_enhance.cli.main

# Custom timeframe
python -m modules.adaptive_trend_enhance.cli.main BTC/USDT --timeframe 1h
```

## Cấu hình

### ATCConfig

```python
@dataclass
class ATCConfig:
    # Moving Average lengths
    ema_len: int = 28
    hma_len: int = 28
    wma_len: int = 28
    dema_len: int = 28
    lsma_len: int = 28
    kama_len: int = 28
    
    # ATC parameters
    robustness: str = "Medium"  # "Narrow", "Medium", or "Wide"
    lambda_param: float = 0.02  # Growth rate for equity
    decay: float = 0.03         # Decay rate for equity
    cutout: int = 0            # Bars to skip at beginning
    
    # Data parameters
    limit: int = 1500          # Number of candles to fetch
    timeframe: str = "15m"     # Timeframe
```

### Robustness Modes

- **Narrow**: Offsets nhỏ → ít variation trong MA lengths → nhạy cảm hơn
- **Medium**: Offsets trung bình → cân bằng
- **Wide**: Offsets lớn → nhiều variation → ổn định hơn, ít nhạy cảm hơn

## Kết quả

`compute_atc_signals` trả về dictionary chứa:

- **Average_Signal**: Signal cuối cùng (kết hợp tất cả MAs)
- **EMA_Signal**, **HMA_Signal**, **WMA_Signal**, **DEMA_Signal**, **LSMA_Signal**, **KAMA_Signal**: Layer 1 signals cho từng MA type
- **EMA_Weight**, **HMA_Weight**, **WMA_Weight**, **DEMA_Weight**, **LSMA_Weight**, **KAMA_Weight**: Weights cho từng MA type
- **EMA_Equity**, **HMA_Equity**, ...: Equity curves cho từng MA type

Tất cả đều là `pd.Series` với cùng index như input prices.

## Signal Interpretation

- **Positive values (> 0)**: Bullish signal, giá trên MA
- **Negative values (< 0)**: Bearish signal, giá dưới MA
- **Zero (0)**: Neutral, không có signal rõ ràng
- **Magnitude**: Độ mạnh của signal (cao hơn = mạnh hơn)

## Utilities

### rate_of_change

Tính toán rate of change (tỷ lệ thay đổi) của một series:

```python
from modules.adaptive_trend_enhance.utils import rate_of_change

roc = rate_of_change(prices, period=1)
```

### diflen

Tính toán độ dài khác biệt dựa trên robustness mode:

```python
from modules.adaptive_trend_enhance.utils import diflen

offset = diflen(robustness="Medium")  # Returns offset value
```

### exp_growth

Tính toán exponential growth factor:

```python
from modules.adaptive_trend_enhance.utils import exp_growth

growth = exp_growth(La=0.02, period=1)
```

## Rust Backend & Performance

**Rust backend** được dùng mặc định khi đã build (xem [Setup](#setup-rust-extensions) bên dưới). Các hàm equity, KAMA, MAs (EMA/WMA/DEMA/LSMA/HMA), signal persistence chạy trên Rust; nếu chưa build thì fallback sang Numba.

**Benchmarks (10k bars, `cargo bench` trong `rust_extensions/`):**

| Thành phần        | Thời gian (µs) | Ghi chú        |
|-------------------|----------------|----------------|
| Equity            | ~32            | 2–3x+ vs Numba |
| KAMA              | ~164           | 2–3x+ vs Numba |
| Signal persistence| ~8.5           | ~5x vs Numba   |
| EMA / DEMA        | ~14 / ~31      | MA Rust        |
| WMA / LSMA / HMA  | ~131 / ~194 / ~232 | MA Rust   |

- **Numba JIT**: Fallback khi Rust chưa có; equity và MA compile với Numba.
- **Vectorized operations**: NumPy cho các phép tính cuối.
- **Caching**: Rate of change được cache.
- **Parallel scanning**: Scanner hỗ trợ parallel cho nhiều symbols.

## Lưu ý

1. **Data quality**: ATC cần dữ liệu OHLCV chất lượng cao. Đảm bảo data không có gaps lớn.

2. **Timeframe**: ATC hoạt động tốt trên nhiều timeframes, nhưng parameters có thể cần điều chỉnh:
   - Timeframe ngắn (1m, 5m): Có thể cần giảm lengths
   - Timeframe dài (4h, 1d): Có thể cần tăng lengths

3. **Robustness**:
   - "Narrow" cho thị trường trending mạnh
   - "Medium" cho thị trường cân bằng
   - "Wide" cho thị trường volatile

4. **Lambda và Decay**:
   - Lambda cao → equity tăng nhanh → weights thay đổi nhanh
   - Decay cao → equity giảm nhanh → weights giảm nhanh

5. **Cutout**: Bỏ qua một số bars đầu tiên để tránh initialization artifacts.

## CLI Commands

Module cung cấp CLI interface qua `modules/adaptive_trend_enhance/cli/main.py`:

```bash
# Basic usage
python -m modules.adaptive_trend_enhance.cli.main <SYMBOL>

# Options
--timeframe TIMEFRAME    # Set timeframe (default: 15m)
--auto                   # Auto mode (scan all futures symbols)
--min-signal FLOAT       # Minimum signal strength for scan
--no-menu                # Skip interactive menu
--batch-size INT         # Batch size for memory optimization
```

## Ví dụ nâng cao

### Custom configuration từ dictionary

```python
from modules.adaptive_trend_enhance.utils.config import create_atc_config_from_dict

params = {
    "ema_len": 21,
    "hma_len": 21,
    "wma_len": 21,
    "dema_len": 21,
    "lsma_len": 21,
    "kama_len": 21,
    "robustness": "Narrow",
    "lambda_param": 0.03,
    "decay": 0.02,
    "limit": 2000,
}

config = create_atc_config_from_dict(params, timeframe="1h")
```

### Kết hợp với các indicators khác

```python
from modules.adaptive_trend_enhance import compute_atc_signals
from modules.common.core.indicator_engine import IndicatorEngine

# Tính ATC signals
atc_results = compute_atc_signals(prices=df['close'], ...)

# Tính các indicators khác
engine = IndicatorEngine()
df_with_indicators, metadata = engine.compute(df)

# Kết hợp signals
combined_signal = (
    atc_results['Average_Signal'] * 0.6 +
    (df_with_indicators['RSI_14'] - 50) / 50 * 0.4
)
```

## Troubleshooting

- **Rust không nhận / `rustc` not in PATH**: Thêm `%USERPROFILE%\.cargo\bin` vào PATH, hoặc chạy `.\build_rust.bat` / `.\build_rust.ps1` (tự thêm PATH). Chi tiết: [docs/phase3_task.md#troubleshooting](docs/phase3_task.md#troubleshooting).
- **Maturin build lỗi**: Kiểm tra `rustc --version`, `python --version`; kích hoạt venv trước khi build.
- **Import `atc_rust` lỗi**: Chạy `maturin develop --release` trong `rust_extensions/`; xác nhận bằng `pip show atc-rust`.
- **Numba cache sau đổi tên module**: Xóa `__pycache__` chứa `*.nbc` / `*.nbi` trong `core/signal_detection/` nếu gặp `ModuleNotFoundError` với đường dẫn module cũ.

## Tài liệu tham khảo

- Port từ Pine Script indicator "Adaptive Trend Classification"
- Sử dụng multiple Moving Averages với adaptive weighting
- Equity-based weighting để tự động điều chỉnh trọng số
