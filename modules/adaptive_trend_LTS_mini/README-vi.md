# Adaptive Trend Classification LTS (ATC LTS)

**Phiên bản hỗ trợ lâu dài với backend được tăng tốc bởi Rust, tối ưu GPU/CPU và quản lý bộ nhớ tự động**

> **Language / Ngôn ngữ**: [English](README-en.md) | [Tiếng Việt](README-vi.md)

Module Adaptive Trend Classification LTS là phiên bản ổn định của ATC với:

- **Rust backend**: Equity, KAMA, MAs (EMA/WMA/DEMA/LSMA/HMA), signal persistence chạy trên Rust khi đã build; fallback Numba nếu chưa build.
- **Parallel computing**: Multi-processing + multi-threading với auto-detection CPU/RAM
- **GPU acceleration**: Tự động detect và sử dụng GPU (CUDA/OpenCL) nếu có
- **Memory management**: Automatic cleanup, monitoring và prevention memory leaks
- **Numba JIT**: Fallback cho MA calculations khi Rust chưa có
- **Caching**: Intelligent caching cho MA results
- **Memory Optimizations**: Memory-mapped arrays cho backtesting và blosc compression cho cache
- **NumPy optimization**: Pre-allocated arrays và NumPy operations thay vì Pandas

Module cung cấp hệ thống phân tích xu hướng thích ứng sử dụng nhiều loại Moving Averages với adaptive weighting dựa trên equity curves.

## Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc Module](#cấu-trúc-module)
- [Phiên bản Mini chỉ CPU](#phiên-bản-mini-chỉ-cpu)
- [Cách hoạt động](#cách-hoạt-động)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Cấu hình](#cấu-hình)
- [Kết quả](#kết-quả)
- [Giải thích Signal](#giải-thích-signal)
- [Tiện ích](#tiện-ích)
- [Hiệu suất](#hiệu-suất)
- [Tối ưu hóa bộ nhớ](#tối-ưu-hóa-bộ-nhớ)
- [Lưu ý quan trọng](#lưu-ý-quan-trọng)
- [Lệnh CLI](#lệnh-cli)
- [Ví dụ nâng cao](#ví-dụ-nâng-cao)
- [Xử lý sự cố](#xử-lý-sự-cố)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)
- [Changelog](#changelog)

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
adaptive_trend_LTS_mini/
├── __init__.py              # Module exports
├── README.md                # Chọn ngôn ngữ
├── README-en.md             # Tài liệu tiếng Anh
├── README-vi.md             # Tài liệu này (Tiếng Việt)
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

## Phiên bản Mini chỉ CPU

Đây là **phiên bản mini chỉ dùng CPU** của module Adaptive Trend LTS với tất cả code CUDA/GPU đã được loại bỏ. Module vẫn giữ đầy đủ chức năng sử dụng backend Rust/Rayon CPU với hiệu suất đa nhân xuất sắc.

### ✅ Tính năng có sẵn (Chỉ CPU)
- Tất cả các tính toán ATC signals (Layer 1 & Layer 2)
- 6 loại MA: EMA, HMA, WMA, DEMA, LSMA, KAMA
- Xử lý CPU đa luồng thông qua Rust/Rayon
- Hỗ trợ xử lý batch với Dask
- Quét xấp xỉ để khám phá nhanh hơn
- Giao diện CLI đầy đủ
- Cập nhật ATC tăng dần

### ❌ Không có sẵn (Đã xóa)
- Tăng tốc GPU/CUDA
- Phụ thuộc CuPy/PyCUDA
- Biên dịch CUDA kernel

### Hiệu suất

| Thao tác | Phiên bản CPU (Rayon) |
|----------|----------------------|
| Symbol đơn (1000 bars) | ~100-500ms |
| 10 symbols batch | ~1-5s |
| 100 symbols batch | ~10-50s |
| 1000 symbols batch | ~100-500s |

**Khả năng mở rộng**: Tỷ lệ tuyến tính với các nhân CPU thông qua song song hóa Rayon

### Khi nào nên sử dụng phiên bản chỉ CPU

✅ **Sử dụng khi:**
- Không có GPU NVIDIA
- Môi trường cloud không có GPU instances
- Phát triển/test trên laptop
- Yêu cầu bộ nhớ thấp hơn
- Triển khai production trên máy chủ chỉ CPU

❌ **Không nên dùng khi:**
- Yêu cầu phân tích real-time trên 1000+ symbols
- Cần độ trễ cực thấp (<50ms)
- Giao dịch tần suất cao

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

## Cài đặt

### Yêu cầu
- Python 3.9+
- Rust toolchain (để build extensions)
- Không cần CUDA/GPU

### Build Rust Extension

```bash
cd modules/adaptive_trend_LTS_mini/rust_extensions
cargo build --release
```

Hoặc từ thư mục gốc project: `.\build_rust.bat` (Windows) / `.\build_rust.ps1`.

**Yêu cầu:** [Rust](https://rustup.rs/), [maturin](https://www.maturin.rs/) (`pip install maturin`). Chi tiết và xử lý lỗi: [docs/phase3_task.md#prerequisites--setup](docs/phase3_task.md#prerequisites--setup).

### Cài đặt Python Dependencies

```bash
pip install pandas numpy dask
```

## Sử dụng

### Phân tích cơ bản

```bash
python -m modules.adaptive_trend_LTS_mini.cli.main --symbol BTC/USDT --timeframe 1h
```

### Quét batch

```bash
python -m modules.adaptive_trend_LTS_mini.cli.main --scan --top 100
```

### Python API

Các ví dụ dưới dùng `legacy.adaptive_trend_enhance`; có thể thay bằng `modules.adaptive_trend_LTS_mini` (cùng API, dùng Rust backend khi đã build).

```python
import pandas as pd
from legacy.adaptive_trend_enhance import compute_atc_signals, ATCConfig

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
    hma_len=config.hma_len,
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
from legacy.adaptive_trend_enhance import analyze_symbol, ATCConfig
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
from legacy.adaptive_trend_enhance import scan_all_symbols, ATCConfig
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
python -m legacy.adaptive_trend_enhance.cli.main BTC/USDT

# Scan tất cả futures symbols
python -m legacy.adaptive_trend_enhance.cli.main --auto

# Interactive mode
python -m legacy.adaptive_trend_enhance.cli.main

# Custom timeframe
python -m legacy.adaptive_trend_enhance.cli.main BTC/USDT --timeframe 1h
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

## Giải thích Signal

- **Giá trị dương (> 0)**: Tín hiệu tăng giá (bullish), giá trên MA
- **Giá trị âm (< 0)**: Tín hiệu giảm giá (bearish), giá dưới MA
- **Bằng không (0)**: Trung lập, không có tín hiệu rõ ràng
- **Độ lớn**: Độ mạnh của signal (cao hơn = mạnh hơn)

## Tiện ích

### rate_of_change

Tính toán rate of change (tỷ lệ thay đổi) của một series:

```python
from legacy.adaptive_trend_enhance.utils import rate_of_change

roc = rate_of_change(prices, period=1)
```

### diflen

Tính toán độ dài khác biệt dựa trên robustness mode:

```python
from legacy.adaptive_trend_enhance.utils import diflen

offset = diflen(robustness="Medium")  # Returns offset value
```

### exp_growth

Tính toán exponential growth factor:

```python
from legacy.adaptive_trend_enhance.utils import exp_growth

growth = exp_growth(La=0.02, period=1)
```

## Hiệu suất

### Rust Backend

**Rust backend** được dùng mặc định khi đã build (xem [Cài đặt](#cài-đặt) bên trên). Các hàm equity, KAMA, MAs (EMA/WMA/DEMA/LSMA/HMA), signal persistence chạy trên Rust; nếu chưa build thì fallback sang Numba.

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

### Cấu hình chỉ CPU

Tất cả cấu hình chỉ dùng CPU:

```python
from modules.adaptive_trend_LTS_mini.utils.config import ATCConfig

config = ATCConfig(
    use_rust_backend=True,  # Dùng Rust/Rayon CPU backend
    use_approximate=True,   # Bật approximate để quét nhanh hơn
    parallel_l1=True,       # Tính toán Layer 1 song song
    parallel_l2=True,       # Tính toán Layer 2 song song
)
```

### Mẹo hiệu suất chỉ CPU

1. **Dùng Rust backend**: Luôn đặt `use_rust_backend=True`
2. **Bật chế độ approximate**: Đặt `use_approximate=True` để quét
3. **Tối ưu kích thước batch**: 50-200 symbols per batch (điều chỉnh theo CPU)
4. **Dùng tất cả CPU cores**: Rayon tự động dùng các cores có sẵn
5. **Cache dữ liệu**: Tái sử dụng dữ liệu OHLCV đã fetch nếu có thể

## Tối ưu hóa bộ nhớ

Module hỗ trợ các tối ưu hóa bộ nhớ cho datasets lớn:

1.  **Memory-Mapped Arrays**:
    - Xử lý datasets lớn mà không cần load toàn bộ vào RAM
    - Giảm 90%+ RAM usage cho backtesting
    - Enable qua `use_memory_mapped=True` trong `ATCConfig`

2.  **Data Compression**:
    - Nén cache files sử dụng `blosc`
    - Giảm 5-10x storage footprint
    - Enable qua `use_compression=True` trong `ATCConfig`

Xem chi tiết: [docs/memory_optimizations_usage_guide.md](docs/memory_optimizations_usage_guide.md)

## Lưu ý quan trọng

1. **Chất lượng dữ liệu**: ATC cần dữ liệu OHLCV chất lượng cao. Đảm bảo data không có gaps lớn.

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

## Lệnh CLI

Module cung cấp CLI interface qua `legacy/adaptive_trend_enhance/cli/main.py`:

```bash
# Sử dụng cơ bản
python -m legacy.adaptive_trend_enhance.cli.main <SYMBOL>

# Các tùy chọn
--timeframe TIMEFRAME    # Đặt timeframe (mặc định: 15m)
--auto                   # Chế độ tự động (quét tất cả futures symbols)
--min-signal FLOAT       # Độ mạnh signal tối thiểu để quét
--no-menu                # Bỏ qua menu interactive
--batch-size INT         # Kích thước batch để tối ưu bộ nhớ
```

## Ví dụ nâng cao

### Custom configuration từ dictionary

```python
from legacy.adaptive_trend_enhance.utils.config import create_atc_config_from_dict

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
from legacy.adaptive_trend_enhance import compute_atc_signals
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

## Xử lý sự cố

### Lỗi: Rust không nhận / `rustc` không trong PATH

**Giải pháp**: Thêm `%USERPROFILE%\.cargo\bin` vào PATH, hoặc chạy `.\build_rust.bat` / `.\build_rust.ps1` (tự thêm PATH). Chi tiết: [docs/phase3_task.md#troubleshooting](docs/phase3_task.md#troubleshooting).

### Lỗi: Maturin build lỗi

**Giải pháp**: Kiểm tra `rustc --version`, `python --version`; kích hoạt venv trước khi build.

### Lỗi: Import `atc_rust` lỗi

**Giải pháp**: Chạy `maturin develop --release` trong `rust_extensions/`; xác nhận bằng `pip show atc-rust`.

### Lỗi: Numba cache sau đổi tên module

**Giải pháp**: Xóa `__pycache__` chứa `*.nbc` / `*.nbi` trong `core/signal_detection/` nếu gặp `ModuleNotFoundError` với đường dẫn module cũ.

### Lỗi: Hiệu suất chậm

**Giải pháp:**
1. Xác nhận Rust backend được bật: `use_rust_backend=True`
2. Build ở chế độ release: `cargo build --release`
3. Bật xử lý song song: `parallel_l1=True, parallel_l2=True`
4. Kiểm tra mức sử dụng CPU cores trong quá trình xử lý

### Lỗi: Build Errors

**Giải pháp:**
```bash
cd rust_extensions
cargo clean
cargo build --release
```

### Lỗi: Import Errors

**Giải pháp:**
Đảm bảo Rust extension đã được build:
```bash
ls rust_extensions/target/release/*.dll  # Windows
ls rust_extensions/target/release/*.so   # Linux/Mac
```

## Tài liệu tham khảo

- Port từ Pine Script indicator "Adaptive Trend Classification"
- Sử dụng multiple Moving Averages với adaptive weighting
- Equity-based weighting để tự động điều chỉnh trọng số

---

## Changelog - Phiên bản Mini chỉ CPU

### Version 1.0.0 (2026-01-31)

#### Tổng quan
Hoàn thiện việc di chuyển từ các phụ thuộc GPU/CUDA sang triển khai chỉ CPU sử dụng backend Rust/Rayon.

#### Đã xóa

##### Các phụ thuộc CUDA/GPU
- Tất cả các phụ thuộc Rust liên quan đến CUDA
- Các gói Python CuPy và PyCUDA
- Tất cả các file CUDA kernel `.cu`
- Các module Python backend GPU
- Các file test và benchmark GPU

#### Đã sửa đổi

##### Thay đổi API
- Xóa tham số `use_cuda` khỏi tất cả các hàm
- Xóa tham số `prefer_gpu` khỏi `compute_atc_signals()`
- Xóa cờ CLI `--use-cuda`
- Xóa trường `use_cuda` khỏi dataclass `ATCConfig`

##### Thay đổi Build
- Kích thước build Rust giảm: ~1.5MB → ~637KB (nhỏ hơn 57%)
- Logic routing được đơn giản hóa (chỉ CPU)

#### Giữ nguyên

##### Chức năng cốt lõi (100% được bảo toàn)
- Tất cả các thuật toán tính toán ATC signals
- Tất cả 6 loại MA (EMA, HMA, WMA, DEMA, LSMA, KAMA)
- Logic tính toán Layer 1 & Layer 2
- Backend Rust/Rayon CPU (đa luồng)
- Hỗ trợ xử lý batch Dask
- Chế độ quét xấp xỉ
- Giao diện CLI đầy đủ
- Cập nhật ATC tăng dần
- Tính toán persistence signal

#### So sánh hiệu suất

| Chỉ số | Phiên bản GPU | Phiên bản chỉ CPU | Hệ số |
|--------|---------------|-------------------|--------|
| Symbol đơn | 10-50ms | 100-500ms | ~10x chậm hơn |
| Batch 100 symbols | 2-10s | 10-50s | ~5x chậm hơn |
| Sử dụng bộ nhớ | GPU VRAM + RAM | Chỉ RAM | Hiệu quả hơn |
| CPU cores sử dụng | 1-2 | Tất cả có sẵn | Tận dụng tốt hơn |

#### Lợi ích của phiên bản chỉ CPU

1. **Không phụ thuộc phần cứng**: Hoạt động trên bất kỳ CPU nào, không cần GPU NVIDIA
2. **Bộ nhớ thấp hơn**: Không có overhead GPU memory
3. **Thân thiện với Cloud**: Chạy trên bất kỳ cloud instance nào không có GPU
4. **Dễ phát triển**: Test trên laptop không có GPU rời
5. **Build nhỏ hơn**: Kích thước binary nhỏ hơn 57%
6. **Triển khai đơn giản hơn**: Không có phụ thuộc CUDA runtime

#### Kết quả xác thực
- Rust extension build thành công không có CUDA
- Tất cả imports hoạt động không cần phụ thuộc GPU
- Không có imports cupy/pycuda trong codebase
- Tất cả các test không GPU đều pass
- Các test xác thực chỉ CPU đều pass

**Ngày di chuyển**: 2026-01-31
**Phiên bản**: 1.0.0
**Trạng thái**: Sẵn sàng Production
