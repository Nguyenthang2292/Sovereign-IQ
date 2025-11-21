# 📚 PortfolioCorrelationAnalyzer Documentation

## Mục lục
1. [Tổng quan](#tổng-quan)
2. [Khởi tạo](#khởi-tạo)
3. [Phương thức chính](#phương-thức-chính)
4. [Ví dụ sử dụng](#ví-dụ-sử-dụng)
5. [Best Practices](#best-practices)
6. [Giải thích kỹ thuật](#giải-thích-kỹ-thuật)
7. [Troubleshooting](#troubleshooting)

---

## Tổng quan

`PortfolioCorrelationAnalyzer` là một lớp phân tích correlation (tương quan) giữa các positions trong portfolio và các symbols mới. Lớp này cung cấp:

- ✅ **Portfolio Internal Correlation** - Tính correlation nội bộ giữa các positions trong portfolio
- ✅ **Weighted Correlation** - Tính correlation có trọng số giữa symbol mới và portfolio
- ✅ **Portfolio Return Correlation** - Tính correlation trên portfolio aggregated returns
- ✅ **Correlation Impact Analysis** - Phân tích impact khi thêm symbol mới vào portfolio
- ✅ **LONG/SHORT Support** - Xử lý đúng correlation cho LONG và SHORT positions
- ✅ **Returns-based Calculation** - Tính correlation trên returns thay vì prices (tránh spurious correlation)
- ✅ **Vectorized Operations** - Sử dụng vectorization để tối ưu hiệu suất
- ✅ **Caching** - Cache price series để tránh fetch lại

### Khi nào dùng PortfolioCorrelationAnalyzer?

| Mục đích | Dùng PortfolioCorrelationAnalyzer? | Phương thức |
|----------|-----------------------------------|-------------|
| Phân tích correlation giữa các positions trong portfolio | ✅ Có | `calculate_weighted_correlation()` |
| Đánh giá correlation của symbol mới với portfolio | ✅ Có | `calculate_weighted_correlation_with_new_symbol()` |
| Tính correlation trên portfolio aggregated returns | ✅ Có | `calculate_portfolio_return_correlation()` |
| Phân tích impact khi thêm position mới | ✅ Có | `analyze_correlation_with_new_symbol()` |
| Tìm hedge candidates | ✅ Có | Kết hợp với `HedgeFinder` |
| Đánh giá diversification | ✅ Có | `analyze_correlation_with_new_symbol()` |

---

## Khởi tạo

### Cú pháp

```python
from modules.PortfolioCorrelationAnalyzer import PortfolioCorrelationAnalyzer
from modules.DataFetcher import DataFetcher
from modules.ExchangeManager import ExchangeManager
from modules.Position import Position

# Khởi tạo dependencies
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Tạo danh sách positions
positions = [
    Position("BTC/USDT", "LONG", entry_price=50000.0, size_usdt=1000.0),
    Position("ETH/USDT", "LONG", entry_price=3000.0, size_usdt=500.0),
    Position("SOL/USDT", "SHORT", entry_price=100.0, size_usdt=300.0),
]

# Khởi tạo analyzer
analyzer = PortfolioCorrelationAnalyzer(data_fetcher, positions)
```

### Tham số

- `data_fetcher` (DataFetcher): Instance của DataFetcher để fetch price data
- `positions` (List[Position]): Danh sách các positions trong portfolio

### Attributes

- `data_fetcher`: DataFetcher instance
- `positions`: List các Position objects
- `_series_cache`: Dictionary cache các price series đã fetch (internal)

---

## Phương thức chính

### 1. `calculate_weighted_correlation(verbose=True)`

Tính correlation nội bộ của portfolio (giữa các positions với nhau).

#### Tham số

- `verbose` (bool): Có in output chi tiết hay không (default: `True`)

#### Returns

- `tuple[float | None, list]`: 
  - `weighted_correlation`: Correlation trung bình có trọng số giữa tất cả các cặp positions
  - `position_correlations_list`: List các dict chứa chi tiết correlation cho từng cặp

#### Ví dụ

```python
internal_corr, pairs = analyzer.calculate_weighted_correlation(verbose=True)

# Output:
# Portfolio Internal Correlation Analysis:
# Position Pair Correlations (PnL-adjusted):
#   BTC/USDT     (LONG ) <-> ETH/USDT     (LONG ) (  750.00 USDT, 100.0%): 0.8512
# Weighted Internal Correlation:
#   Portfolio Internal: 0.8512

print(f"Internal correlation: {internal_corr}")
for pair in pairs:
    print(f"{pair['symbol1']} <-> {pair['symbol2']}: {pair['correlation']:.4f}")
```

#### Chi tiết

- Tính correlation giữa tất cả các cặp positions
- Sử dụng returns (pct_change) thay vì prices
- Điều chỉnh returns cho SHORT positions (đảo dấu)
- Weighted average theo position size
- Yêu cầu ít nhất 2 positions

---

### 2. `calculate_weighted_correlation_with_new_symbol(new_symbol, verbose=True)`

Tính correlation có trọng số giữa một symbol mới và portfolio.

#### Tham số

- `new_symbol` (str): Symbol cần phân tích (ví dụ: "BNB/USDT")
- `verbose` (bool): Có in output chi tiết hay không (default: `True`)

#### Returns

- `tuple[float | None, list]`:
  - `weighted_correlation`: Correlation trung bình có trọng số
  - `position_details`: List các dict chứa correlation với từng position

#### Ví dụ

```python
weighted_corr, details = analyzer.calculate_weighted_correlation_with_new_symbol("BNB/USDT", verbose=True)

# Output:
# Correlation Analysis (Weighted by Position Size):
# Individual Correlations:
#   BTC/USDT     (LONG ,  1000.00 USDT,  66.7%): 0.6906
#   ETH/USDT     (LONG ,   500.00 USDT,  33.3%): 0.6964
# Weighted Portfolio Correlation:
#   BNB/USDT vs Portfolio: 0.6925

print(f"Weighted correlation: {weighted_corr}")
for detail in details:
    print(f"{detail['symbol']}: {detail['correlation']:.4f}")
```

#### Chi tiết

- Tính correlation giữa new_symbol và từng position trong portfolio
- Weighted average theo position size
- Xử lý LONG/SHORT đúng cách
- Sử dụng returns thay vì prices

---

### 3. `calculate_portfolio_return_correlation(new_symbol, min_points=10, verbose=True)`

Tính correlation giữa portfolio aggregated returns và symbol mới.

#### Tham số

- `new_symbol` (str): Symbol cần phân tích
- `min_points` (int): Số điểm dữ liệu tối thiểu (default: `DEFAULT_CORRELATION_MIN_POINTS`)
- `verbose` (bool): Có in output chi tiết hay không (default: `True`)

#### Returns

- `tuple[float | None, dict]`:
  - `correlation`: Correlation coefficient
  - `metadata`: Dict chứa thông tin bổ sung (ví dụ: `{"samples": 1499}`)

#### Ví dụ

```python
corr, metadata = analyzer.calculate_portfolio_return_correlation("BNB/USDT", verbose=True)

# Output:
# Portfolio Return Correlation Analysis:
#   Portfolio Return vs BNB/USDT: 0.7202
#   Samples used: 1499

print(f"Correlation: {corr:.4f}")
print(f"Samples: {metadata.get('samples', 'N/A')}")
```

#### Chi tiết

- Tính portfolio aggregated returns (weighted average của tất cả positions)
- Xử lý LONG/SHORT đúng cách
- Sử dụng vectorization để tối ưu hiệu suất
- Tính correlation trên returns series

---

### 4. `analyze_correlation_with_new_symbol(new_symbol, new_position_size=0.0, new_direction="LONG", verbose=True)`

Phân tích impact của việc thêm một symbol mới vào portfolio.

#### Tham số

- `new_symbol` (str): Symbol cần phân tích
- `new_position_size` (float): Size của position mới (USDT) (default: `0.0`)
- `new_direction` (str): Hướng position mới ("LONG" hoặc "SHORT") (default: `"LONG"`)
- `verbose` (bool): Có in output chi tiết hay không (default: `True`)

#### Returns

- `dict`: Dictionary chứa:
  ```python
  {
      "before": {
          "internal_correlation": float  # Correlation nội bộ trước khi thêm
      },
      "after": {
          "new_symbol_correlation": float,  # Correlation với symbol mới
          "portfolio_return_correlation": float,  # Portfolio return correlation
          "internal_correlation": float  # Correlation nội bộ sau khi thêm (nếu new_position_size > 0)
      },
      "impact": {
          "correlation_change": float,  # Thay đổi correlation
          "diversification_improvement": bool  # Có cải thiện diversification không
      }
  }
  ```

#### Ví dụ

```python
impact = analyzer.analyze_correlation_with_new_symbol(
    new_symbol="BNB/USDT",
    new_position_size=800.0,
    new_direction="LONG",
    verbose=True
)

# Output:
# === Analyzing Correlation Impact of Adding New Symbol ===
# === Summary ===
# Current Portfolio Internal Correlation: 0.8512
# New Symbol vs Portfolio Correlation: 0.6925
# Portfolio Return vs New Symbol Correlation: 0.7202
# Portfolio Internal Correlation After: 0.7446
# Correlation Change: -0.1066
# Diversification Improvement: True

print(f"Correlation change: {impact['impact']['correlation_change']:.4f}")
print(f"Improvement: {impact['impact']['diversification_improvement']}")
```

#### Chi tiết

- Tính correlation nội bộ trước khi thêm symbol
- Tính correlation với symbol mới
- Simulate thêm position và tính lại correlation nội bộ
- Đánh giá diversification improvement (correlation giảm = tốt hơn)

---

## Ví dụ sử dụng

### Ví dụ 1: Phân tích correlation nội bộ portfolio

```python
from modules.PortfolioCorrelationAnalyzer import PortfolioCorrelationAnalyzer
from modules.DataFetcher import DataFetcher
from modules.ExchangeManager import ExchangeManager
from modules.Position import Position

# Setup
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

positions = [
    Position("BTC/USDT", "LONG", entry_price=50000.0, size_usdt=1000.0),
    Position("ETH/USDT", "LONG", entry_price=3000.0, size_usdt=500.0),
    Position("SOL/USDT", "SHORT", entry_price=100.0, size_usdt=300.0),
]

analyzer = PortfolioCorrelationAnalyzer(data_fetcher, positions)

# Tính correlation nội bộ
internal_corr, pairs = analyzer.calculate_weighted_correlation(verbose=True)

# Phân tích kết quả
if internal_corr is not None:
    if abs(internal_corr) > 0.7:
        print("⚠️  High correlation - Portfolio is concentrated")
    elif abs(internal_corr) > 0.4:
        print("⚠️  Moderate correlation")
    else:
        print("✅ Low correlation - Good diversification")
```

### Ví dụ 2: Đánh giá symbol mới

```python
# Đánh giá correlation của BNB với portfolio
weighted_corr, details = analyzer.calculate_weighted_correlation_with_new_symbol("BNB/USDT", verbose=True)

if weighted_corr is not None:
    if abs(weighted_corr) > 0.7:
        print("⚠️  BNB highly correlated with portfolio")
    else:
        print("✅ BNB has low correlation - Good for diversification")
```

### Ví dụ 3: Phân tích impact trước khi thêm position

```python
# Phân tích impact khi thêm BNB vào portfolio
impact = analyzer.analyze_correlation_with_new_symbol(
    new_symbol="BNB/USDT",
    new_position_size=800.0,
    new_direction="LONG",
    verbose=True
)

# Quyết định dựa trên kết quả
if impact['impact'].get('diversification_improvement', False):
    print("✅ Adding BNB will improve diversification")
    print(f"Correlation will decrease by {abs(impact['impact']['correlation_change']):.4f}")
else:
    print("⚠️  Adding BNB may increase portfolio concentration")
```

### Ví dụ 4: So sánh nhiều symbols

```python
candidates = ["BNB/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"]

results = []
for symbol in candidates:
    weighted_corr, _ = analyzer.calculate_weighted_correlation_with_new_symbol(symbol, verbose=False)
    portfolio_return_corr, _ = analyzer.calculate_portfolio_return_correlation(symbol, verbose=False)
    
    if weighted_corr is not None and portfolio_return_corr is not None:
        results.append({
            "symbol": symbol,
            "weighted_corr": weighted_corr,
            "return_corr": portfolio_return_corr,
            "avg_corr": (abs(weighted_corr) + abs(portfolio_return_corr)) / 2
        })

# Sắp xếp theo correlation thấp nhất (tốt nhất cho diversification)
results.sort(key=lambda x: x["avg_corr"])

print("\nBest diversification candidates (lowest correlation):")
for i, result in enumerate(results[:3], 1):
    print(f"{i}. {result['symbol']}: {result['avg_corr']:.4f}")
```

---

## Best Practices

### 1. Sử dụng Returns thay vì Prices

✅ **ĐÚNG**: Correlation được tính trên returns (pct_change)
- Tránh spurious correlation từ non-stationary price series
- Phản ánh đúng mối quan hệ biến động

❌ **SAI**: Tính correlation trực tiếp trên prices
- Có thể tạo spurious correlation
- Không phản ánh đúng mối quan hệ thực tế

### 2. Xử lý LONG/SHORT đúng cách

✅ **ĐÚNG**: Code tự động điều chỉnh returns cho SHORT positions
- Long BTC + Short ETH: Correlation âm = Hedge tốt ✅
- Long BTC + Long ETH: Correlation dương = Rủi ro cao ⚠️

❌ **SAI**: Không xét hướng position
- Long BTC + Short ETH sẽ báo correlation dương cao (sai!)

### 3. Sử dụng Weighted Correlation

✅ **ĐÚNG**: Sử dụng weighted correlation theo position size
- Positions lớn hơn có ảnh hưởng lớn hơn đến correlation

❌ **SAI**: Tính correlation đơn giản (không weighted)
- Không phản ánh đúng ảnh hưởng của từng position

### 4. Đánh giá Diversification

✅ **ĐÚNG**: 
```python
# Correlation thấp = Diversification tốt
if abs(correlation) < 0.4:
    print("Good diversification")
elif abs(correlation) < 0.7:
    print("Moderate correlation")
else:
    print("High correlation - Consider hedging")
```

### 5. Cache Management

✅ **ĐÚNG**: Analyzer tự động cache price series
- Tránh fetch lại dữ liệu đã có
- Tăng hiệu suất khi tính nhiều correlations

---

## Giải thích kỹ thuật

### 1. Tại sao tính correlation trên Returns?

**Vấn đề với Prices:**
- Price series thường là non-stationary (có trend)
- Hai assets cùng có trend tăng sẽ có correlation cao dù không thực sự liên quan
- Ví dụ: BTC và ETH đều tăng theo thời gian → correlation cao giả tạo

**Giải pháp - Returns:**
- Returns (pct_change) thường stationary hơn
- Phản ánh đúng sự biến động cùng chiều/ngược chiều
- Correlation trên returns = correlation thực tế về biến động

**Code:**
```python
# Tính returns từ prices
returns_df = df.pct_change().dropna()

# Tính correlation trên returns
corr = returns_df.iloc[:, 0].corr(returns_df.iloc[:, 1])
```

### 2. Xử lý LONG/SHORT Positions

**Vấn đề:**
- Long BTC + Short ETH: Giá cùng tăng → BTC profit, ETH loss
- Về mặt PnL: Correlation âm (hedge nhau)
- Nhưng về mặt giá: Correlation dương (cùng tăng)

**Giải pháp:**
- Đảo dấu returns cho SHORT positions trước khi tính correlation
- Correlation trên adjusted returns = PnL correlation

**Code:**
```python
# Điều chỉnh returns cho SHORT positions
adjusted_returns = returns_df.copy()
if pos.direction == "SHORT":
    adjusted_returns.iloc[:, 0] = -adjusted_returns.iloc[:, 0]

# Tính correlation trên adjusted returns
corr = adjusted_returns.iloc[:, 0].corr(adjusted_returns.iloc[:, 1])
```

### 3. Vectorization trong Portfolio Return Correlation

**Vấn đề với vòng lặp:**
```python
# Chậm với dữ liệu lớn
for idx in common_index:
    for pos in positions:
        ret = portfolio_returns_df.at[idx, pos.symbol]  # Truy cập từng cell
        weighted_return += ret * weight
```

**Giải pháp - Vectorization:**
```python
# Nhanh với vectorized operations
adjusted_common = adjusted_returns_df.loc[common_index, valid_symbols]
weights_array = np.array([position_weights[sym] for sym in valid_symbols])
weighted_sums = (adjusted_common * weights_array).sum(axis=1)  # Vectorized
```

**Lợi ích:**
- Nhanh hơn 10-100x với dữ liệu lớn
- Tận dụng NumPy/Pandas optimizations
- Code gọn và dễ đọc hơn

### 4. Weighted Average Correlation

**Công thức:**
```
Weighted Correlation = Σ(correlation_i × weight_i) / Σ(weight_i)
```

**Ví dụ:**
- BTC/USDT: correlation = 0.7, weight = 1000 USDT
- ETH/USDT: correlation = 0.5, weight = 500 USDT
- Weighted = (0.7 × 1000 + 0.5 × 500) / (1000 + 500) = 0.633

---

## Troubleshooting

### Lỗi: "Need at least 2 positions to calculate internal correlation"

**Nguyên nhân:** Portfolio chỉ có 1 position hoặc không có position nào.

**Giải pháp:**
```python
if len(positions) < 2:
    print("Add more positions to calculate internal correlation")
```

### Lỗi: "Insufficient data for correlation analysis"

**Nguyên nhân:** Không đủ dữ liệu overlapping giữa các symbols.

**Giải pháp:**
- Kiểm tra xem symbols có tồn tại trên exchange không
- Tăng `limit` khi fetch OHLCV data
- Kiểm tra timeframe (một số symbols có thể không có data ở timeframe nhỏ)

### Correlation luôn là NaN

**Nguyên nhân:** 
- Không đủ variance trong returns
- Tất cả returns đều bằng nhau (không có biến động)

**Giải pháp:**
- Kiểm tra dữ liệu có đúng không
- Thử với timeframe khác
- Kiểm tra xem có đủ số điểm dữ liệu không (min_points)

### Correlation cao bất thường

**Nguyên nhân có thể:**
- Tính trên prices thay vì returns (đã được fix trong code)
- Không xử lý LONG/SHORT đúng (đã được fix trong code)
- Symbols thực sự có correlation cao (ví dụ: BTC và ETH)

**Giải pháp:**
- Code đã tự động xử lý đúng
- Nếu vẫn cao, có thể là correlation thực tế
- Xem xét thêm symbols khác để diversify

---

## Liên kết

- [DataFetcher Documentation](./DataFetcher.md) - Tài liệu về DataFetcher
- [ExchangeManager Documentation](./ExchangeManager.md) - Tài liệu về ExchangeManager
- [PortfolioRiskCalculator](../modules/PortfolioRiskCalculator.py) - Risk calculator cho portfolio

