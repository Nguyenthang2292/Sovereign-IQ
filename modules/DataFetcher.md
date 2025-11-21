# 📚 DataFetcher Documentation

## Mục lục
1. [Tổng quan](#tổng-quan)
2. [Khởi tạo](#khởi-tạo)
3. [Phương thức](#phương-thức)
4. [Ví dụ sử dụng](#ví-dụ-sử-dụng)
5. [Best Practices](#best-practices)
6. [Tính năng nâng cao](#tính-năng-nâng-cao)

---

## Tổng quan

`DataFetcher` là một lớp tiện ích để lấy dữ liệu thị trường từ các sàn giao dịch crypto. Lớp này cung cấp:

- ✅ **Lấy giá hiện tại** (`fetch_current_prices_from_binance`) - Lấy giá ticker từ Binance (cần credentials)
- ✅ **Lấy dữ liệu OHLCV** (`fetch_ohlcv_with_fallback_exchange`) - Lấy dữ liệu lịch sử với fallback tự động (không cần credentials)
- ✅ **Caching tự động** - Cache OHLCV data (riêng cho Series và DataFrame) để tránh fetch lại
- ✅ **Fallback mechanism** - Tự động thử các exchange khác nếu một exchange fail
- ✅ **Freshness checking** - Kiểm tra độ tươi của dữ liệu và tìm exchange có data tươi nhất
- ✅ **Flexible return format** - Trả về Series (mặc định) hoặc full DataFrame với exchange_id
- ✅ **Shutdown support** - Hỗ trợ graceful shutdown khi có signal
- ✅ **Progress tracking** - Hiển thị progress bar khi fetch prices
- ✅ **Error handling** - Xử lý lỗi một cách graceful

### Khi nào dùng DataFetcher?

| Mục đích | Dùng DataFetcher? | Phương thức |
|----------|------------------|-------------|
| Lấy giá hiện tại của nhiều symbols | ✅ Có | `fetch_current_prices_from_binance()` |
| Lấy dữ liệu OHLCV lịch sử | ✅ Có | `fetch_ohlcv_with_fallback_exchange()` |
| Lấy dữ liệu với fallback tự động | ✅ Có | `fetch_ohlcv_with_fallback_exchange()` |
| Cần caching để tối ưu performance | ✅ Có | Tự động trong `fetch_ohlcv_with_fallback_exchange()` |
| Cần kiểm tra độ tươi của dữ liệu | ✅ Có | `fetch_ohlcv_with_fallback_exchange(check_freshness=True)` |
| Cần làm việc với DataFrame/Series | ✅ Có | Hàm trả về DataFrame; dùng trực tiếp hoặc gọi `DataFetcher.dataframe_to_close_series()` để lấy Series |
| Lấy dữ liệu từ một exchange cụ thể | ❌ Không | Dùng `ExchangeManager` trực tiếp |

---

## Khởi tạo

### Cú pháp

```python
from modules.DataFetcher import DataFetcher
from modules.ExchangeManager import ExchangeManager

# Khởi tạo ExchangeManager trước
em = ExchangeManager(api_key="your_key", api_secret="your_secret")

# Khởi tạo DataFetcher
data_fetcher = DataFetcher(
    exchange_manager=em,
    shutdown_event=None  # Optional: threading.Event() để hỗ trợ shutdown
)
```

### Tham số

- `exchange_manager` (ExchangeManager, **bắt buộc**): Instance của ExchangeManager để kết nối đến exchanges
- `shutdown_event` (threading.Event, **tùy chọn**): Event object để hỗ trợ graceful shutdown. Nếu được set, các phương thức sẽ kiểm tra và dừng khi event được set.

### Ví dụ khởi tạo

```python
from modules.DataFetcher import DataFetcher
from modules.ExchangeManager import ExchangeManager
import threading

# Cách 1: Khởi tạo đơn giản (không có shutdown event)
em = ExchangeManager(api_key="key", api_secret="secret")
data_fetcher = DataFetcher(em)

# Cách 2: Với shutdown event (cho multi-threading)
em = ExchangeManager(api_key="key", api_secret="secret")
shutdown = threading.Event()
data_fetcher = DataFetcher(em, shutdown_event=shutdown)

# Cách 3: Không có credentials (chỉ dùng cho OHLCV)
em = ExchangeManager()  # Không cần credentials cho OHLCV
data_fetcher = DataFetcher(em)
```

### Thuộc tính

Sau khi khởi tạo, `DataFetcher` có các thuộc tính:

- `exchange_manager`: ExchangeManager instance được truyền vào
- `shutdown_event`: Shutdown event (nếu có)
- `_ohlcv_cache`: Cache nội bộ cho OHLCV Series data (Dict[Tuple[str, str, int], pd.Series])
- `_ohlcv_dataframe_cache`: Cache nội bộ cho OHLCV DataFrame data (Dict[Tuple[str, str, int], pd.DataFrame])
- `market_prices`: Dictionary lưu giá hiện tại của các symbols (Dict[str, float])

---

## Phương thức

### `fetch_current_prices_from_binance(symbols: list) -> None`

**Mục đích**: Lấy giá hiện tại (ticker) của nhiều symbols từ Binance.

**Khi nào dùng:**
- ✅ Cần lấy giá hiện tại của nhiều symbols cùng lúc
- ✅ Cần hiển thị progress bar khi fetch
- ✅ Cần lưu giá vào `market_prices` dictionary

**Tham số:**
- `symbols` (list): Danh sách các symbols cần lấy giá (ví dụ: `["BTC/USDT", "ETH/USDT"]`)

**Trả về:**
- `None` (giá được lưu vào `self.market_prices`)

**Ví dụ:**
```python
from modules.DataFetcher import DataFetcher
from modules.ExchangeManager import ExchangeManager

em = ExchangeManager(api_key="key", api_secret="secret")
data_fetcher = DataFetcher(em)

# Lấy giá của nhiều symbols
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
data_fetcher.fetch_current_prices_from_binance(symbols)

# Truy cập giá đã lấy
print(f"BTC/USDT: {data_fetcher.market_prices.get('BTC/USDT')}")
print(f"ETH/USDT: {data_fetcher.market_prices.get('ETH/USDT')}")
```

**Lưu ý:**
- ⚠️ **Cần credentials** (API key/secret) để lấy giá từ Binance
- ⚠️ Nếu không có credentials, sẽ in error message và return sớm
- ✅ Tự động normalize symbols (ví dụ: "BTCUSDT" → "BTC/USDT")
- ✅ Hiển thị progress bar khi fetch
- ✅ Xử lý lỗi graceful, tiếp tục fetch các symbols khác nếu một symbol fail
- ✅ Hỗ trợ shutdown signal (nếu có `shutdown_event`)
- ✅ In thông báo thành công/thất bại cho từng symbol

**Output mẫu:**
```
Fetching current prices from Binance...
  [BINANCE] BTC/USDT: 43250.50000000
  [BINANCE] ETH/USDT: 2650.75000000
  [BINANCE] BNB/USDT: 315.20000000

Successfully fetched prices for 3/3 symbols
```

---

### `fetch_ohlcv_with_fallback_exchange(symbol: str, limit: int = 1500, timeframe: str = '1h', check_freshness: bool = False, exchanges: list = None) -> Tuple[pd.DataFrame, str] | (None, None)`

**Mục đích**: Lấy dữ liệu OHLCV (Open, High, Low, Close, Volume) lịch sử với fallback tự động, caching, và tùy chọn kiểm tra độ tươi của dữ liệu.

**Khi nào dùng:**
- ✅ Cần dữ liệu OHLCV lịch sử để phân tích kỹ thuật
- ✅ Cần fallback tự động nếu một exchange không có dữ liệu
- ✅ Cần caching để tối ưu performance (tránh fetch lại dữ liệu đã có)
- ✅ Cần kiểm tra độ tươi của dữ liệu (freshness checking)
- ✅ Cần full DataFrame thay vì chỉ Series

**Tham số:**
- `symbol` (str): Symbol cần lấy (ví dụ: "BTC/USDT", "ETH/USDT")
- `limit` (int, optional): Số lượng candles cần lấy (mặc định: 1500)
- `timeframe` (str, optional): Timeframe (mặc định: '1h'). Các giá trị phổ biến: '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'
- `check_freshness` (bool, optional): Nếu `True`, kiểm tra độ tươi của dữ liệu và thử nhiều exchanges để tìm data tươi nhất (mặc định: `False`)
- `exchanges` (list, optional): Danh sách exchange IDs để thử. Nếu `None`, sử dụng `exchange_manager.public.exchange_priority_for_fallback` (mặc định: `None`)

**Trả về:**
- Luôn trả về `Tuple[pd.DataFrame, str]`: DataFrame chứa đầy đủ OHLCV và exchange cung cấp dữ liệu.
- Trả về `(None, None)` nếu không thể lấy dữ liệu từ bất kỳ exchange nào.
- Sử dụng `DataFetcher.dataframe_to_close_series(df)` nếu bạn cần Series giá `close`.

**Ví dụ cơ bản:**
```python
from modules.DataFetcher import DataFetcher
from modules.ExchangeManager import ExchangeManager

em = ExchangeManager()  # Không cần credentials cho OHLCV
data_fetcher = DataFetcher(em)

# Lấy 1000 candles 1h của BTC/USDT
df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=1000, timeframe="1h")

if df is not None:
    print(f"Lấy được {len(df)} candles từ {exchange_id}")
    print(f"Giá gần nhất: {df['close'].iloc[-1]}")
    print(f"Timestamp gần nhất: {df['timestamp'].iloc[-1]}")
else:
    print("Không thể lấy dữ liệu OHLCV")
```

**Ví dụ với freshness checking:**
```python
# Kiểm tra độ tươi của dữ liệu và thử nhiều exchanges
df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
    "BTC/USDT",
    limit=1000,
    timeframe="1h",
    check_freshness=True
)

if df is not None:
    print(f"Data tươi từ {exchange_id.upper()}")
    print(f"Lấy được {len(df)} candles")
else:
    print("Không thể lấy dữ liệu tươi từ bất kỳ exchange nào")
```

**Ví dụ với custom exchanges:**
```python
# Chỉ thử các exchanges cụ thể
df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
    "BTC/USDT",
    limit=1000,
    timeframe="1h",
    check_freshness=True,
    exchanges=['binance', 'kraken', 'kucoin']
)
```

**Lưu ý:**
- ✅ **Không cần credentials** để lấy OHLCV (dùng public API)
- ✅ **Caching tự động**: Nếu đã fetch cùng symbol/timeframe/limit trước đó, sẽ trả về từ cache (trừ khi `check_freshness=True`)
- ✅ **Fallback tự động**: Tự động thử các exchanges theo thứ tự ưu tiên trong `exchange_manager.public.exchange_priority_for_fallback`
- ✅ **Freshness checking**: Khi `check_freshness=True`, kiểm tra độ tươi của dữ liệu (age <= timeframe * 1.5 minutes, tối thiểu 5 phút) và thử các exchanges khác nếu data cũ
- ✅ **Hỗ trợ shutdown signal**: Kiểm tra và dừng nếu có `shutdown_event`
- ✅ Tự động normalize symbol
- ✅ **Return format**: Luôn trả về `(DataFrame, exchange_id)`; gọi `DataFetcher.dataframe_to_close_series(df)` nếu cần Series
- ✅ Khi `check_freshness=True`, vẫn trả về tuple `(DataFrame, exchange_id)` để biết exchange nào cung cấp data

**Các exchanges được thử (theo thứ tự ưu tiên):**
1. Binance
2. Kraken
3. KuCoin
4. Gate.io
5. OKX
6. Bybit
7. MEXC
8. Huobi

Có thể thay đổi thứ tự ưu tiên qua:
```python
em.public.exchange_priority_for_fallback = ['kraken', 'binance', 'kucoin']
```

**Output mẫu:**
```
  [OHLCV] BTC/USDT loaded from binance (1000 bars)
```

### `dataframe_to_close_series(df: pd.DataFrame) -> pd.Series | None`

**Mục đích**: Chuyển DataFrame OHLCV do `fetch_ohlcv_with_fallback_exchange` trả về thành Series giá `close` với index là timestamp.

**Khi nào dùng:**
- ✅ Cần tính toán dựa trên giá đóng cửa (ví dụ: beta, VaR, correlation)
- ✅ Muốn tái sử dụng logic cũ dựa trên Series

**Trả về:**
- `pd.Series`: Series giá `close` với index là `timestamp`
- `None`: Nếu DataFrame không hợp lệ hoặc rỗng

**Ví dụ:**
```python
df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=1000)
close_series = DataFetcher.dataframe_to_close_series(df)
if close_series is not None:
    returns = close_series.pct_change().dropna()
```

**Ví dụ với nhiều timeframes:**
```python
# Lấy dữ liệu 1h
df_1h, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=1000, timeframe="1h")

# Lấy dữ liệu 4h
df_4h, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=500, timeframe="4h")

# Lấy dữ liệu 1d
df_1d, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
    "BTC/USDT", 
    limit=365, 
    timeframe="1d"
)
```

---

### `should_stop() -> bool`

**Mục đích**: Kiểm tra xem có shutdown signal không (dùng nội bộ).

**Khi nào dùng:**
- ✅ Khi implement custom logic cần kiểm tra shutdown signal
- ✅ Thường không cần gọi trực tiếp (đã được tích hợp trong các phương thức khác)

**Trả về:**
- `bool`: `True` nếu có shutdown signal, `False` nếu không

**Ví dụ:**
```python
# Thường không cần gọi trực tiếp
# Nhưng nếu cần custom logic:
if data_fetcher.should_stop():
    print("Shutdown signal received, stopping...")
    return
```

---

## Ví dụ sử dụng

### Ví dụ 1: Lấy giá hiện tại của nhiều symbols

```python
from modules.DataFetcher import DataFetcher
from modules.ExchangeManager import ExchangeManager

# Khởi tạo
em = ExchangeManager(api_key="your_key", api_secret="your_secret")
data_fetcher = DataFetcher(em)

# Lấy giá
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
data_fetcher.fetch_current_prices_from_binance(symbols)

# Sử dụng giá đã lấy
for symbol in symbols:
    price = data_fetcher.market_prices.get(symbol)
    if price:
        print(f"{symbol}: ${price:,.2f}")
```

### Ví dụ 2: Lấy dữ liệu OHLCV với fallback và freshness checking

```python
from modules.DataFetcher import DataFetcher
from modules.ExchangeManager import ExchangeManager

# Khởi tạo (không cần credentials cho OHLCV)
em = ExchangeManager()
data_fetcher = DataFetcher(em)

# Lấy OHLCV với freshness checking (tự động fallback và tìm data tươi nhất)
df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
    "BTC/USDT", 
    limit=1000, 
    timeframe="1h",
    check_freshness=True
)

if df is not None:
    print(f"Lấy được {len(df)} candles từ {exchange_id.upper()}")
    print(f"Giá đầu tiên: {df['close'].iloc[0]}")
    print(f"Giá cuối cùng: {df['close'].iloc[-1]}")
    print(f"Timestamp cuối: {df['timestamp'].iloc[-1]}")
else:
    print("Không thể lấy dữ liệu từ bất kỳ exchange nào")
```

### Ví dụ 3: Sử dụng với shutdown event (multi-threading)

```python
from modules.DataFetcher import DataFetcher
from modules.ExchangeManager import ExchangeManager
import threading
import time

# Khởi tạo với shutdown event
em = ExchangeManager(api_key="key", api_secret="secret")
shutdown = threading.Event()
data_fetcher = DataFetcher(em, shutdown_event=shutdown)

# Trong một thread khác, có thể set shutdown event
def stop_fetching():
    time.sleep(10)  # Sau 10 giây
    shutdown.set()
    print("Shutdown signal sent")

# Chạy trong thread riêng
threading.Thread(target=stop_fetching, daemon=True).start()

# Fetch sẽ tự động dừng khi shutdown event được set
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
data_fetcher.fetch_current_prices_from_binance(symbols)  # Sẽ dừng nếu shutdown được set
```

### Ví dụ 4: Lấy dữ liệu cho nhiều symbols và timeframes

```python
from modules.DataFetcher import DataFetcher
from modules.ExchangeManager import ExchangeManager

em = ExchangeManager()
data_fetcher = DataFetcher(em)

symbols = ["BTC/USDT", "ETH/USDT"]
timeframes = ["1h", "4h", "1d"]

# Lấy dữ liệu cho tất cả combinations
for symbol in symbols:
    for timeframe in timeframes:
        df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol, 
            limit=1000, 
            timeframe=timeframe
        )
        if df is not None:
            print(f"{symbol} {timeframe}: {len(df)} candles from {exchange_id}")
        else:
            print(f"Failed to fetch {symbol} {timeframe}")
```

### Ví dụ 5: Sử dụng cache để tối ưu

```python
from modules.DataFetcher import DataFetcher
from modules.ExchangeManager import ExchangeManager

em = ExchangeManager()
data_fetcher = DataFetcher(em)

# Lần đầu: Fetch từ exchange
df1, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=1000, timeframe="1h")
# Output: [OHLCV] BTC/USDT loaded from binance (1000 bars)

# Lần hai: Lấy từ cache (nhanh hơn, không cần network call)
df2, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=1000, timeframe="1h")
# Không có output (lấy từ DataFrame cache)

# Cache key dựa trên (symbol, timeframe, limit)
# Nếu thay đổi bất kỳ tham số nào, sẽ fetch lại
df3, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=2000, timeframe="1h")
# Output: [OHLCV] BTC/USDT loaded from binance (2000 bars) - Fetch lại vì limit khác

# Lưu ý: check_freshness=True sẽ bypass cache để đảm bảo data tươi
df_fresh, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
    "BTC/USDT", 
    limit=1000, 
    timeframe="1h",
    check_freshness=True
)
# Sẽ fetch lại từ exchange (không dùng cache)
```

### Ví dụ 6: Tích hợp với portfolio management

```python
from modules.DataFetcher import DataFetcher
from modules.ExchangeManager import ExchangeManager

em = ExchangeManager(api_key="key", api_secret="secret")
data_fetcher = DataFetcher(em)

# Lấy danh sách symbols từ positions
positions = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

# Lấy giá hiện tại
data_fetcher.fetch_current_prices_from_binance(positions)

# Tính tổng giá trị portfolio
total_value = 0
for symbol in positions:
    price = data_fetcher.market_prices.get(symbol)
    if price:
        # Giả sử có 1 unit mỗi symbol
        total_value += price
        print(f"{symbol}: ${price:,.2f}")

print(f"\nTotal portfolio value: ${total_value:,.2f}")
```

---

## Best Practices

### 1. Sử dụng đúng manager cho từng loại dữ liệu

```python
# ✅ ĐÚNG: Dùng authenticated cho prices (cần credentials)
em = ExchangeManager(api_key="key", api_secret="secret")
data_fetcher = DataFetcher(em)
data_fetcher.fetch_current_prices_from_binance(["BTC/USDT"])

# ✅ ĐÚNG: Không cần credentials cho OHLCV
em = ExchangeManager()  # Không cần credentials
data_fetcher = DataFetcher(em)
ohlcv = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT")

# ❌ SAI: Cố gắng fetch prices mà không có credentials
em = ExchangeManager()  # Không có credentials
data_fetcher = DataFetcher(em)
data_fetcher.fetch_current_prices_from_binance(["BTC/USDT"])  # Sẽ fail
```

### 2. Tận dụng caching

```python
# ✅ ĐÚNG: Fetch một lần, dùng nhiều lần
df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=1000, timeframe="1h")
# Sử dụng df nhiều lần mà không cần fetch lại

# ❌ SAI: Fetch lại nhiều lần không cần thiết
for i in range(10):
    df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=1000, timeframe="1h")
    # Mỗi lần đều fetch từ cache, nhưng không cần thiết
```

### 3. Xử lý lỗi đúng cách

```python
# ✅ ĐÚNG: Kiểm tra tuple (DataFrame, exchange_id) trước khi sử dụng
df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT")
if df is not None:
    close_series = DataFetcher.dataframe_to_close_series(df)
    print(f"Got {len(df)} candles from {exchange_id}")
else:
    print("Failed to fetch OHLCV")
    # Xử lý fallback hoặc retry

# ✅ ĐÚNG: Với freshness checking
df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
    "BTC/USDT",
    check_freshness=True
)
if df is not None:
    print(f"Got {len(df)} fresh candles from {exchange_id}")
else:
    print("Failed to fetch OHLCV")

# ❌ SAI: Không kiểm tra None
df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT")
print(len(df))  # Có thể lỗi nếu df là None
```

### 4. Sử dụng shutdown event cho long-running tasks

```python
# ✅ ĐÚNG: Dùng shutdown event cho tasks dài
import threading

shutdown = threading.Event()
data_fetcher = DataFetcher(em, shutdown_event=shutdown)

# Trong một thread khác
def stop_after_timeout():
    time.sleep(60)
    shutdown.set()

threading.Thread(target=stop_after_timeout, daemon=True).start()

# Fetch sẽ tự động dừng khi timeout
data_fetcher.fetch_current_prices_from_binance(large_symbol_list)
```

### 5. Normalize symbols trước khi sử dụng

```python
# ✅ ĐÚNG: DataFetcher tự động normalize
data_fetcher.fetch_current_prices_from_binance(["BTCUSDT", "ETH/USDT", "BNB"])  # Tất cả đều OK

# ✅ ĐÚNG: Hoặc normalize trước
from modules.utils import normalize_symbol
symbols = [normalize_symbol(s) for s in ["BTC", "ETH", "BNB"]]
data_fetcher.fetch_current_prices_from_binance(symbols)
```

### 6. Sử dụng limit hợp lý

```python
# ✅ ĐÚNG: Dùng limit phù hợp với nhu cầu
# Cho phân tích ngắn hạn: 500-1000 candles
df_short, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=1000, timeframe="1h")

# Cho phân tích dài hạn: 365-1000 candles với timeframe lớn hơn
df_long, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=365, timeframe="1d")

# ❌ SAI: Fetch quá nhiều không cần thiết (chậm, tốn bộ nhớ)
df_huge, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange("BTC/USDT", limit=10000, timeframe="1m")
```

---

## Tính năng nâng cao

### 1. Caching Mechanism

DataFetcher tự động cache OHLCV data dựa trên:
- Symbol (normalized, uppercase)
- Timeframe
- Limit

**Cache key format**: `(symbol.upper(), timeframe, int(limit))`

**Lưu ý:**
- Cache chỉ tồn tại trong memory (không persist sau khi restart)
- `_ohlcv_dataframe_cache` lưu `(DataFrame, exchange_id)` để dùng lại nhanh chóng
- Cache bị bypass khi `check_freshness=True` để đảm bảo data tươi

**Ví dụ:**
```python
# Cache key cho "BTC/USDT", "1h", 1000
cache_key = ("BTC/USDT", "1h", 1000)

# Nếu fetch lại với cùng parameters, sẽ lấy từ cache
```

### 2. Fallback Mechanism

Khi fetch OHLCV, DataFetcher tự động thử các exchanges theo thứ tự ưu tiên:

1. Thử exchange đầu tiên trong `exchange_priority_for_fallback`
2. Nếu fail, thử exchange tiếp theo
3. Tiếp tục cho đến khi thành công hoặc hết exchanges
4. Nếu tất cả đều fail, trả về `None`

**Thay đổi thứ tự ưu tiên:**
```python
# Mặc định: ['binance', 'kraken', 'kucoin', 'gate', 'okx', 'bybit', 'mexc', 'huobi']
em.public.exchange_priority_for_fallback = ['kraken', 'binance', 'kucoin']
```

### 3. Shutdown Support

DataFetcher hỗ trợ graceful shutdown thông qua `shutdown_event`:

- `fetch_current_prices_from_binance()`: Kiểm tra `should_stop()` trước mỗi symbol
- `fetch_ohlcv_with_fallback_exchange()`: Kiểm tra `should_stop()` trước mỗi exchange attempt

**Use case:**
- Long-running tasks cần có khả năng dừng
- Multi-threading applications
- Background tasks cần responsive shutdown

### 4. Progress Tracking

`fetch_current_prices_from_binance()` tự động hiển thị progress bar khi fetch nhiều symbols:

- Sử dụng `ProgressBar` class
- Hiển thị progress real-time
- Tự động finish khi hoàn thành

### 5. Error Handling

DataFetcher xử lý lỗi một cách graceful:

- **fetch_current_prices_from_binance()**: Tiếp tục fetch các symbols khác nếu một symbol fail
- **fetch_ohlcv_with_fallback_exchange()**: Tự động thử exchange khác nếu một exchange fail
- In thông báo lỗi rõ ràng với colorama
- Trả về `None` thay vì raise exception (cho `fetch_ohlcv_with_fallback_exchange()`)

---

## Tóm tắt

| Tính năng | Mô tả |
|-----------|-------|
| **fetch_current_prices_from_binance()** | Lấy giá hiện tại từ Binance (cần credentials) |
| **fetch_ohlcv_with_fallback_exchange()** | Lấy OHLCV với fallback, caching, và tùy chọn freshness checking (không cần credentials) |
| **Caching** | Tự động cache OHLCV data (riêng cho Series và DataFrame) để tối ưu performance |
| **Fallback** | Tự động thử các exchanges khác nếu một exchange fail |
| **Freshness Checking** | Kiểm tra độ tươi của dữ liệu và tìm exchange có data tươi nhất |
| **Return Options** | Có thể trả về Series (mặc định) hoặc full DataFrame với exchange_id |
| **Shutdown Support** | Hỗ trợ graceful shutdown với threading.Event |
| **Progress Tracking** | Hiển thị progress bar khi fetch prices |
| **Error Handling** | Xử lý lỗi graceful, không crash |

### Khi nào dùng DataFetcher?

✅ **Nên dùng** khi:
- Cần lấy giá hiện tại của nhiều symbols
- Cần lấy OHLCV với fallback tự động
- Cần caching để tối ưu performance
- Cần kiểm tra độ tươi của dữ liệu (freshness checking)
- Cần full DataFrame với tất cả OHLCV columns (hàm luôn trả về DataFrame)
- Cần progress tracking
- Cần shutdown support

❌ **Không nên dùng** khi:
- Chỉ cần lấy dữ liệu từ một exchange cụ thể (dùng ExchangeManager trực tiếp)
- Cần custom error handling phức tạp

---

## Liên kết

- [ExchangeManager Documentation](./ExchangeManager.md) - Tài liệu về ExchangeManager
- [ccxt Documentation](https://docs.ccxt.com/) - Tài liệu về ccxt library

