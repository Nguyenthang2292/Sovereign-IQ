# Backtester Tests - Avoiding API Rate Limits

## Vấn đề

Các test cases trong `tests/backtester` trước đây đang gọi API thật từ Binance, gây ra rate limit errors.

## Giải pháp

### 1. Shared Fixtures (conftest.py)

File `tests/backtester/conftest.py` cung cấp các fixtures được tự động sử dụng:

- **`mock_data_fetcher`**: Mock DataFetcher hoàn chỉnh, không gọi API
- **`mock_signal_calculators`** (autouse=True): Tự động mock tất cả signal calculators
- **`mock_ohlcv_data`**: Helper để generate mock OHLCV data

### 2. Test Helpers (test_helpers.py)

File `tests/backtester/test_helpers.py` cung cấp context managers:

```python
from tests.backtester.test_helpers import mock_all_signal_calculators, mock_no_signals

# Mock với signals mặc định
with mock_all_signal_calculators():
    result = backtester.backtest(...)

# Mock với no signals
with mock_no_signals():
    result = backtester.backtest(...)
```

## Cách sử dụng

### Option 1: Sử dụng fixtures tự động (Recommended)

Fixtures trong `conftest.py` sẽ tự động được áp dụng cho tất cả tests:

```python
def test_my_backtest(mock_data_fetcher):
    # mock_signal_calculators đã được tự động apply
    backtester = FullBacktester(mock_data_fetcher)
    result = backtester.backtest(...)
```

### Option 2: Sử dụng context managers

Nếu cần customize signals:

```python
from tests.backtester.test_helpers import mock_all_signal_calculators

def test_custom_signals(mock_data_fetcher):
    with mock_all_signal_calculators(
        osc_signal=1, osc_confidence=0.8,
        spc_signal=-1, spc_confidence=0.6,  # SHORT signal
    ):
        backtester = FullBacktester(mock_data_fetcher)
        result = backtester.backtest(...)
```

### Option 3: Manual patching

Nếu cần full control:

```python
from unittest.mock import patch

def test_manual_mock(mock_data_fetcher):
    with patch('core.signal_calculators.get_range_oscillator_signal', return_value=(1, 0.7)), \
         patch('core.signal_calculators.get_spc_signal', return_value=(1, 0.6)):
        backtester = FullBacktester(mock_data_fetcher)
        result = backtester.backtest(...)
```

## Mock Data Generation

`mock_ohlcv_data` fixture tạo realistic OHLCV data:

```python
def test_with_custom_data(mock_ohlcv_data):
    # Generate custom data
    df = mock_ohlcv_data(periods=500, base_price=50000.0, volatility=1.0)
    
    def custom_fetch(symbol, **kwargs):
        return df, "binance"
    
    fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=custom_fetch,
    )
    
    backtester = FullBacktester(fetcher)
    result = backtester.backtest(...)
```

## Lưu ý

1. **Autouse fixture**: `mock_signal_calculators` được set `autouse=True`, nghĩa là nó sẽ tự động được áp dụng cho tất cả tests trong thư mục này.

2. **Override khi cần**: Nếu test cần signals khác, có thể override bằng cách patch lại trong test function.

3. **Không cần credentials**: Tất cả tests sử dụng mock data, không cần API credentials.

4. **Fast execution**: Tests chạy nhanh vì không có network calls.

## Alternative: Sử dụng CCXT với cached data

Nếu muốn test với real data nhưng tránh rate limit, có thể:

1. Cache data từ ccxt vào file
2. Load cached data trong tests
3. Chỉ fetch mới khi cache expired

Ví dụ implementation:

```python
import ccxt
import pickle
from pathlib import Path

CACHE_DIR = Path("tests/cache")

def get_cached_ohlcv(symbol, timeframe, limit):
    cache_file = CACHE_DIR / f"{symbol}_{timeframe}_{limit}.pkl"
    
    if cache_file.exists():
        # Load from cache
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        # Fetch from exchange (no credentials needed for public data)
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Cache it
        cache_file.parent.mkdir(exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        
        return df
```

Tuy nhiên, cách này vẫn có thể gặp rate limit nếu cache chưa có. **Recommended approach**: Sử dụng mock data hoàn toàn như đã implement.

