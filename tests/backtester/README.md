# Backtester Tests - Avoiding API Rate Limits

## Vấn đề

Các test cases trong `tests/backtester` trước đây đang gọi API thật từ Binance thông qua:

1. `DataFetcher.fetch_ohlcv_with_fallback_exchange()` - Fetch OHLCV data
2. Signal calculators (`get_range_oscillator_signal`, `get_spc_signal`, etc.) - Fetch data để tính signals

Điều này gây ra:

- ❌ Rate limit errors: "Your operation is too frequent, please try again later"
- ❌ Tests chạy chậm do network latency
- ❌ Tests không reliable, phụ thuộc vào network
- ❌ Cần API credentials hoặc có thể bị block IP

## Giải pháp đã implement

### 1. Mock Data Fetcher (Recommended)

**File**: `tests/backtester/conftest.py`

Tạo `mock_data_fetcher` fixture để mock toàn bộ data fetching:

```python
@pytest.fixture
def mock_data_fetcher(mock_ohlcv_data):
    """Create a fully mocked DataFetcher that doesn't call real APIs."""
    def fake_fetch(symbol, **kwargs):
        limit = kwargs.get('limit', 200)
        df = mock_ohlcv_data(periods=limit)
        return df, "binance"
    
    return SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
    )
```

Sử dụng fixtures để mock toàn bộ data fetching:

```python
def test_my_backtest(mock_data_fetcher):
    # mock_data_fetcher đã được mock, không gọi API
    backtester = FullBacktester(mock_data_fetcher)
    result = backtester.backtest(...)
```

### 2. Auto-mock Signal Calculators (Automatic)

**File**: `tests/backtester/conftest.py`

Fixture `auto_mock_signal_calculators` với `autouse=True` tự động mock tất cả signal calculators:

- `get_range_oscillator_signal` → returns (1, 0.7)
- `get_spc_signal` → returns (1, 0.6)
- `get_xgboost_signal` → returns (1, 0.8)
- `get_hmm_signal` → returns (1, 0.65)
- `get_random_forest_signal` → returns (1, 0.75)

**Không cần làm gì thêm** - fixture tự động được apply!

**Lợi ích**:

- ✅ Tự động apply cho tất cả tests
- ✅ Không cần thêm code trong mỗi test
- ✅ Có thể override khi cần

### 3. Test Helpers

**File**: `tests/backtester/test_helpers.py`

Context managers để customize signals:

```python
from tests.backtester.test_helpers import mock_all_signal_calculators, mock_no_signals

# Custom signals
with mock_all_signal_calculators(
    osc_signal=1, osc_confidence=0.8,
    spc_signal=-1, spc_confidence=0.6,  # SHORT
):
    result = backtester.backtest(...)

# No signals
with mock_no_signals():
    result = backtester.backtest(...)
```

## Cách sử dụng

### Basic Test (Recommended)

```python
def test_backtest_basic(mock_data_fetcher):
    """Test với mock data và mock signals (automatic)."""
    backtester = FullBacktester(mock_data_fetcher)
    
    result = backtester.backtest(
        symbol="BTC/USDT",
        timeframe="1h",
        lookback=200,
        signal_type="LONG",
    )
    
    assert 'trades' in result
    assert 'metrics' in result
```

**Không cần làm gì thêm** - `auto_mock_signal_calculators` đã tự động mock!

### Custom Signals

```python
def test_backtest_custom_signals(mock_data_fetcher):
    """Test với custom signals."""
    from tests.backtester.test_helpers import mock_all_signal_calculators
    
    with mock_all_signal_calculators(
        osc_signal=-1,  # SHORT signal
        osc_confidence=0.9,
    ):
        backtester = FullBacktester(mock_data_fetcher)
        result = backtester.backtest(...)
```

### Override Signals khi cần

```python
def test_backtest_no_signals(mock_data_fetcher):
    """Test với no signals."""
    from tests.backtester.test_helpers import mock_no_signals
    
    with mock_no_signals():
        backtester = FullBacktester(mock_data_fetcher)
        result = backtester.backtest(...)
        assert result['metrics']['num_trades'] == 0
```

### Custom Data

```python
def test_backtest_custom_data(mock_ohlcv_data):
    """Test với custom OHLCV data."""
    # Generate custom data
    df = mock_ohlcv_data(
        periods=500,
        base_price=50000.0,
        volatility=2.0  # High volatility
    )
    
    def custom_fetch(symbol, **kwargs):
        return df, "binance"
    
    fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=custom_fetch,
    )
    
    backtester = FullBacktester(fetcher)
    result = backtester.backtest(...)
```

### Manual Patching (Full Control)

Nếu cần full control:

```python
from unittest.mock import patch

def test_manual_mock(mock_data_fetcher):
    with patch('core.signal_calculators.get_range_oscillator_signal', return_value=(1, 0.7)), \
         patch('core.signal_calculators.get_spc_signal', return_value=(1, 0.6)):
        backtester = FullBacktester(mock_data_fetcher)
        result = backtester.backtest(...)
```

## Best Practices

1. **Luôn sử dụng mock_data_fetcher fixture** - không tạo DataFetcher thật trong tests
2. **Không cần patch signal calculators** - đã được auto-mock
3. **Override khi cần** - patch lại trong test nếu cần signals khác
4. **Test với various scenarios** - sử dụng mock_ohlcv_data để tạo different data patterns

## Lưu ý

1. **Autouse fixture**: `auto_mock_signal_calculators` được set `autouse=True`, nghĩa là nó sẽ tự động được áp dụng cho tất cả tests trong thư mục này.

2. **Override khi cần**: Nếu test cần signals khác, có thể override bằng cách patch lại trong test function.

3. **Không cần credentials**: Tất cả tests sử dụng mock data, không cần API credentials.

4. **Fast execution**: Tests chạy nhanh vì không có network calls.

## Alternative: CCXT với Cached Data

Nếu muốn test với real data (không recommended cho CI/CD):

Xem file `CCXT_CACHED_DATA_EXAMPLE.py` để biết cách implement.

**Lưu ý**:

- Vẫn có thể gặp rate limit lần đầu fetch
- Cần network connection
- Tests chạy chậm hơn
- Không reliable cho CI/CD

**Recommended**: Sử dụng mock data hoàn toàn như đã implement.

## Kết quả

### Trước khi fix

- ❌ Tests gọi API thật → Rate limit errors
- ❌ Tests chạy chậm (network latency)
- ❌ Tests không reliable

### Sau khi fix

- ✅ Không còn API calls → Không còn rate limit
- ✅ Tests chạy nhanh (no network)
- ✅ Tests reliable và deterministic
- ✅ Không cần API credentials
- ✅ Có thể chạy offline

## Verification

Để verify rằng tests không còn gọi API:

1. **Disconnect network** và chạy tests - should pass
2. **Check logs** - không có network requests
3. **Run tests multiple times** - không có rate limit errors

## Troubleshooting

### Test vẫn gọi API?

1. Kiểm tra xem có import đúng fixtures không
2. Đảm bảo `auto_mock_signal_calculators` fixture được load
3. Kiểm tra xem có patch nào override không

### Test fails với "rate limit"?

1. Đảm bảo đang sử dụng `mock_data_fetcher` fixture
2. Kiểm tra xem signal calculators có được mock không
3. Thêm explicit patches nếu cần

### Muốn test với real data?

1. Sử dụng `CCXT_CACHED_DATA_EXAMPLE.py` approach
2. Hoặc tạo fixture riêng với real DataFetcher (không recommended)

## Files Structure

```text
tests/backtester/
├── conftest.py              # Shared fixtures (auto-mock)
├── test_helpers.py          # Helper utilities
├── test_parallel_processing.py
├── test_performance.py
├── test_edge_cases.py
├── test_dataframe_parameter.py  # DataFrame optimization tests
├── README.md                # This file
└── CCXT_CACHED_DATA_EXAMPLE.py  # Alternative approach
```

## Files đã được cập nhật

### Tests đã fix

- ✅ `tests/backtester/test_parallel_processing.py` - Thêm mock_signal_calculators
- ✅ `tests/backtester/test_performance.py` - Thêm mock_signal_calculators
- ✅ `tests/backtester/test_edge_cases.py` - Thêm explicit patches
- ✅ `tests/backtester/test_dataframe_parameter.py` - DataFrame optimization tests
- ✅ `tests/position_sizing/test_backtester.py` - Thêm explicit patches

### Files mới

- ✅ `tests/backtester/conftest.py` - Shared fixtures với auto-mock
- ✅ `tests/backtester/test_helpers.py` - Helper utilities
- ✅ `tests/backtester/README.md` - Documentation (this file)
- ✅ `tests/backtester/CCXT_CACHED_DATA_EXAMPLE.py` - Alternative approach
- ✅ `tests/position_sizing/conftest.py` - Shared fixtures cho position_sizing tests
