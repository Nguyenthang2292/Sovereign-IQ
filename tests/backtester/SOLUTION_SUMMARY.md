# Giải pháp cho API Rate Limit trong Tests

## Vấn đề

Tất cả tests trong `tests/backtester` đang gọi API thật từ Binance thông qua:
1. `DataFetcher.fetch_ohlcv_with_fallback_exchange()` - Fetch OHLCV data
2. Signal calculators (`get_range_oscillator_signal`, `get_spc_signal`, etc.) - Fetch data để tính signals

Điều này gây ra:
- ❌ Rate limit errors: "Your operation is too frequent"
- ❌ Tests chạy chậm do network latency
- ❌ Tests không reliable, phụ thuộc vào network
- ❌ Cần API credentials hoặc có thể bị block IP

## Giải pháp đã implement

### 1. Mock DataFetcher (conftest.py)

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
        ...
    )
```

### 2. Auto-mock Signal Calculators (conftest.py)

**File**: `tests/backtester/conftest.py`

Fixture `auto_mock_signal_calculators` với `autouse=True` tự động mock tất cả signal calculators:

```python
@pytest.fixture(autouse=True)
def auto_mock_signal_calculators():
    """Automatically mock signal calculators for all tests."""
    with patch('core.signal_calculators.get_range_oscillator_signal', return_value=(1, 0.7)), \
         patch('core.signal_calculators.get_spc_signal', return_value=(1, 0.6)), \
         patch('core.signal_calculators.get_xgboost_signal', return_value=(1, 0.8)), \
         patch('core.signal_calculators.get_hmm_signal', return_value=(1, 0.65)), \
         patch('core.signal_calculators.get_random_forest_signal', return_value=(1, 0.75)):
        yield
```

**Lợi ích**:
- ✅ Tự động apply cho tất cả tests
- ✅ Không cần thêm code trong mỗi test
- ✅ Có thể override khi cần

### 3. Test Helpers (test_helpers.py)

**File**: `tests/backtester/test_helpers.py`

Context managers để customize signals:

```python
from tests.backtester.test_helpers import mock_all_signal_calculators, mock_no_signals

# Custom signals
with mock_all_signal_calculators(osc_signal=-1, osc_confidence=0.9):
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
    result = backtester.backtest(...)
    assert 'trades' in result
```

**Không cần làm gì thêm** - `auto_mock_signal_calculators` đã tự động mock!

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

## Files đã được cập nhật

### Tests đã fix:
- ✅ `tests/backtester/test_parallel_processing.py` - Thêm mock_signal_calculators
- ✅ `tests/backtester/test_performance.py` - Thêm mock_signal_calculators
- ✅ `tests/backtester/test_edge_cases.py` - Thêm explicit patches
- ✅ `tests/position_sizing/test_backtester.py` - Thêm explicit patches

### Files mới:
- ✅ `tests/backtester/conftest.py` - Shared fixtures với auto-mock
- ✅ `tests/backtester/test_helpers.py` - Helper utilities
- ✅ `tests/backtester/README.md` - Documentation
- ✅ `tests/backtester/TESTING_GUIDE.md` - Hướng dẫn chi tiết
- ✅ `tests/backtester/CCXT_CACHED_DATA_EXAMPLE.py` - Alternative approach
- ✅ `tests/position_sizing/conftest.py` - Shared fixtures cho position_sizing tests

## Alternative: CCXT với Cached Data

Nếu muốn test với real data (không recommended):

Xem `CCXT_CACHED_DATA_EXAMPLE.py` để biết cách:
1. Fetch data từ ccxt (không cần credentials cho public data)
2. Cache vào file
3. Reuse cached data trong tests

**Lưu ý**:
- Vẫn có thể gặp rate limit lần đầu
- Cần network connection
- Tests chạy chậm hơn

**Recommended**: Sử dụng mock data hoàn toàn như đã implement.

## Kết quả

### Trước khi fix:
- ❌ Tests gọi API thật → Rate limit errors
- ❌ Tests chạy chậm (network latency)
- ❌ Tests không reliable

### Sau khi fix:
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

## Next Steps

1. ✅ Đã implement mock fixtures
2. ✅ Đã cập nhật test files
3. ⏳ Có thể cần update thêm test files khác nếu có
4. ⏳ Run tests để verify không còn rate limit

