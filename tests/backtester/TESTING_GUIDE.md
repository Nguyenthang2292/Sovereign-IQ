# Hướng dẫn Testing cho Backtester Module

## Vấn đề: API Rate Limits

Khi chạy tests, các signal calculators gọi API từ Binance, gây ra rate limit errors:
- "Your operation is too frequent, please try again later"
- Tests chạy chậm do network latency
- Tests không reliable do phụ thuộc vào network

## Giải pháp đã implement

### 1. Mock Data Fetcher (Recommended)

**File**: `tests/backtester/conftest.py`

Sử dụng fixtures để mock toàn bộ data fetching:

```python
def test_my_backtest(mock_data_fetcher):
    # mock_data_fetcher đã được mock, không gọi API
    backtester = FullBacktester(mock_data_fetcher)
    result = backtester.backtest(...)
```

### 2. Mock Signal Calculators (Automatic)

**File**: `tests/backtester/conftest.py`

Fixture `auto_mock_signal_calculators` (autouse=True) tự động mock tất cả signal calculators:

- `get_range_oscillator_signal` → returns (1, 0.7)
- `get_spc_signal` → returns (1, 0.6)
- `get_xgboost_signal` → returns (1, 0.8)
- `get_hmm_signal` → returns (1, 0.65)
- `get_random_forest_signal` → returns (1, 0.75)

**Không cần làm gì thêm** - fixture tự động được apply!

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

## Alternative: CCXT với Cached Data

Nếu muốn test với real data (không recommended cho CI/CD):

Xem file `CCXT_CACHED_DATA_EXAMPLE.py` để biết cách implement.

**Lưu ý**:
- Vẫn có thể gặp rate limit lần đầu fetch
- Cần network connection
- Tests chạy chậm hơn
- Không reliable cho CI/CD

**Recommended**: Sử dụng mock data hoàn toàn.

## Best Practices

1. **Luôn sử dụng mock_data_fetcher fixture** - không tạo DataFetcher thật trong tests
2. **Không cần patch signal calculators** - đã được auto-mock
3. **Override khi cần** - patch lại trong test nếu cần signals khác
4. **Test với various scenarios** - sử dụng mock_ohlcv_data để tạo different data patterns

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

```
tests/backtester/
├── conftest.py              # Shared fixtures (auto-mock)
├── test_helpers.py          # Helper utilities
├── test_parallel_processing.py
├── test_performance.py
├── test_edge_cases.py
├── README.md                # This file
└── CCXT_CACHED_DATA_EXAMPLE.py  # Alternative approach
```

