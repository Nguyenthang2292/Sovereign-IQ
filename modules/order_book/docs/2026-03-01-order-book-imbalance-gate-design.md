# Order Book Imbalance Gate — Design Document

**Date**: 2026-03-01  
**Author**: Brainstorming session  
**Status**: Validated  

---

## 1. Mục tiêu

Tích hợp **Order Book Imbalance (OBI)** và **Cumulative Delta (Trade Flow)** như một **Order Book Imbalance Gate** — confirmation gate vào cuối pipeline `auto_trade`, nhằm cải thiện timing vào lệnh bằng cách xác nhận rằng áp lực orderbook và dòng tiền khớp lệnh thực tế ủng hộ hướng của signal trước khi đặt lệnh.

### Vấn đề hiện tại

Pipeline hiện có (ATC → XGBoost → Gemini → SignalSelector) phân tích hoàn toàn dựa trên **dữ liệu lịch sử** (OHLCV) và **chart patterns**. Không có thành phần nào kiểm tra trạng thái **thị trường real-time** tại thời điểm vào lệnh. Điều này dẫn tới nguy cơ:

- Vào lệnh LONG ngay khi "cá mập" đang xả lệnh (áp lực bán mạnh).
- Dễ bị lừa bởi các lệnh giả (Spoofing) nếu chỉ phân tích sổ lệnh đơn thuần.
- **Adverse Selection**: bị khớp lệnh ngay trước khi giá chạy ngược hướng.

### Giải pháp

Thêm module `order_book` cung cấp một **Order Book Imbalance Gate** — gate cuối cùng trong `OrderExecutor`. Gate này sẽ:

1. Đọc dữ liệu sổ lệnh (Depth) để tính OBI trên các "bins" gộp (tránh spoofing).
2. Đọc dữ liệu giao dịch 5-10 phút gần nhất (Aggregate Trades) để tính Cumulative Delta.
3. Chấm điểm tổng hợp (40% OBI + 60% Delta) để quyết định PASS, RETRY, hay SKIP.

Gate này **không thay thế** bất kỳ filter nào mà bổ sung một lớp kiểm tra real-time bằng luồng dữ liệu REST (nhẹ nhàng, phù hợp MFT/Scalping).

---

## 2. Kiến trúc tổng thể

### Luồng tích hợp

```
SignalPipeline.run_pipeline()
    └── FinalSignal (symbol, signal_type, entry, sl, tp)
            ↓
OrderExecutor.execute_from_signal(signal_dict)
    ├── fetch_ticker(symbol)                                         [đã có]
    ├── [NEW] OrderBookImbalanceGate.check(symbol, signal_type)   ← điểm tích hợp
    │       ├── PASS  → tiếp tục
    │       └── SKIP  → return {success: False, skipped: True}
    └── FinalSignal → OrderManager → Binance API                    [đã có]
```

### Tại sao tích hợp trong `OrderExecutor`?

- Đây là điểm **gần nhất với lệnh thực tế** — decision tại đây có giá trị timing cao nhất, kết hợp dữ liệu 5 phút sát nhất với thời điểm phát sinh signal.
- Không làm phức tạp hoặc chậm pipeline scan (ATC/XGBoost/Gemini chạy hàng phút).
- Backward-compatible: Order Book Imbalance Gate có thể disable hoàn toàn nếu không cấu hình.

---

## 3. File Structure

```
modules/order_book/
├── __init__.py
├── models.py                              # Data models: OrderBookSnapshot, AggTrade, CombinedResult
├── market_data_fetcher.py                 # Fetch Binance Futures depth & aggTrades qua REST
├── order_book_imbalance_calculator.py     # Tính Order Book Imbalance, Cumulative Delta, Combined Score
└── order_book_imbalance_gate.py           # OrderBookImbalanceGate với delay-retry logic

tests/modules/order_book/
├── __init__.py
├── test_models.py
├── test_order_book_imbalance_calculator.py
├── test_market_data_fetcher.py
└── test_order_book_imbalance_gate.py
```

---

## 4. Data Models (`models.py`)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


@dataclass
class OrderBookSnapshot:
    """Raw orderbook snapshot từ Binance."""
    symbol: str
    bids: List[Tuple[float, float]]  # [(price, qty), ...] sorted desc
    asks: List[Tuple[float, float]]  # [(price, qty), ...] sorted asc
    timestamp: float

@dataclass
class AggTrade:
    """Aggregated trades snapshot."""
    price: float
    quantity: float
    timestamp: float
    is_buyer_maker: bool   # True = Sell aggressive, False = Buy aggressive

@dataclass
class CombinedResult:
    """Kết quả tính toán tổng hợp OBI và Delta."""
    obi_score: float           # Normalized OBI từ aggregated bins: -1.0 → +1.0
    delta_score: float         # Normalized Cumulative Delta: -1.0 → +1.0
    combined_score: float      # Score tổng (40% OBI + 60% Delta): -1.0 → +1.0
    obi_raw: float             # Giá trị raw OBI
    delta_raw: float           # Giá trị raw Cumulative Delta
    weighted_mid: float        # Weighted mid-price theo volume

class OBIDecision(str, Enum):
    PASS  = "PASS"   # Score tổng hợp xác nhận hướng signal → vào lệnh
    SKIP  = "SKIP"   # Sau retry vẫn conflict → bỏ signal này
```

---

## 5. Order Book Imbalance & Delta Calculator (`order_book_imbalance_calculator.py`)

### Công thức Aggregated OBI (40% Trọng số)

Thay vì tính OBI trên từng bước giá (dễ bị spoofing), dữ liệu sẽ được gộp vào vùng giá (bins) ví dụ 0.1% biên độ từ Mid-price:
$$I_{bins} = \frac{\sum Q_{bid\_bins} - \sum Q_{ask\_bins}}{\sum Q_{bid\_bins} + \sum Q_{ask\_bins}}$$

- `Q_bid_bins` = tổng khối lượng gom theo bins phía bid
- Kết quả $I_{bins} \in [-1, +1]$

### Công thức Cumulative Delta (60% Trọng số)

Đo lường sự hung hãn của dòng tiền thực tế trong 5-10 phút qua:
$$\Delta_{cum} = \sum Q_{Buy\_Market} - \sum Q_{Sell\_Market}$$

Bên mua chủ động `is_buyer_maker = False`. Bên bán chủ động `is_buyer_maker = True`.

Chuẩn hóa Delta về khoảng `[-1, +1]` bằng hàm `tanh` (tránh outlier làm lệch score):
$$\hat{\Delta}_{norm} = \tanh\left( \frac{\Delta_{cum}}{V_{avg\_5m}} \right)$$

### Combined Score

$$S_{combined} = 0.4 \cdot I_{bins} + 0.6 \cdot \hat{\Delta}_{norm}$$

### Key function

```python
def calculate_combined_score(
    snapshot: OrderBookSnapshot, 
    trades: List[AggTrade],
    bin_step_pct: float = 0.001
) -> CombinedResult:
    """
    Tính Toán Combined Score (40% Aggregated OBI + 60% normalized Cumulative Delta).
    """
```

---

## 6. Market Data Fetcher (`market_data_fetcher.py`)

Cung cấp các API REST gọi qua Binance Futures.

### Endpoint 1: Depth Snapshot

```
GET https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=100
```

- `limit=100` để có đủ dữ liệu gom bins an toàn.

### Endpoint 2: Aggregate Trades

```
GET https://fapi.binance.com/fapi/v1/aggTrades?symbol=BTCUSDT&startTime=<now-5m>&endTime=<now>
```

- Truy xuất các trades gộp trong khoảng 5 phút quá khứ để tính Cumulative Delta.

**Fail-open policy**: Nếu API call thất bại (timeout, network error), hàm trả về `None`. Khi dữ liệu `None`, Order Book Imbalance Gate sẽ **PASS** (trạng thái neutral) với warning log, không bao giờ block trade tốt chỉ vì lỗi mạng nhất thời.

---

## 7. Order Book Imbalance Gate (`order_book_imbalance_gate.py`)

### Delay-Retry Logic

Khi Combined Score conflict với signal direction:

1. Ghi log warning.
2. Sleep `retry_wait_seconds` (mặc định 30s).
3. Fetch depth & trades snapshot, tính lại Combined Score.
4. Trong `max_retries` lần: nếu Combined Score đồng thuận chiều → PASS.
5. Sau `max_retries` vẫn conflict → SKIP.

**Tổng thời gian chờ tối đa**: `max_retries × retry_wait_seconds` = 2 × 30s = 60s

### Alignment Rule

| Signal | Combined Score ($S_{combined}$) condition | Decision |
|--------|-------------------------------------------|----------|
| LONG   | Score > +threshold | PASS |
| LONG   | Score ≈ 0 (\|Score\| < threshold) | PASS (neutral không block) |
| LONG   | Score < -threshold | conflict → retry |
| SHORT  | Score < -threshold | PASS |
| SHORT  | Score ≈ 0 (\|Score\| < threshold) | PASS (neutral không block) |
| SHORT  | Score > +threshold | conflict → retry |

**Tại sao `|Score| < threshold` là PASS?**  
Khi lực lượng cân đối (Combined Score ≈ 0), không có lực nào đẩy ngược lại signal. An toàn để vào lệnh. Chỉ block khi score **thực sự cảnh báo rủi ro bẻ gãy xu hướng**.

### Interface

```python
class OrderBookImbalanceGate:
    def __init__(
        self,
        threshold: float = 0.15,        # |Score| threshold để detect conflict
        retry_wait_seconds: int = 30,   # Giây chờ giữa các lần retry
        max_retries: int = 2,           # Số lần retry tối đa
        delta_window_minutes: int = 5,  # Số phút dữ liệu quá khứ cho Delta
        testnet: bool = False,
        enabled: bool = True,
    ): ...

    def check(
        self,
        symbol: str,
        signal_type: str   # "LONG" hoặc "SHORT"
    ) -> tuple[OBIDecision, Optional[CombinedResult]]:
        """
        Kiểm tra tổng hợp OBI và CumDelta với delay-retry.

        Returns:
            (OBIDecision.PASS, result) hoặc (OBIDecision.SKIP, last_result)
        """
```

---

## 8. Tích hợp vào `OrderExecutor`

### Thay đổi `__init__`

```python
def __init__(
    self,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: Optional[bool] = None,
    dry_run: bool = False,
    recovery_manager: Optional[Any] = None,
    order_book_imbalance_config: Optional[Dict[str, Any]] = None,   # [NEW]
):
    ...
    # [NEW] Order Book Imbalance Gate — disabled by default nếu không có config
    if order_book_imbalance_config is not None:
        from modules.order_book.order_book_imbalance_gate import OrderBookImbalanceGate
        self._order_book_imbalance_gate: Optional[OrderBookImbalanceGate] = OrderBookImbalanceGate(
            testnet=self._testnet,
            **order_book_imbalance_config,
        )
    else:
        self._order_book_imbalance_gate = None
```

### Thay đổi `execute_from_signal`

```python
# Sau khi fetch ticker, trước khi tạo FinalSignal:

ticker = self._fetch_ticker(symbol)
entry = float(ticker.get("last", 0))

# [NEW] Order Book Imbalance Gate Check
if self._order_book_imbalance_gate is not None:
    decision, combined_result = self._order_book_imbalance_gate.check(symbol, signal_type)
    score_str = f"{combined_result.combined_score:.3f}" if combined_result else "N/A"
    
    if decision == OBIDecision.SKIP:
        log_warn(
            f"[OrderBookImbalanceGate] {symbol} {signal_type} SKIPPED after retry. "
            f"Combined Score={score_str} opposes direction."
        )
        return {
            "success": False,
            "skipped": True,
            "reason": f"ORDER_BOOK_IMBALANCE_CONFLICT (score={score_str})",
        }
    log_info(
        f"[OrderBookImbalanceGate] {symbol} PASSED "
        f"(Combined Score={score_str}, decision={decision})"
    )

# ... tiếp tục tạo FinalSignal như bình thường
```

---

## 9. Configuration

### Trong `auto_trade_config.py` / `settings.yaml`

```yaml
order_book_imbalance:
  enabled: true
  threshold: 0.15                # |Score| cần vượt để phát hiện conflict
  retry_wait_seconds: 30         # Giây chờ giữa retries
  max_retries: 2                 # Số lần retry tối đa
  delta_window_minutes: 5        # Số phút để lấy CumDelta
```

### Truyền vào OrderExecutor

```python
cfg = config.get("order_book_imbalance", {})
executor = OrderExecutor(
    ...,
    order_book_imbalance_config=cfg if cfg.get("enabled") else None
)
```

---

## 10. Testing Strategy

### Test cases quan trọng

| Scenario | Combined Score | Expected |
|----------|----------------|----------|
| LONG, Score dương ngay lần 1 | +0.4 | PASS |
| LONG, Score gần zero | +0.05 | PASS (neutral) |
| LONG, Score âm lần 1 → dương lần 2 | -0.3 → +0.2 | PASS sau retry |
| LONG, Score âm cả 3 lần | -0.4 → -0.3 → -0.2 | SKIP |
| SHORT, Score âm ngay | -0.5 | PASS |
| REST API lỗi | None | PASS với warning (Fail Open) |
| OBI âm mạnh (-0.5), Delta dương mạnh (+0.8) | Tổng ~+0.28 (dương) | Hướng theo Delta (Trọng số 60%) |

### Files

```
tests/modules/order_book/
├── test_order_book_imbalance_calculator.py  # Unit: OBI Bins, Delta tanh normalization, 40-60 ratio
├── test_market_data_fetcher.py              # Mock HTTP: normal response, timeout, error
└── test_order_book_imbalance_gate.py        # Integration: PASS/SKIP scenarios, retry timing
```

---

## 11. Rủi ro & Mitigations

| Rủi ro | Mitigation |
|--------|------------|
| Spoofing đánh lừa Orderbook | Dùng Aggregation Bins (thay vì các row đầu tiên); thêm 60% trọng số cho Delta (dòng tiền thực, muốn spoofing bắt buộc phải tốn phí tạo vol). |
| REST API `/aggTrades` timeout do nhiều trades | Binance FAPI cho phép lấy 1000 aggTrades mỗi query. Lọc window = 5 mins thường rất an toàn. Chấp nhận Fail Open nếu timeout. |
| Gate block quá nhiều trade tốt | threshold thấp (0.15), fail-open khi API lỗi, tunable. Score C2 giúp linh hoạt hơn. |
| Retry block 60s quá lâu | Configurable, `max_retries=0` để vô hiệu hóa retry. |

---

## 12. Out of Scope (YAGNI)

Những thứ **không** làm trong scope này:

- ❌ Market Making (MM) — quá phức tạp cho cá nhân, cần separate project.
- ❌ Persistent WebSocket Data Stream — overkill cho MFT, REST đủ dùng cho gate validation.
- ❌ Machine Learning predicting trên OBI — đã có filter XGBoost trong pipeline.

---

## 13. Next Steps (Implementation Plan)

1. **Viết tests** (pytest) trước hoặc song song với implementation, đặc biệt test công thức 40-60%.
2. **Tạo `modules/order_book/`** với các file logic (`market_data_fetcher`, `_calculator`, `_gate`).
3. **Tích hợp `OrderExecutor`** — thêm `order_book_imbalance_config` param và config vào `settings.yaml`.
4. **Test end-to-end** với dry_run mode để quan sát behavior qua log.
5. **Tuning threshold** dựa trên report log sau vài ngày chạy thực tế.
