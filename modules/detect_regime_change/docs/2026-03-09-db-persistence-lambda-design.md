# Thiết Kế: DB Persistence & Lambda Offloading cho Adaptive Close Time

> **Ngày tạo:** 2026-03-09
> **Trạng thái:** Design Approved — Chờ Implementation
> **Tác giả:** Multi-Agent Brainstorming Session
> **Phụ thuộc:** [2026-03-09-adaptive-close-time-design.md](./2026-03-09-adaptive-close-time-design.md) (Phase 1 + 2 đã hoàn thành)

---

## Mục Lục

1. [Bối Cảnh](#1-bối-cảnh)
2. [Feature A — DB Persistence for Restart Recovery](#2-feature-a--db-persistence-for-restart-recovery)
3. [Feature B — AWS Lambda Offloading](#3-feature-b--aws-lambda-offloading)
4. [Data Flow Chi Tiết](#4-data-flow-chi-tiết)
5. [Cấu Trúc Thư Mục](#5-cấu-trúc-thư-mục)
6. [Cấu Hình Settings](#6-cấu-hình-settings)
7. [Decision Log](#7-decision-log)
8. [Rủi Ro và Mitigations](#8-rủi-ro-và-mitigations)

---

## 1. Bối Cảnh

### 1.1 Hiện trạng (Phase 1 + 2 đã hoàn thành)

- `auto_close_deadline_utc` **đã được lưu** vào DynamoDB order item khi order mở (task 1.9 ✅)
- `auto_close_timer_job.py` đọc field này khi restart — deadline không bị mất ✅
- Rust PELT extension đang chạy **local** (Phase 2 ✅), tốn CPU tại thời điểm mở order
- DynamoDB chỉ lưu **giá trị tuyệt đối** (ISO timestamp), không lưu metadata về cách tính toán
- Database backend: **DynamoDB** (table `AutoTrade`, single-table design), không còn SQLite

### 1.2 Vấn đề cần giải quyết

**Feature A:**

- Sau restart, GUI không biết deadline này là "adaptive" hay "static fallback"
- Không có audit trail: tại sao PELT cho ra X giờ, HMM cho ra Y giờ?
- Panel Scheduled Exits không thể hiện thông tin nguồn gốc deadline

**Feature B:**

- PELT + HMM analysis = CPU-intensive, chạy đồng bộ tại order open time
- Muốn có khả năng offload sang AWS Lambda để giải phóng local CPU
- Khi scan nhiều symbol song song → local machine có thể bị bottleneck

### 1.3 Nguyên tắc thiết kế

- **Feature A là bắt buộc** — cần để hoàn thiện Phase 3 (GUI)
- **Feature B là optional** — opt-in qua config, fallback về local nếu không dùng
- **Không phá vỡ gì** — `auto_close_timer_job.py` không thay đổi
- **YAGNI** — không lưu JSON blob, chỉ flat attributes; không thêm Lambda nếu local đủ

---

## 2. Feature A — DB Persistence for Restart Recovery

### 2.1 Thiết kế: 4 Attributes mới trong DynamoDB Order Item

Field `auto_close_deadline_utc` đã có và hoạt động đúng. Thêm 4 attributes metadata.

**DynamoDB là schema-less** — không cần migration script, không có khái niệm `ALTER TABLE`.
Chỉ cần include các fields mới trong `data` dict khi gọi `create_order()` hoặc `ctx.orders.update_order()`,
DynamoDB sẽ tự lưu chúng vào item. Order items cũ đơn giản là thiếu các attributes này — không lỗi.

| Attribute | DynamoDB Type | Description |
| --- | --- | --- |
| `auto_close_deadline_source` | S (String) | `"adaptive"` \| `"static"` \| `"adaptive_fallback"` |
| `adaptive_close_duration_hours` | N (Number) | Giờ đã tính sau clamp (float) |
| `adaptive_close_pelt_hours` | N (Number) | PELT trimmed avg hours, omitted if None |
| `adaptive_close_hmm_hours` | N (Number) | HMM predicted next-state hours, omitted if None |

**Lưu ý quan trọng:**

- `auto_close_timer_job.py` **KHÔNG thay đổi** — chỉ cần `auto_close_deadline_utc`
- 4 attributes mới chỉ phục vụ **GUI display** và **audit**
- Restart recovery đã hoạt động qua `auto_close_deadline_utc` — 4 attributes này là bonus
- DynamoDB table: `AutoTrade`, PK: `ORDER#{order_id}`, SK: `METADATA`
- Các order items cũ (không có 4 attributes này) sẽ trả về `None`/key missing khi đọc — GUI cần xử lý gracefully

### 2.2 Logic gán `auto_close_deadline_source`

```text
Scenario                                    → auto_close_deadline_source
────────────────────────────────────────────────────────────────────────
adaptive.enabled=True, analysis thành công → "adaptive"
adaptive.enabled=True, analysis thất bại   → "adaptive_fallback"
  → dùng max_duration_hours static fallback
adaptive.enabled=False                     → "static"
```

### 2.3 Thay đổi trong `AdaptiveCloseCalculator`

Phương thức `compute_adaptive_deadline()` hiện trả về `Optional[datetime]`.
Thêm dataclass mới và method mới — không sửa method cũ để tránh breaking change:

```python
@dataclass
class AdaptiveCloseResult:
    deadline_utc: Optional[datetime]
    source: str                          # "adaptive" | "static" | "adaptive_fallback"
    duration_hours: Optional[float]      # Giá trị đã clamp
    pelt_hours: Optional[float]          # Raw PELT avg
    hmm_hours: Optional[float]           # Raw HMM prediction


def compute_adaptive_deadline_with_meta(
    self, symbol: str, opened_at: datetime, ohlcv_df=None
) -> AdaptiveCloseResult:
    ...
```

### 2.4 Integration vào Order Flow

```python
# Trong execution flow (order_manager.py hoặc order_executor.py):

result = calculator.compute_adaptive_deadline_with_meta(
    symbol=order["symbol"],
    opened_at=order_opened_at,
)

# Merge vào order_data dict trước khi gọi ctx.orders.create_order(order_data)
# DynamoDB sẽ tự lưu các attributes này vào item — không cần migration
order_data.update({
    "auto_close_deadline_utc": result.deadline_utc.isoformat() if result.deadline_utc else None,
    "auto_close_deadline_source": result.source,
    "adaptive_close_duration_hours": result.duration_hours,
    "adaptive_close_pelt_hours": result.pelt_hours,
    "adaptive_close_hmm_hours": result.hmm_hours,
})

ctx.orders.create_order(order_data)
```

### 2.5 GUI Display (Phase 3)

Với 4 attributes này, panel Scheduled Exits có thể hiển thị:

```text
Symbol    Deadline              Source             Duration    Detail
────────  ────────────────────  ─────────────────  ──────────  ──────────────────────
BTC/USDT  2026-03-09 08:44 UTC  adaptive           3h 44m      PELT:3.2h, HMM:4.1h
ETH/USDT  2026-03-09 07:30 UTC  static             4h 00m      —
ADA/USDT  2026-03-09 06:15 UTC  adaptive_fallback  4h 00m      Analysis failed
```

---

## 3. Feature B — AWS Lambda Offloading

> **Status:** Optional Enhancement — opt-in qua config `use_lambda: false` (mặc định)

### 3.1 Rationale

| Khi nào cần Lambda | Khi nào không cần |
| --- | --- |
| Scan nhiều symbol song song (pre-computation) | Chỉ 1 order/lần, local Rust đủ nhanh |
| Local machine CPU yếu | Local machine có nhiều core |
| Muốn tách phân tích ra khỏi trading process | Chấp nhận local latency |

**Quyết định:** Thiết kế đầy đủ, nhưng `use_lambda: false` mặc định. Bật khi có nhu cầu thực sự.

### 3.2 Architecture

```text
Local Python                              regime-analysis-lambda (Rust)
─────────────────────────────────         ──────────────────────────────────
AdaptiveCloseCalculator
  │
  ├─ 1. _fetch_ohlcv() [GIỮ LOCAL]
  │      ccxt.binance → DataFrame
  │
  ├─ 2. use_lambda=True?
  │     │
  │     ├─ YES → RegimeLambdaClient        Nhận: {symbol, ohlcv_json, config}
  │     │        .invoke(ohlcv_df, ...)    ├─ PELT (Rust, parallel Rayon)
  │     │        timeout=3s           ───► ├─ HMM estimation
  │     │        Nhận kết quả         ◄─── └─ _combine_results()
  │     │                                  Trả: RegimeDurationResult JSON
  │     │
  │     └─ NO → RegimeDurationAnalyzer local (hiện tại)
  │
  ├─ 3. Fallback chain (nếu Lambda fail):
  │     Lambda timeout/error
  │       → Local Rust PELT extension
  │       → Python ruptures
  │       → Static 4h fallback
  │
  └─ 4. Set deadline + metadata → DynamoDB order item
```

### 3.3 Module mới: `regime_lambda_client.py`

```text
modules/detect_regime_change/
└── regime_lambda_client.py     # NEW: HTTP client gọi Lambda
```

**Interface:**

```python
class RegimeLambdaClient:
    def __init__(self, endpoint: str, timeout_seconds: float = 3.0):
        ...

    def invoke(
        self,
        ohlcv_df: pd.DataFrame,
        symbol: str,
        config: Dict[str, Any],
    ) -> Optional[RegimeDurationResult]:
        """
        Gửi OHLCV data lên Lambda, nhận RegimeDurationResult.
        Trả None nếu timeout hoặc lỗi (caller tự fallback).
        """
        ...

    def _serialize_ohlcv(self, df: pd.DataFrame) -> dict:
        """Convert DataFrame → JSON-serializable dict."""
        ...

    def _deserialize_result(self, data: dict) -> RegimeDurationResult:
        """Parse Lambda response → RegimeDurationResult."""
        ...
```

### 3.4 Lambda Handler Structure (Rust)

Reuse pattern `adaptive_trend_LTS_serverless`:

```text
modules/detect_regime_change/regime_lambda/
├── Cargo.toml                  # Lambda dependencies + pyo3 disabled
├── src/
│   ├── main.rs                 # Lambda entry point (cargo-lambda)
│   ├── handler.rs              # Gọi PELT + HMM Rust logic
│   └── models.rs               # Request/Response JSON structs
└── template.yaml               # SAM deployment template
```

**Request/Response:**

```json
{
  "symbol": "BTC/USDT",
  "timeframe": "15m",
  "ohlcv": {
    "timestamps": [1704067200000],
    "open": [42000.0],
    "high": [42200.0],
    "low": [41900.0],
    "close": [42100.0],
    "volume": [100.0]
  },
  "config": {
    "lookback_days": 60,
    "pelt_model": "rbf",
    "pelt_min_segment": 10,
    "hmm_train_ratio": 0.8,
    "hmm_high_confidence_threshold": 0.7
  }
}
```

```json
{
  "symbol": "BTC/USDT",
  "recommended_duration_hours": 3.74,
  "pelt_avg_duration_hours": 3.2,
  "pelt_median_duration_hours": 2.8,
  "hmm_next_state_duration_hours": 4.1,
  "hmm_state": 1,
  "hmm_state_probability": 0.82,
  "data_points_analyzed": 5760,
  "computation_time_ms": 120,
  "error": null
}
```

### 3.5 Payload Size Estimation

```text
60 ngày × 15m candles = 5760 candles
6 fields × float64 = 48 bytes/candle
5760 × 48 = ~276KB raw
JSON overhead ≈ ×3 → ~800KB

Lambda sync payload limit: 6MB → OK
Nên gửi compressed (gzip) nếu > 1MB
```

### 3.6 Fallback Chain

```python
# Trong AdaptiveCloseCalculator.compute_adaptive_deadline_with_meta():

if cfg["use_lambda"] and cfg["lambda_endpoint"]:
    # Attempt 1: Lambda
    lambda_result = regime_lambda_client.invoke(ohlcv_df, symbol, config)
    if lambda_result is not None and lambda_result.is_valid:
        return AdaptiveCloseResult(from_result=lambda_result, source="adaptive")

# Attempt 2: Local Rust/Python (hiện tại)
local_result = RegimeDurationAnalyzer(...).analyze(ohlcv_df, symbol, timeframe)
if local_result.is_valid:
    return AdaptiveCloseResult(from_result=local_result, source="adaptive")

# Attempt 3: Static fallback
return AdaptiveCloseResult(
    deadline_utc=opened_at + timedelta(hours=fallback_hours),
    source="adaptive_fallback",
    duration_hours=fallback_hours,
    pelt_hours=None,
    hmm_hours=None,
)
```

---

## 4. Data Flow Chi Tiết

```text
╔══════════════════════════════════════════════════════════════════╗
║               ORDER PLACEMENT (với Feature A+B)                  ║
║                                                                  ║
║  1. Scanner/User places order on BTC/USDT                        ║
║     │                                                            ║
║  2. AdaptiveCloseCalculator                                      ║
║     .compute_adaptive_deadline_with_meta()                       ║
║     │                                                            ║
║  3. _fetch_ohlcv("BTC/USDT", "15m", 60d) [local, ccxt]         ║
║     │                                                            ║
║  4. use_lambda=True? ─── YES ──► RegimeLambdaClient.invoke()    ║
║     │                              timeout=3s                   ║
║     │                           ◄── RegimeDurationResult JSON   ║
║     │                                                            ║
║     └── NO (hoặc Lambda fail) ──► RegimeDurationAnalyzer local  ║
║                                    PELT + HMM                   ║
║                                                                  ║
║  5. AdaptiveCloseResult:                                         ║
║     ├─ deadline_utc  = opened_at + 3.74h                        ║
║     ├─ source        = "adaptive"                                ║
║     ├─ duration_hours= 3.74                                      ║
║     ├─ pelt_hours    = 3.2                                       ║
║     └─ hmm_hours     = 4.1                                       ║
║                                                                  ║
║  6. ctx.orders.create_order(order_data) → DynamoDB PutItem      ║
║     Item: pk=ORDER#<id>, sk=METADATA                            ║
║     + auto_close_deadline_utc: "2026-03-09T08:44:00Z"           ║
║     + auto_close_deadline_source: "adaptive"                    ║
║     + adaptive_close_duration_hours: 3.74                       ║
║     + adaptive_close_pelt_hours: 3.2                            ║
║     + adaptive_close_hmm_hours: 4.1                             ║
║     (schema-less: không cần migration)                          ║
║                                                                  ║
║  7. Program restart → auto_close_timer_job đọc                  ║
║     auto_close_deadline_utc → vẫn hoạt động đúng ✅             ║
║     GUI đọc 4 metadata attributes → hiển thị đầy đủ ✅          ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 5. Cấu Trúc Thư Mục

### Feature A (bắt buộc)

```text
modules/
├── detect_regime_change/
│   └── models.py                          # MODIFIED: thêm AdaptiveCloseResult dataclass
│
└── auto_trade/
    └── execution/
        └── adaptive_close_calculator.py   # MODIFIED:
                                           # - thêm AdaptiveCloseResult dataclass
                                           # - thêm compute_adaptive_deadline_with_meta()
                                           # - cập nhật order flow gọi method mới
```

**Không cần migration file** — DynamoDB schema-less, chỉ thêm attributes vào `order_data` dict.
Các order items cũ đơn giản là không có 4 attributes mới, GUI dùng `.get()` với default.

### Feature B (optional)

```text
modules/
├── detect_regime_change/
│   ├── regime_lambda_client.py    # NEW: HTTP client
│   └── regime_lambda/             # NEW: Lambda handler (Rust)
│       ├── Cargo.toml
│       ├── template.yaml          # SAM
│       └── src/
│           ├── main.rs
│           ├── handler.rs
│           └── models.rs
│
└── auto_trade/
    └── settings.yaml              # MODIFIED: thêm use_lambda, lambda_endpoint
```

---

## 6. Cấu Hình Settings

```yaml
auto_close:
  enabled: true
  max_duration_hours: 4.0          # Fallback tĩnh — giữ nguyên

  adaptive:
    enabled: false                 # Phase 1: tắt mặc định — giữ nguyên
    min_duration_hours: 1.0
    max_duration_hours: 12.0
    lookback_days: 60
    timeframe: '15m'

    # NEW: Lambda Offloading (Feature B)
    use_lambda: false              # Mặc định tắt, opt-in
    lambda_endpoint: ""            # Lambda Function URL
    lambda_timeout_seconds: 3.0    # Timeout trước khi fallback local
```

---

## 7. Decision Log

| # | Quyết định | Phương án đã xét | Lý do chọn |
| --- | --- | --- | --- |
| D1 | **Flat attributes thay JSON blob** | JSON blob `analysis_meta` vs. flat attributes | Không có schema drift, dễ đọc trong DynamoDB console, dễ display trong GUI. JSON là YAGNI giai đoạn này. |
| D2 | **Timer job không thay đổi** | Thêm logic đọc source vào timer job | Timer job chỉ cần `auto_close_deadline_utc`. Metadata chỉ cho GUI/audit. Zero risk. |
| D3 | **Lambda call non-blocking, timeout=3s** | Blocking sync call | Order open không nên bị delay bởi network/Lambda. Fallback đảm bảo order luôn được mở. |
| D4 | **Feature B optional, `use_lambda: false` mặc định** | Bắt buộc cùng Feature A | Rust local đủ nhanh cho 1 order/lần. Lambda chỉ cần khi scan nhiều symbol hoặc local yếu. |
| D5 | **Local fetch OHLCV, gửi data lên Lambda** | Lambda tự fetch từ Binance | Tránh quản lý API key trên cloud. OHLCV là public data, safe để gửi. |
| D6 | **`AdaptiveCloseResult` dataclass mới, method mới** | Mở rộng tuple return, sửa method cũ | Type-safe, extensible. Method cũ `compute_adaptive_deadline()` giữ nguyên — không breaking change. |
| D7 | **Fallback chain 3 tầng cho Feature B** | 2 tầng hoặc chỉ Lambda | Defense in depth: Lambda → local Rust/Python → static. Order không bao giờ "treo". |
| D8 | **DynamoDB schema-less, không cần migration** | SQL migration script | DynamoDB không có fixed schema. Thêm attributes mới vào `order_data` dict là đủ. Items cũ không bị ảnh hưởng. |

---

## 8. Rủi Ro và Mitigations

| Rủi ro | Khả năng | Impact | Mitigation |
| --- | --- | --- | --- |
| Lambda cold start block order | Thấp (timeout=3s) | Trung bình | Non-blocking + fallback |
| GUI crash khi đọc missing attribute (items cũ) | Trung bình | Thấp | Dùng `item.get("auto_close_deadline_source", "unknown")` |
| Lambda payload > 6MB (nhiều candles) | Thấp | Thấp | Compress + giới hạn `limit=1000` candles |
| DynamoDB attribute name collision với fields tương lai | Rất thấp | Thấp | Prefix `adaptive_close_*` đã đủ distinct |

---

*Document generated from Multi-Agent Brainstorming session — 2026-03-09*
*Reviewed by: Primary Designer, Skeptic, Constraint Guardian, User Advocate, Integrator/Arbiter*
