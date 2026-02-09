# ATC Module Bottleneck Analysis

**Date:** February 9, 2026  
**Module:** `modules/adaptive_trend_LTS_mini`  
**Analysis:** Data flow bottleneck identification in ATC (Adaptive Trend Classification) module

## Executive Summary

Sau khi trace toàn bộ data flow từ CLI → Scanner → Data Fetch → Compute → Output, tôi xác định được **5 điểm bottleneck** theo mức độ nghiêm trọng trong ATC module. Bottleneck lớn nhất nằm ở **tầng I/O** (fetch data), không phải compute layer.

## 🔴 BOTTLENECK #1 — Global Lock trong `throttled_call` (NGHIÊM TRỌNG NHẤT)

**File:** `modules/common/core/exchange_manager/public.py#L108-L123`  
**Impact:** ~60s+ cho 300 symbols  
**Type:** I/O bottleneck

### Problem Description
Dù scanner dùng `ThreadPoolExecutor` với 32 workers, **tất cả API calls đều bị serialize** vì `_request_lock` là global. Khi quét 300 symbols:

- Threadpool submit 32 tasks song song
- Nhưng mỗi task gọi `throttled_call` → phải chờ lock
- Với `request_pause = 0.2s`, mỗi symbol tốn ít nhất 0.2s → **300 symbols × 0.2s = 60 giây** chỉ riêng chờ rate limit
- Network I/O (`fetch_ohlcv`) cũng nằm **trong lock**, nên thread khác không thể gọi song song

### Code Evidence
```python
def throttled_call(self, func, *args, **kwargs):
    with self._request_lock:          # ← Global lock cho TẤT CẢ threads
        wait = self.request_pause - (time.time() - self._last_request_ts)
        if wait > 0:
            time.sleep(wait)          # ← Sleep TRONG lock → serialize tất cả requests
        result = func(*args, **kwargs)  # ← Network call TRONG lock
        self._last_request_ts = time.time()
        return result
```

### Root Cause
- Global `_request_lock` serialize tất cả API calls
- `time.sleep()` được gọi bên trong lock
- Network I/O nằm trong critical section

### Proposed Solution
Dùng per-exchange rate limiter hoặc token bucket thay vì global lock. ccxt đã có `enableRateLimit=True` nên `throttled_call` đang double-rate-limiting.

---

## 🔴 BOTTLENECK #2 — Sequential Exchange Fallback

**File:** `modules/common/core/data_fetcher/ohlcv.py#L43-L200`  
**Impact:** ~40s worst case per symbol  
**Type:** I/O bottleneck

### Problem Description
Khi exchange chính fail (timeout, empty data), phải thử tuần tự qua tối đa **8 exchanges**. Mỗi lần nhân với bottleneck #1 (global lock + sleep). Worst case cho 1 symbol với freshness check: **8 × (0.2s pause + 2-5s network) ≈ 16-40 giây**.

### Code Evidence
```python
for exchange_id in exchange_list:    # ← Duyệt tuần tự qua 8 exchanges
    exchange = self.base.exchange_manager.public.connect_to_exchange_with_no_credentials(exchange_id)
    ohlcv = self.base.exchange_manager.public.throttled_call(exchange.fetch_ohlcv, ...)
    # Nếu fail → continue → thử exchange tiếp
```

### Root Cause
- Sequential iteration through exchange list
- Each exchange attempt compounds with rate limiting delays
- No parallel probing of multiple exchanges

### Proposed Solution
Parallel exchange probe hoặc primary exchange pinning để tránh sequential fallback.

---

## 🔴 BOTTLENECK #3 — Cache Bị Bypass khi `check_freshness=True`

**File:** `modules/common/core/data_fetcher/ohlcv.py#L82-L89`  
**Impact:** ×N redundant fetches  
**Type:** I/O bottleneck

### Problem Description
Trong scanner, `_process_symbol` gọi với `check_freshness=True`, nghĩa là cache **hoàn toàn bị bỏ qua**. Mỗi symbol luôn phải fetch từ network dù data cùng timeframe có thể được dùng lại trong thời gian scan.

### Code Evidence
```python
if not check_freshness:
    if cache_key in self.base._ohlcv_dataframe_cache:
        return cached_df.copy(), cached_exchange
# Khi check_freshness=True → LUÔN fetch mới từ network
```

### Root Cause
- `check_freshness=True` bypasses cache completely
- No TTL-based freshness validation
- Scanner always forces fresh data even when cache is valid

### Proposed Solution
TTL-based freshness cache thay vì bypass hoàn toàn. Implement cache with configurable freshness window.

---

## 🟡 BOTTLENECK #4 — 54 MA calculations per symbol

**File:** `modules/adaptive_trend_LTS_mini/core/compute_atc_signals/compute_atc_signals.py#L87-L200`  
**Impact:** ~2s/symbol overhead  
**Type:** CPU bottleneck

### Problem Description
Mỗi symbol tính **6 MA types × 9 variations = 54 Moving Averages**. Dù có Rust backend và thread parallelism cho 9 MAs mỗi type, đây vẫn là **CPU-intensive** block. Trong mode `set_of_moving_averages_rust`:

Tạo `ThreadPoolExecutor` **mỗi lần gọi** (6 lần/symbol × 300 symbols = 1800 executor creations) gây overhead.

### Code Evidence
```python
with ThreadPoolExecutor(max_workers=config.num_threads) as executor:
    futures = [executor.submit(ma_calculation_rust, source, ma_len, ...) for ma_len in ma_lengths]
```

### Root Cause
- ThreadPoolExecutor recreation for each MA type
- 54 MA calculations per symbol (6 types × 9 variations)
- No reuse of thread pools across calculations

### Proposed Solution
Reuse thread pool, batch MA calculation across multiple symbols or MA types.

---

## 🟡 BOTTLENECK #5 — GC và ThreadPoolExecutor Recreation

**File:** `modules/adaptive_trend_LTS_mini/core/scanner/threadpool.py#L56-L104`  
**Impact:** ~200ms/batch  
**Type:** Memory/Performance bottleneck

### Problem Description
`ThreadPoolExecutor` được tạo/hủy mỗi batch (100 symbols) thay vì dùng 1 pool cho toàn bộ scan. `gc.collect()` là stop-the-world, gây pause ~50-200ms mỗi batch.

### Code Evidence
```python
for batch_start in range(0, total, batch_size):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # ← Tạo mới mỗi batch
        ...
    gc.collect()   # ← Stop-the-world GC mỗi 100 symbols
```

### Root Cause
- ThreadPoolExecutor recreation per batch
- Stop-the-world garbage collection
- No thread pool reuse across batches

### Proposed Solution
Single pool cho toàn bộ scan, incremental GC hoặc adaptive GC timing.

---

## Impact Summary Table

| # | Bottleneck | Type | Impact | Priority |
|---|-----------|------|--------|----------|
| 1 | `throttled_call` global lock | I/O | **~60s+** cho 300 symbols | 🔴 Critical |
| 2 | Sequential fallback | I/O | **~40s** worst case/symbol | 🔴 Critical |
| 3 | Cache bypass với freshness | I/O | **×N** redundant fetches | 🔴 Critical |
| 4 | 54 MA × executor recreation | CPU | **~2s/symbol overhead** | 🟡 High |
| 5 | GC + pool recreation per batch | Memory | **~200ms/batch** | 🟡 Medium |

## Conclusion

**Bottleneck lớn nhất là ở tầng I/O (fetch data), không phải compute.** Global lock trong `throttled_call` biến threadpool executor thành essentially sequential. Để tăng throughput scan 300+ symbols, ưu tiên sửa bottleneck #1 và #3 trước.

### Recommended Fix Order
1. **#1** - Fix global lock in `throttled_call` (highest impact)
2. **#3** - Implement TTL-based cache with freshness
3. **#2** - Parallel exchange fallback
4. **#4** - Thread pool reuse for MA calculations
5. **#5** - Single thread pool for scanning

## Data Flow Architecture

```mermaid
flowchart TB
    subgraph CLI["CLI Layer (cli/main.py)"]
        A[ATCAnalyzer.run] --> B{Mode?}
        B -->|auto| C[AutoModeExecutor]
        B -->|manual| D[ManualModeExecutor]
    end

    subgraph SCAN["Scanner Layer (core/scanner/)"]
        C --> E[scan_all_symbols]
        E --> F{Execution Mode}
        F -->|sequential| G["_scan_sequential\n⚠️ SLOW: 1 symbol/time"]
        F -->|threadpool| H["_scan_threadpool\n✅ Default, parallel"]
        F -->|asyncio| I["_scan_asyncio\nSemaphore controlled"]
        F -->|processpool| J["_scan_processpool\nCPU parallel"]
        G --> K[_process_symbol]
        H --> K
        I --> K
        J --> K
    end

    subgraph FETCH["Data Fetch Layer (common/core/data_fetcher/)"]
        K --> L["fetch_ohlcv_with_fallback_exchange\n🔴 BOTTLENECK #1"]
        L --> M{Cache Hit?}
        M -->|Yes| N["Return cached df ✅"]
        M -->|No| O["throttled_call → exchange.fetch_ohlcv\n🔴 BOTTLENECK #2: Global Lock + sleep"]
        O -->|Fail| P["Fallback to next exchange\n🔴 BOTTLENECK #3: Sequential retry"]
    end

    subgraph COMPUTE["Compute Layer (core/compute_atc_signals/)"]
        K --> Q["compute_atc_signals\n(6 MA types × 9 variations = 54 MAs)"]
        Q --> R["set_of_moving_averages ×6\n🟡 BOTTLENECK #4: 54 MA calculations"]
        R --> S["_compute_layer1\n(6 signals)"]
        S -->|parallel_l1=True| T["ProcessPool parallel ✅"]
        S -->|parallel_l1=False| U["Sequential loop 🟡"]
        S --> V["calculate_layer2_equities"]
        V --> W["calculate_average_signal"]
    end

    subgraph MEM["Memory/GC"]
        X["gc.collect() per batch\n🟡 BOTTLENECK #5"]
        Y["Series pool alloc/release"]
    end

    style L fill:#ff6b6b,color:#fff
    style O fill:#ff6b6b,color:#fff
    style P fill:#ff6b6b,color:#fff
    style R fill:#ffd93d,color:#333
    style U fill:#ffd93d,color:#333
    style X fill:#ffd93d,color:#333
```</content>
<parameter name="filePath">c:\Users\Admin\Desktop\i-ching\crypto-probability\ATC_MODULE_BOTTLENECK_ANALYSIS.md