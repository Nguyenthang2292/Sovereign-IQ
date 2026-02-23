# Code Review — `xgboost_LTS_serverless`

> **Ngày review:** 2026-02-22  
> **Reviewer:** Antigravity AI  
> **Module:** `modules/xgboost_LTS_serverless`  
> **Ngôn ngữ:** Rust (core library) + Rust (Lambda handler) + Python (scripts)  
> **Phiên bản:** 0.1.0

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Điểm mạnh](#2-điểm-mạnh)
3. [Vấn đề nghiêm trọng (Critical)](#3-vấn-đề-nghiêm-trọng-critical)
4. [Vấn đề quan trọng (Major)](#4-vấn-đề-quan-trọng-major)
5. [Vấn đề nên cải thiện (Minor)](#5-vấn-đề-nên-cải-thiện-minor)
6. [Phân tích theo file](#6-phân-tích-theo-file)
7. [Đề xuất hành động](#7-đề-xuất-hành-động)
8. [Tóm tắt điểm số](#8-tóm-tắt-điểm-số)

---

## 1. Tổng quan kiến trúc

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   API Gateway   │───▶│  Lambda Function │───▶│      S3         │
│   (REST API)    │     │  (XGBoost Rust)  │     │  (Model Store)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │       SQS        │
                        │  (Predictions)   │
                        └──────────────────┘
```

**Cấu trúc workspace:**

- `xgboost_serverless` (core library) — Feature engineering, inference, model management
- `xgboost_lambda` (binary) — AWS Lambda handler với API Gateway + S3 + SQS
- `scripts/` — Build, deploy, validation, load test
- `tests/` — Integration tests cho features và inference

**Tổng dòng code Rust:** ~2,900 LOC  
**Tổng dòng code Python (scripts):** ~1,200 LOC  

---

## 2. Điểm mạnh

### ✅ Kiến trúc rõ ràng & phân tách trách nhiệm

- Workspace Cargo tách biệt `core library` và `lambda handler` — cho phép test core library độc lập.
- Feature engineering modules được tổ chức theo domain (price_derived, indicators, moving_averages, candlestick, advanced, lag_features).

### ✅ Cross-compilation awareness

- Sử dụng `cfg(not(windows))` và optional `xgboost` feature flag để xử lý vấn đề XGBoost native binding trên Windows → cho phép dev/test trên Windows mà không cần `libxgboost`.
- Stub inference trả về `NEUTRAL` khi xgboost feature không enabled — đảm bảo tests chạy được cross-platform.

### ✅ Performance-oriented design

- Release profile tối ưu: `opt-level = 3`, `lto = "thin"`, `strip = true`, `codegen-units = 1`.
- Lambda handler sử dụng `JoinSet` cho parallel processing batch requests.
- `spawn_blocking` cho feature calculation — tránh block Tokio async runtime.
- `OnceLock` cho `ModelManager` singleton — tối ưu cold start.
- Deploy script bật AVX2 SIMD (`-C target-cpu=haswell -C target-feature=+avx2`).

### ✅ Error handling có cấu trúc

- `XGBoostError` enum với `thiserror` — 8 error variants rõ ràng.
- Validation logic tại entry point Lambda handler.

### ✅ Deploy automation hoàn chỉnh

- `deploy_lambda.py` — 507 LOC script xử lý toàn bộ lifecycle: dependency check, IAM role, S3 bucket, SQS queue, Zig linker detection, build, deploy.
- Feature parity validation (`validate_feature_parity.py`) giữa Rust và Python pipelines.

### ✅ Test infrastructure

- Property-based testing với `proptest` (dev-dependency).
- Inference tests kiểm tra output format, feature count, probability sum.
- Feature tests kiểm tra 92-feature vector consistency.

---

## 3. Vấn đề nghiêm trọng (Critical)

### 🔴 C-01: `RwLock::unwrap()` có thể panic trong production

**File:** `src/model_manager.rs:40, 54, 70`

```rust
let mut cache_write = self.cache.write().unwrap();  // Line 40
let cache_read = self.cache.read().unwrap();         // Line 54
let mut cache_write = self.cache.write().unwrap();   // Line 70
```

**Vấn đề:** Nếu một thread panic trong khi giữ lock, lock sẽ bị "poisoned" và tất cả các lần gọi `.unwrap()` tiếp theo sẽ panic → Lambda crash không thể recover.

**Đề xuất:**

```rust
let cache_read = self.cache.read()
    .unwrap_or_else(|poisoned| poisoned.into_inner());
```

Hoặc convert sang `XGBoostError`:

```rust
let cache_read = self.cache.read()
    .map_err(|_| XGBoostError::InferenceError("Lock poisoned".to_string()))?;
```

---

### 🔴 C-02: Division by zero trong `returns_1_tail` và `returns_n_last`

**File:** `src/feature_engine.rs:65`, `src/features/price_derived.rs:14`

```rust
// feature_engine.rs:65
result.push((close[i] - close[i - 1]) / close[i - 1]);

// price_derived.rs:14
(close[i] - close[i - n]) / close[i - n]
```

**Vấn đề:** Nếu `close[i-1]` hoặc `close[i-n]` == 0.0 → Rust sẽ trả về `f64::INFINITY` hoặc `f64::NEG_INFINITY` (Rust không panic trên floating-point division by zero, nhưng INF sẽ propagate và gây NaN/corrupt toàn bộ feature vector).

**Đề xuất:** Sử dụng `safe_ratio()` (đã tồn tại trong `feature_engine.rs:6`) hoặc thêm guard:

```rust
if close[i - 1] == 0.0 { f64::NAN } else { (close[i] - close[i - 1]) / close[i - 1] }
```

---

### 🔴 C-03: Bounds check thiếu trong `OHLCVData`

**File:** `src/ohlcv.rs`

**Vấn đề:** `OHLCVData::new()` KHÔNG validate rằng tất cả các vectors có cùng length. Nếu `timestamp`, `open`, `high`, `low`, `close`, `volume` có length khác nhau → out-of-bounds panic trong feature engineering.

**Đề xuất:**

```rust
pub fn new(/* ... */) -> Result<Self, XGBoostError> {
    let n = close.len();
    if timestamp.len() != n || open.len() != n || high.len() != n
        || low.len() != n || volume.len() != n {
        return Err(XGBoostError::ValidationError(
            "All OHLCV vectors must have the same length".to_string(),
        ));
    }
    Ok(Self { timestamp, open, high, low, close, volume })
}
```

---

### 🔴 C-04: `partial_cmp().unwrap()` sẽ panic trên NaN

**File:** `src/xgboost_inference.rs:62`

```rust
let (idx, &confidence) = probabilities
    .iter()
    .enumerate()
    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())  // PANIC nếu NaN!
    .unwrap();
```

**Vấn đề:** Nếu XGBoost model trả về NaN probability → `partial_cmp` returns `None` → `.unwrap()` panic.

**Đề xuất:**

```rust
.max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
```

---

## 4. Vấn đề quan trọng (Major)

### 🟡 M-01: `build.sh` dùng `--arm64` nhưng `deploy_lambda.py` dùng `x86_64`

**File:** `scripts/build.sh:10` vs `scripts/deploy_lambda.py:351`

```bash
# build.sh
cargo lambda build --release --arm64

# deploy_lambda.py
cmd = ["cargo", "lambda", "build", "--release", "--target", "x86_64-unknown-linux-gnu"]
```

**Vấn đề:** Hai deployment paths target architectures khác nhau (ARM64 vs x86_64). Nếu dùng `build.sh` rồi deploy bằng SAM template → binary ARM64 chạy trên Lambda runtime `provided.al2` (mặc định x86_64) → crash.

**Đề xuất:** Thống nhất architecture. Nếu dùng `x86_64` (với AVX2 optimization) → sửa `build.sh` thành `--target x86_64-unknown-linux-gnu`. Nếu muốn ARM (Graviton2) → bỏ AVX2 flags.

---

### 🟡 M-02: `S3Client` và `SqsClient` tạo mới AWS SDK config mỗi lần gọi

**File:** `lambda/src/s3_client.rs:10-11`, `lambda/src/sqs_client.rs:10-11`

```rust
pub async fn new(bucket_name: String) -> Self {
    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = Client::new(&config);
    // ...
}
```

**Vấn đề:** Mỗi lần request có SQS result → tạo mới AWS SDK config (gọi IMDS metadata service). Trên Lambda, điều này thêm ~50-200ms latency mỗi request.

**Đề xuất:** Dùng `OnceLock` giống `MODEL_MANAGER`:

```rust
static AWS_CONFIG: OnceLock<aws_config::SdkConfig> = OnceLock::new();

pub async fn shared_config() -> &'static aws_config::SdkConfig {
    AWS_CONFIG.get_or_init(|| {
        // Block on async init
        futures::executor::block_on(
            aws_config::load_defaults(aws_config::BehaviorVersion::latest())
        )
    })
}
```

---

### 🟡 M-03: `ModelManager.get_or_load` TOCTOU race condition

**File:** `src/model_manager.rs:45-74`

```rust
pub fn get_or_load(&self, symbol: &str, timeframe: &str, version: &str) -> Result<...> {
    // 1. Read lock - check cache
    { let cache_read = self.cache.read().unwrap(); /* ... */ }
    // 2. Check file exists
    // 3. Load model
    // 4. Write lock - insert
}
```

**Vấn đề:** Giữa step 1 (nhả read lock) và step 4 (lấy write lock), thread khác có thể đã insert model cùng key → duplicate work (không crash nhưng waste CPU/memory).

**Đề xuất:** Dùng pattern "double-checked locking":

```rust
pub fn get_or_load(&self, ...) -> Result<...> {
    // Fast path: read lock
    { /* check cache, return if found */ }
    // Slow path: write lock, re-check, then load
    let mut cache_write = self.cache.write()...;
    if let Some(model) = cache_write.get(&cache_key) {
        return Ok(Arc::clone(model));
    }
    // Load and insert under write lock
}
```

---

### 🟡 M-04: Magic number `92` hardcoded ở nhiều nơi

**Files:** `src/feature_engine.rs:125,241`, `src/xgboost_inference.rs:32`, `src/bin/calculate_features.rs:8`, `tests/inference_tests.rs` (nhiều chỗ)

**Vấn đề:** Nếu thêm/bớt features → phải sửa ở >= 5 chỗ. Dễ sai sót.

**Đề xuất:** Tạo constant:

```rust
// lib.rs
pub const EXPECTED_FEATURE_COUNT: usize = 92;
```

Và dùng ở tất cả các nơi, hoặc tốt hơn, derive từ `FEATURE_NAMES.len()`.

---

### 🟡 M-05: `handler.rs` — `request` bị moved nhưng vẫn được dùng sau đó

**File:** `lambda/src/handler.rs:70, 93, 168`

```rust
let request = event.payload;           // Line 70
// ...
for item in request.requests {          // Line 93 — moves request.requests
// ...
if let Some(options) = &request.options { // Line 168 — ERROR: request partially moved!
```

**Vấn đề:** `for item in request.requests` sẽ **consume** `request.requests`. Sau đó `&request.options` vẫn hợp lệ vì Rust cho phép truy cập fields khác sau partial move. Tuy nhiên, đây là **design smell** — nếu ai đó thêm code truy cập `request.requests` sau loop → compiler error khó hiểu.

**Đề xuất:** Clone hoặc destructure rõ ràng:

```rust
let XGBoostRequest { requests, options, version, mode } = event.payload;
```

---

### 🟡 M-06: Missing `validate_request` → Error trait compatibility

**File:** `lambda/src/handler.rs:217`

```rust
fn validate_request(request: &XGBoostRequest) -> Result<(), XGBoostError> {
```

**Vấn đề:** Function trả về `XGBoostError` nhưng handler trả về `lambda_runtime::Error`. Tại line 71, `validate_request(&request)?` hoạt động vì `XGBoostError` implement `std::error::Error` (via `thiserror`) → auto-convert. Tuy nhiên, `XGBoostError` không implement `From<XGBoostError> for lambda_runtime::Error` nên chỉ hoạt động nhờ `?` operator boxing. Đây là OK nhưng nên explicit.

---

### 🟡 M-07: `stochastic_rsi` filtering logic nghi vấn

**File:** `src/features/indicators.rs:299`

```rust
if !v.is_nan() && v != 50.0 && v != 0.0 {
    // basic skip of rsi padded values if any
```

**Vấn đề:** Skip `v == 50.0` (giá trị padding) và `v == 0.0` là heuristic — RSI thực tế CÓ THỂ bằng 50.0 hoặc 0.0. Điều này sẽ sai khi RSI tự nhiên đạt đúng 50.0 → kết quả Stochastic RSI bị lệch.

**Đề xuất:** Dùng `Option<f64>` hoặc sentinel value riêng (ví dụ: track index nào là "padded").

---

## 5. Vấn đề nên cải thiện (Minor)

### 🟢 m-01: `candlestick.rs` — 1087 LOC, nhiều code lặp

**File:** `src/features/candlestick.rs`

**Vấn đề:** File dài nhất trong project (1087 dòng). 48 pattern detection functions đều có chữ ký giống nhau `(open, high, low, close, i) -> bool`. `to_feature_vec()` có 48 `if/else` blocks giống hệt nhau.

**Đề xuất:**

1. Dùng macro hoặc trait:

```rust
macro_rules! pattern_flag {
    ($val:expr) => { if $val { 1.0 } else { 0.0 } };
}
```

1. Dùng array of function pointers:

```rust
type PatternFn = fn(&[f64], &[f64], &[f64], &[f64], usize) -> bool;
const DETECTORS: &[PatternFn] = &[detect_doji_at, detect_hammer_at, ...];
```

1. Tách `to_feature_vec()` thành:

```rust
pub fn to_feature_vec(&self) -> Vec<f64> {
    self.as_array().iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()
}
```

---

### 🟢 m-02: `FeatureCache` không được sử dụng

**File:** `src/feature_engine.rs:77-98, 100-102`

**Vấn đề:** `FeatureEngine` có field `pub cache: FeatureCache` nhưng `calculate_all()` KHÔNG sử dụng cache. `FeatureCache::get_or_insert()` không bao giờ được gọi trong production code.

**Đề xuất:** Hoặc integrate cache vào `calculate_all()` (cache gains/losses, RSI tail, etc.), hoặc lược bỏ dead code.

---

### 🟢 m-03: `sma()` padding bằng `0.0` thay vì `f64::NAN`

**File:** `src/features/moving_averages.rs:8`

```rust
sma_values.extend(vec![0.0; period - 1]);  // SMA padding = 0.0
```

**Tương tự:** `wma()` (line 77), `roc()` (line 7), `rolling_std()` (line 31), `rolling_skewness()` (line 59)

**Vấn đề:** Padding bằng `0.0` thay vì `NaN` có thể gây misleading — 0.0 là giá trị hợp lệ. Trong context XGBoost, NaN thường được handle tốt hơn (XGBoost tự xử lý missing values).

**Lưu ý:** Việc này chỉ ảnh hưởng nếu dùng full-series functions; `_last` variants không bị vấn đề này.

---

### 🟢 m-04: `generate_candlesticks.py` outdated

**File:** `generate_candlesticks.py`

**Vấn đề:** Script này generate `Vec<bool>` API (per-series) nhưng actual `candlestick.rs` đã refactored sang single-index API `(open, high, low, close, i) -> bool`. Script này sẽ OVERWRITE code hiện tại nếu chạy → **nguy hiểm**.

**Đề xuất:** Xóa hoặc move sang `docs/archives/` và đánh dấu deprecated.

---

### 🟢 m-05: `lag_features.rs` unused

**File:** `src/features/lag_features.rs`

**Vấn đề:** `create_lag_features()` và `create_rolling_lags()` KHÔNG được gọi bởi `feature_engine.rs`. Lag features trong `calculate_all()` được tính trực tiếp qua `returns_1_tail()` và `tail_lag_value()`.

**Đề xuất:** Document rõ rằng đây là utility functions cho future use, hoặc xóa.

---

### 🟢 m-06: SAM template hardcoded bucket name

**File:** `template.yaml:36`

```yaml
BucketName: xgboost-models-store
```

**Vấn đề:** S3 bucket names phải globally unique. Hardcoded name sẽ fail nếu bucket đã tồn tại ở account khác.

**Đề xuất:** Dùng parameter hoặc auto-generated name:

```yaml
Parameters:
  ModelBucketName:
    Type: String
    Default: !Sub '${AWS::StackName}-models-${AWS::AccountId}'
```

---

### 🟢 m-07: Thiếu `#[inline]` cho hot-path functions

**Files:** `src/features/price_derived.rs`, `src/features/moving_averages.rs`

**Vấn đề:** Các hàm `_last` variants là hot-path (chỉ tính 1 giá trị cuối cùng, được gọi nhiều lần). LTO "thin" giúp nhưng không đảm bảo cross-crate inlining.

**Đề xuất:** Thêm `#[inline]` cho `returns_n_last`, `sma_last`, `ema_last`, `rsi_last_from_gains_losses`, `atr_last`.

---

### 🟢 m-08: `assemble_feature_vector` không được sử dụng

**File:** `src/feature_engine.rs:251-261`

**Vấn đề:** Public method `FeatureEngine::assemble_feature_vector()` không được gọi trong project.

---

### 🟢 m-09: `rust-toolchain.toml` quá ngắn

**File:** `rust-toolchain.toml`

**Đề xuất:** Xem xét pin phiên bản cụ thể thay vì channel, ví dụ:

```toml
[toolchain]
channel = "1.75.0"
components = ["rustfmt", "clippy"]
```

---

## 6. Phân tích theo file

| File | LOC | Chất lượng | Ghi chú |
| --- | --- | --- | --- |
| `src/lib.rs` | 15 | ⭐⭐⭐⭐⭐ | Clean module structure |
| `src/error.rs` | 29 | ⭐⭐⭐⭐⭐ | Well-structured error types |
| `src/ohlcv.rs` | 40 | ⭐⭐⭐ | Thiếu length validation (C-03) |
| `src/feature_engine.rs` | 263 | ⭐⭐⭐⭐ | Solid, cần fix div-by-zero (C-02), dead code (m-02, m-08) |
| `src/model_manager.rs` | 76 | ⭐⭐⭐ | Lock poisoning risk (C-01), TOCTOU (M-03) |
| `src/xgboost_inference.rs` | 88 | ⭐⭐⭐ | NaN panic risk (C-04), clean stub pattern |
| `src/features/price_derived.rs` | 99 | ⭐⭐⭐⭐ | Division by zero risk (C-02) |
| `src/features/moving_averages.rs` | 114 | ⭐⭐⭐⭐ | Correct sliding window, padding issue (m-03) |
| `src/features/indicators.rs` | 459 | ⭐⭐⭐⭐ | Comprehensive, filtering issue (M-07) |
| `src/features/advanced.rs` | 128 | ⭐⭐⭐⭐⭐ | Clean, correct |
| `src/features/lag_features.rs` | 35 | ⭐⭐⭐ | Unused, padding = 0.0 |
| `src/features/candlestick.rs` | 1087 | ⭐⭐⭐ | Functional nhưng quá dài, lặp code (m-01) |
| `lambda/src/handler.rs` | 386 | ⭐⭐⭐⭐ | Well-tested, parallel processing, SQS output |
| `lambda/src/main.rs` | 17 | ⭐⭐⭐⭐⭐ | Clean Lambda entry point |
| `lambda/src/s3_client.rs` | 33 | ⭐⭐⭐ | SDK config recreated per call (M-02) |
| `lambda/src/sqs_client.rs` | 27 | ⭐⭐⭐ | SDK config recreated per call (M-02) |
| `scripts/deploy_lambda.py` | 507 | ⭐⭐⭐⭐⭐ | Excellent automation, Zig detection |
| `scripts/validate_feature_parity.py` | 109 | ⭐⭐⭐⭐ | Good cross-language validation |
| `tests/feature_tests.rs` | 118 | ⭐⭐⭐⭐ | Good coverage, lacks edge cases |
| `tests/inference_tests.rs` | 66 | ⭐⭐⭐⭐ | Solid stub tests |

---

## 7. Đề xuất hành động

### 🔥 Ưu tiên cao (fix trước khi deploy production)

| # | Issue | File | Effort |
| --- | --- | --- | --- |
| 1 | Fix `unwrap()` trên `RwLock` | `model_manager.rs` | 30 min |
| 2 | Guard division by zero trong returns | `price_derived.rs`, `feature_engine.rs` | 30 min |
| 3 | Validate `OHLCVData` vector lengths | `ohlcv.rs` | 20 min |
| 4 | Fix `partial_cmp().unwrap()` trên NaN | `xgboost_inference.rs` | 10 min |
| 5 | Thống nhất build architecture (ARM vs x86) | `build.sh`, `deploy_lambda.py` | 15 min |

### 🔧 Ưu tiên trung bình (sprint tới)

| # | Issue | File | Effort |
| --- | --- | --- | --- |
| 6 | Cache AWS SDK config với OnceLock | `s3_client.rs`, `sqs_client.rs` | 1 hour |
| 7 | Fix TOCTOU trong ModelManager | `model_manager.rs` | 30 min |
| 8 | Extract feature count thành constant | multiple files | 30 min |
| 9 | Fix Stochastic RSI 50.0/0.0 filtering | `indicators.rs` | 1 hour |
| 10 | Xóa/archive `generate_candlesticks.py` | root | 5 min |

### 📋 Ưu tiên thấp (technical debt)

| # | Issue | File | Effort |
| --- | --- | --- | --- |
| 11 | Refactor `candlestick.rs` dùng macro/trait | `candlestick.rs` | 3 hours |
| 12 | Remove unused `FeatureCache`, `lag_features`, `assemble_feature_vector` | multiple | 30 min |
| 13 | Fix SMA/WMA/ROC padding (0.0 → NaN) | `moving_averages.rs`, `advanced.rs` | 30 min |
| 14 | Parameterize S3 bucket name in SAM template | `template.yaml` | 15 min |
| 15 | Add `#[inline]` annotations | `price_derived.rs`, etc. | 15 min |

---

## 8. Tóm tắt điểm số

| Tiêu chí | Điểm (1-10) | Ghi chú |
| --- | --- | --- |
| **Kiến trúc** | 8/10 | Tách biệt tốt, workspace Cargo hợp lý |
| **Correctness** | 6/10 | Nhiều potential panics (C-01, C-02, C-04), bounds check thiếu |
| **Performance** | 8/10 | AVX2, parallel processing, OnceLock. AWS SDK cần cache (M-02) |
| **Security** | 7/10 | Input validation tốt, nhưng S3 bucket hardcoded thêm IAM policies quá rộng (SQSFullAccess) |
| **Testing** | 6/10 | Có tests cơ bản nhưng thiếu edge cases (empty data, NaN, extreme values), không có integration tests cho Lambda handler |
| **Documentation** | 7/10 | README tốt, docs đầy đủ, nhưng inline comments ít |
| **Maintainability** | 6/10 | candlestick.rs quá dài, dead code, magic numbers |
| **Deploy** | 9/10 | deploy_lambda.py xuất sắc, feature parity validation |

### **Tổng điểm: 7.1/10** — *Production-ready sau khi fix các vấn đề Critical*

---

## Phụ lục: Security Review

### IAM Permissions

- `AmazonSQSFullAccess` quá rộng → nên dùng custom policy chỉ cho phép `sqs:SendMessage` tới queue cụ thể.
- `AmazonS3ReadOnlyAccess` OK cho read model files.

### Input Validation

- ✅ Minimum 50 data points validated
- ✅ Empty symbol/data rejected
- ✅ Invalid mode rejected
- ❌ Không giới hạn batch size → potential DoS (1000 symbols per request)
- ❌ Không validate `model_s3_key` format → potential S3 path injection

### Lambda Security

- `template.yaml` memory 3008MB là hợp lý cho inference
- Timeout 30s là hợp lý cho single inference, có thể tight cho large batches
- Không có Lambda reserved concurrency → unbounded scaling cost risk

---

*Review hoàn thành. Vui lòng tạo issues cho các vấn đề Critical trước khi triển khai production.*
