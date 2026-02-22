# Codex Fix — 2026-02-22

## Goal

Fix 11 issues (2 High, 4 Medium, 5 Low) từ `docs/codex_review_2026-02-22.md`. Tổng effort ~4h.

## Tasks

### 🔴 High Priority

- [x] **H1** Convert `ATCConfig.robustness: String` → `Robustness` enum  
  **Files**: `src/lib.rs`, `src/validation.rs`, `src/signal_detection.rs`, `tests/atc_tests.rs`  
  - Thêm `#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]` + `#[serde(rename_all = "PascalCase")]` vào `enum Robustness`  
  - Đổi `pub robustness: String` → `pub robustness: Robustness` trong `ATCConfig`  
  - Xoá `.parse::<Robustness>().expect(...)` trong `compute_symbol_score` — dùng trực tiếp `config.robustness`  
  - Cập nhật `validate_config` bỏ check `VALID_ROBUSTNESS_LEVELS` (enum tự validate)  
  - Cập nhật tất cả test fixtures: `robustness: "Medium".to_string()` → `robustness: Robustness::Medium`  
  - Cập nhật `default_robustness()` → `fn default_robustness() -> Robustness { Robustness::Medium }`  
  → Verify: `cargo test` passes; `serde_json::from_str::<ATCConfig>(r#"{"robustness":"Bad",...}"#)` returns `Err`

- [x] **H2** Convert `SignalResult.signal_type: String` → `SignalType` enum  
  **Files**: `src/lib.rs`, `src/signal_detection.rs`, `src/multi_tf_voting.rs`  
  - Thêm `pub enum SignalType { Long, Short, Neutral }` với serde `rename_all = "UPPERCASE"` vào `src/lib.rs`  
  - Đổi `pub signal_type: String` → `pub signal_type: SignalType` trong `SignalResult`  
  - Cập nhật `compute_symbol_score` return `SignalType::Long/Short/Neutral` thay vì string literals  
  - Cập nhật `aggregate_timeframes` return `SignalType` trong `SignalResult`  
  - Cập nhật `calculate_weighted_score` param `signal_type: &SignalType` thay vì `&str`  
  - Cập nhật tất cả test assertions: `"LONG"` → `SignalType::Long`  
  - RE-EXPORT `SignalType` trong `src/lib.rs` public API  
  → Verify: `cargo test` passes; `result.signal_type == SignalType::Long` compiles

---

### 🟡 Medium Priority

- [x] **M3** Chuẩn hóa validation boundaries — `min_signal` cho phép 0.0 nhưng cần document rõ  
  **File**: `src/validation.rs` (lines 227-235), `src/constants.rs`  
  - Thêm constant `MIN_SIGNAL_VALUE: f64 = 0.0` riêng biệt (khác `MIN_NORMALIZED_VALUE`)  
  - Thêm comment giải thích: `min_signal = 0.0` hợp lệ (means "no minimum filter")`  
  - Giữ nguyên check `< 0.0` cho min_signal, nhưng PHẢI có `>= 0.0` guard tường minh  
  - Thêm test `test_validate_config_min_signal_zero_valid` xác nhận `min_signal = 0.0` passes  
  → Verify: `cargo test test_validate_config` passes; code comment rõ ràng về ý nghĩa 0.0

- [x] **M4** Deduplicate `tf_scores` / `tf_strengths` trong `process_single_symbol`  
  **File**: `src/aggregation.rs` (lines 319-338)  
  - Bỏ `tf_strengths` HashMap, dùng lại `tf_scores.clone()` khi gọi `aggregate_timeframes`  
  - Cập nhật signature `aggregate_timeframes` nếu cần: `tf_strengths` → reuse `tf_scores`  
  - Xoá `tf_strengths.insert(tf, score)` (line 328) và HashMap declaration  
  → Verify: `cargo test` passes; không còn 2 HashMap với identical data

- [x] **M1** Gate `calculate_ema_simple` với `#[cfg(not(feature = "simd"))]`  
  **File**: `src/ma_calculations.rs` (lines 4-30)  
  - Wrap toàn bộ fn bằng `#[cfg(not(feature = "simd"))]` vì SIMD build dùng `calculate_ema_simple_simd`  
  - Xóa `#[allow(dead_code)]` attribute  
  → Verify: `cargo build` passes; `cargo build --features simd` không có dead_code warning

- [x] **M2** Document KAMA SIMD edge case — giải thích `continue` unreachable  
  **File**: `src/ma_simd.rs` (lines 353-374)  
  - Thêm comment trước `if base_idx >= 4` block:  

    ```rust
    // SAFETY: start_idx = length and length >= MIN_LENGTH_NARROW (5), so for all
    // valid positions i >= length, base_idx = i - j*4 >= length - (chunks-1)*4 >= 4.
    // The `continue` branch is unreachable for validated inputs but kept for safety.
    ```  

  → Verify: Không có code thay đổi logic, chỉ thêm comment; `cargo test --features simd` passes

---

### 🟢 Low Priority

- [x] **L1** Xóa `use rayon;` bare import không cần thiết  
  **File**: `src/parallelism.rs` (line 5)  
  - Xóa dòng `use rayon;`  
  - Đảm bảo `rayon::ThreadPoolBuilder`, `rayon::ThreadPool`, `rayon::current_num_threads()` vẫn resolve qua qualified paths  
  → Verify: `cargo build` no warnings

- [x] **L3** Xóa `ScopedBuffer` struct không được dùng  
  **File**: `src/buffer_pool.rs` (lines 54-84)  
  - Xóa struct `ScopedBuffer`, impl block và `Drop` impl  
  - Xóa test `test_scoped_buffer`  
  - GIỮ LẠI `get_buffer`, `return_buffer`, `get_buffer_zero`  
  → Verify: `cargo build` no dead_code warnings; `cargo test` passes

- [x] **L5** Extract `SignalParams` struct để giảm số args của `calculate_layer1_signal`  
  **File**: `src/signal_detection.rs`  
  - Tạo struct `SignalParams { lambda_scaled: f64, decay_scaled: f64, cutout: usize, equity_floor: f64, robustness: Robustness }`  
  - Refactor `calculate_layer1_signal(prices, ma_type, base_length, params: &SignalParams)` (8 args → 4)  
  - Xóa `#[allow(clippy::too_many_arguments)]`  
  - Cập nhật callers trong `compute_symbol_score`  
  → Verify: `cargo clippy -- -D warnings` no warnings; `cargo test` passes

- [x] **L4** Thêm mock SQS client cho Lambda handler success-path test  
  **File**: `lambda/src/handler.rs`  
  - Tạo trait `SqsSender: Send + Sync` với method `send_scan_result`  
  - Implement trait cho `SqsClient` và `MockSqsClient` (struct with `pub called: AtomicBool`)  
  - Refactor `handle_request` nhận `&dyn SqsSender` thay vì `&SqsClient`  
  - Thêm test `test_handler_success_path` dùng `MockSqsClient`  
  → Verify: `cargo test -p atc_lambda` shows 2 tests passing

---

## Done When

- [x] `cargo test` passes (all tests, no regressions)
- [x] `cargo clippy -- -D warnings` zero errors
- [x] `cargo build --features simd --release` compiles
- [x] `ATCConfig.robustness` là enum (breaking change → version bump to `0.2.0`)
- [x] `SignalResult.signal_type` là enum
- [x] CHANGELOG `[Unreleased]` updated

## Notes

- **H1 + H2 là breaking API changes** → cần bump version `Cargo.toml`: `version = "0.2.0"`
- Thực hiện theo thứ tự: H1 → H2 → M3 → M4 → M1 → M2 → Low items
- H1 ảnh hưởng nhiều file nhất, nên làm trước để compile check trước khi tiếp tục
- L4 (mock SQS) có thể làm cuối vì phức tạp nhất về refactor

## Related

- Source: `docs/codex_review_2026-02-22.md`
- Previous fixes: `docs/archive/codex_review_2026-02-21-fix.md` (all resolved)
- CHANGELOG: `CHANGELOG.md` → `[Unreleased]`
