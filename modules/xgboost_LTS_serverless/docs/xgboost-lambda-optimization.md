# XGBoost Lambda Optimization

## Goal

Tối ưu CPU efficiency, memory, và parallelism của xgboost_LTS_serverless để giảm latency trên AWS Lambda từ ~5.5/10 lên 9/10.

---

## Tasks

### 🔴 P0 — Critical (CPU / latency trực tiếp)

- [x] **T1: Compute-only-last cho mọi indicator** → Thay vì tính toàn bộ chuỗi O(N) rồi lấy index cuối, refactor mỗi hàm chỉ trả về scalar cuối.
  - File: `src/feature_engine.rs`, `src/features/indicators.rs`, `src/features/moving_averages.rs`, `src/features/advanced.rs`
  - Pattern: `fn sma_last(data: &[f64], period: usize) -> f64`
  - Verify: `cargo test` pass, output vector vẫn đúng 92 features

- [x] **T2: Sliding-sum SMA** → Thay `windows().sum()` O(N×period) bằng sliding window O(N).
  - File: `src/features/moving_averages.rs`
  - Pattern:

    ```rust
    let mut sum: f64 = data[..period].iter().sum();
    // slide: sum += data[i] - data[i - period]
    ```

  - Verify: `cargo test`, giá trị SMA(20/50/200) khớp với baseline

- [x] **T3: WMA sliding O(N)** → Thay double loop O(N×period²) bằng sliding weighted sum.
  - File: `src/features/moving_averages.rs`
  - Verify: `cargo test` moving_averages tests pass

- [x] **T4: RSI shared gains/losses** → Tính `gains` và `losses` 1 lần, share cho RSI(9), RSI(14), RSI(25).
  - File: `src/features/indicators.rs` — extract `compute_gains_losses(close) -> (Vec<f64>, Vec<f64>)`
  - Sau đó: `rsi_from_gains_losses(gains, losses, period) -> Vec<f64>`
  - Verify: RSI values giống hệt baseline

- [x] **T5: Candlestick detect only last bar** → `CandlestickPatterns::detect()` chỉ tính cho index cuối `(N-1)`, không loop toàn bộ chuỗi.
  - Pattern: `detect_at(open: &[f64], ..., i: usize) -> CandlestickPatterns`
  - Verify: test detect candlestick đúng ở sample index., compile OK

---

### 🟠 P1 — High (Memory / clone)

- [x] **T6: Loại bỏ `.clone()` trong FeatureCache** → Trả về `&[f64]` thay vì `Vec<f64>` từ cache.
  - File: `src/feature_engine.rs`
  - Thay: `get_or_insert(...).clone()` → lưu kết quả vào local ref `&[f64]`
  - Verify: `FeatureCache::get_or_insert` đã trả `&[f64]`, bỏ `Arc<Vec<f64>>` và bỏ `.clone()` trong hot path

- [x] **T7: Merge `indicators::ema` và `moving_averages::ema`** → Xóa private `ema()` trong `indicators.rs`, dùng `super::moving_averages::ema()`.
  - File: `src/features/indicators.rs`
  - Verify: `cargo build` OK, MACD output không thay đổi

---

### 🟡 P2 — Medium (Parallelism)

- [x] **T8: Tokio Parallel batch inference** → Thay đổi loop `for item in request.requests` thành `tokio::spawn` và `futures::future::join_all` (hoặc `JoinSet`).
  - File: `lambda/src/handler.rs`
  - Pattern:

    ```rust
    let futs: Vec<_> = request.requests.iter().map(|item| tokio::spawn(...)).collect();
    let results = futures::future::join_all(futs).await;
    ```

  - Cần thêm: Không bắt buộc nếu dùng `JoinSet` (hiện đang dùng `tokio::task::JoinSet`)
  - Verify: batch request với 3 symbols trả về tất cả predictions, không lỗi race

- [x] **T9: Thêm concurrency (nếu cần) với Rayon** -> Skipped, tokio handles batch concurrency adequately.
  - File: `src/feature_engine.rs`
  - Pattern: `rayon::join(|| calc_a(), || calc_b())`
  - LƯU Ý: Phải cẩn thận với FeatureCache (vì `&mut self`). Có thể cần `moka::DashMap` hoặc chia phase tính toán rõ ràng.

---

### 🔵 P3 — Cleanup (Dependencies)

- [x] **T10: Xóa unused dependencies** → Remove `ndarray`, `ta`, `once_cell` khỏi `Cargo.toml`.
  - File: `Cargo.toml`
  - Check: `cargo +nightly udeps` hoặc kiểm tra manual import
  - Verify: `cargo build --release` thành công, binary size giảm

- [x] **T11: Xóa imports `rayon` không cần thiết** → Nếu không làm T9, xóa hẳn `rayon` khỏi `Cargo.toml` và các file `features/*.rs`.
  - Verify: code "sạch", không có IDE warnings. `unused extern crate`

---

### ✅ Phase Cuối — Verification

- [x] **T12: Benchmark / Time Check** → So sánh thời gian chạy `calculate_all` (có thể dùng test hoặc dummy binary).
  - Verify: benchmark local `calculate_features` (50 lần) với 1000 nến: **avg ~10.12ms**/lần

- [x] **T13: Chạy full test suite** → `cargo test` phải xanh hết, đặc biệt là size vector 48 (candlestick) và 92 (tất cả).
  - Lệnh: `cargo test` ✅ pass (feature tests + inference tests).
  - Lệnh: `cargo clippy --all-targets -- -D warnings` ✅ pass.

- [ ] **T14: Release Build + Deploy Test** → Build bằng `cargo-lambda`.
  - Lệnh: `cargo lambda build --release --arm64` (hoặc test với script python)
  - Verify: ✅ `cargo lambda build --release --arm64 --skip-target-check` thành công sau khi thêm target `aarch64-unknown-linux-gnu` và Zig user-scope
  - Verify: ✅ ARM64 bootstrap size ~15.26MB (`target/lambda/bootstrap/bootstrap`), nhỏ hơn ngưỡng 20MB
  - Verify: ⏳ Chưa verify deploy/invoke production hoặc staging (cần endpoint + credentials AWS)

---

## Done When

- [x] Feature calculation thực hiện **compute-only-last** cho tất cả indicators
- [ ] Sliding-sum SMA/WMA thay thế naive O(N×period)
- [ ] RSI chỉ tính gains/losses 1 lần
- [ ] Candlestick chỉ scan last bar
- [x] Không còn `Vec::clone()` không cần thiết trong hot path
- [ ] Unused deps (`ndarray`, `ta`, `once_cell`) đã xóa
- [x] Tất cả `cargo test` pass
- [ ] Lambda latency target: inference < 500ms (warm), cold start < 3s

---

## Notes

### Cập nhật thực tế (2026-02-21)

- Đã verify bằng chạy thực tế: `cargo test`, `cargo clippy --all-targets -- -D warnings`, `cargo build --release`, `cargo build -p xgboost_lambda --release`.
- Đã hoàn tất refactor compute-only-last trong `FeatureEngine` + helper `_last` ở `price_derived`, `moving_averages`, `indicators`, `advanced`.
- Đã hoàn tất tối ưu clone: `FeatureCache` trả `&[f64]`, bỏ `Arc<Vec<f64>>` và `.clone()`.
- Đã benchmark local: 1000 nến, 50 iterations, trung bình ~10.12ms/lần cho binary `calculate_features`.
- Đã build ARM64 bằng cargo-lambda, bootstrap ~15.26MB (<20MB).
- Deploy/invoke trên AWS Lambda chưa chạy trong workspace hiện tại vì thiếu endpoint/credentials.
- `T8` đang dùng `JoinSet`; vì vậy không cần thêm dependency `futures` nếu giữ implementation hiện tại.

- **Thứ tự thực hiện**: T1→T2→T3→T4→T5 (P0 trước, từng task một, test sau mỗi task)
- **T1 là refactor lớn nhất** — cần đổi signature của nhiều public fn. Làm T2-T5 song song với T1 sau khi đã thiết kế xong signature mới.
- **T8 cần cẩn thận** với `MODEL_MANAGER: OnceLock` — hiện tại dùng `JoinSet` + `spawn_blocking` cho feature calculation, cần giữ thread-safe access như hiện trạng
- Binary name trong `lambda/Cargo.toml` là `bootstrap` (không phải `xgboost_lambda`), deploy script đã handle đúng
