# xgboost_LTS_serverless — Fix Plan

> Source: `docs/2026-02-22-code-review.md`  
> Created: 2026-02-22

## Goal

Fix 4 Critical bugs (panic-risk in production) + 5 Major issues trước khi deploy Lambda lên production.

---

## Phase 1 — Critical Fixes (Panic-risk)

- [x] **C-01** `src/model_manager.rs` — Thay `RwLock::unwrap()` bằng `unwrap_or_else(|p| p.into_inner())` tại 3 chỗ (line 40, 54, 70).  
  → Verify: `cargo test --workspace` pass, không còn `unwrap` trên lock result.

- [x] **C-02** `src/features/price_derived.rs` — Thêm guard `if close[i-n] == 0.0 { f64::NAN }` trong `returns_n_last()` (line 14) và `returns_n()` (line 4) trước khi chia.  
  → Verify: Unit test với input `close = [0.0, 1.0]` → trả về `NaN`, không trả `Infinity`.

- [x] **C-02b** `src/feature_engine.rs` — `returns_1_tail()` (line 65): thêm guard tương tự, dùng `safe_ratio()` đã có sẵn.  
  → Verify: Test với `close = [0.0, 1.0, 2.0]` không panic.

- [x] **C-03** `src/ohlcv.rs` — Đổi `OHLCVData::new()` trả về `Result<Self, XGBoostError>`, validate tất cả vectors cùng length.  
  → Cập nhật tất cả call sites (`feature_tests.rs`, `handler.rs`, `calculate_features.rs`).  
  → Verify: `cargo build --workspace` pass. Test với vectors length khác nhau → trả `Err`.

- [x] **C-04** `src/xgboost_inference.rs` line 62 — Thay `.partial_cmp(b.1).unwrap()` thành `.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)`.  
  → Verify: Test với `features = vec![f64::NAN; 92]` không panic (chỉ chạy khi không có `xgboost` feature).

---

## Phase 2 — Major Issues

- [x] **M-01** Thống nhất build architecture: chọn `x86_64` (có AVX2).  
  Sửa `scripts/build.sh` line 10: `--arm64` → `--target x86_64-unknown-linux-gnu`.  
  Thêm `RUSTFLAGS="-C target-cpu=haswell -C target-feature=+avx2"` vào `build.sh`.  
  → Verify: `bash scripts/build.sh` sinh ra binary tại `target/lambda/bootstrap` cho x86_64.

- [x] **M-02** `lambda/src/s3_client.rs` và `sqs_client.rs` — Tạo `static AWS_CONFIG: OnceLock<SdkConfig>` trong `handler.rs`, khởi tạo 1 lần trong `handle_request()`, truyền vào S3Client/SqsClient.  
  → Verify: Log Lambda cold start time giảm, không tạo lại config mỗi request.

- [x] **M-03** `src/model_manager.rs` `get_or_load()` — Thêm double-checked locking: Re-check cache sau khi lấy write lock trước khi load model.  
  → Verify: Chạy concurrent test (`std::thread::spawn` x10 cùng load 1 key) — model chỉ được đọc từ disk 1 lần.

- [x] **M-04** Tạo constant `EXPECTED_FEATURE_COUNT: usize = 92` trong `src/lib.rs`, thay tất cả literal `92` ở `feature_engine.rs`, `xgboost_inference.rs`, `calculate_features.rs`, `tests/inference_tests.rs`.  
  → Verify: `grep -r " 92" src/ tests/` trả về 0 kết quả.

- [x] **M-05** `template.yaml` — Parameterize bucket name: thêm `Parameters.ModelBucketName` và fallback động bằng `!Sub '${AWS::StackName}-models-${AWS::AccountId}'` khi không truyền parameter.  
  → Verify: `sam validate` pass, `sam deploy` không hardcode bucket name.

---

## Phase 3 — Cleanup & Safety (Minor)

- [x] **m-01** Xóa `generate_candlesticks.py` khỏi root (outdated, dangerous if run).  
  → Verify: File không còn tồn tại.

- [x] **m-02** Đánh dấu `src/features/lag_features.rs` là `#[allow(dead_code)]` hoặc remove nếu không dùng.  
  Tương tự `FeatureEngine::assemble_feature_vector()` và `FeatureCache`.  
  → Verify: `cargo clippy -- -D dead_code` không report warnings.

- [x] **m-03** `template.yaml` IAM: Thay `AmazonSQSFullAccess` bằng inline policy chỉ `sqs:SendMessage` tới `!GetAtt PredictionQueue.Arn`.  
  → Verify: `sam validate` pass, Lambda chỉ có quyền SendMessage.

- [x] **m-04** Thêm `max_batch_size` validation trong `handler.rs::validate_request()`: reject nếu `request.requests.len() > 50`.  
  → Verify: Test với 51 items → trả về `ValidationError`.

---

## Phase 4 — Verification

- [x] Chạy `cargo test --workspace` — tất cả tests pass.
- [x] Chạy `cargo clippy --workspace -- -W clippy::all` — 0 warnings.
- [x] Chạy `cargo build --workspace` — compile thành công.
- [x] Chạy `python scripts/validate_feature_parity.py` — PASSED.
- [x] Review diff cuối cùng, update `CHANGELOG.md`.

---

## Done When

- [x] 0 `unwrap()` trên lock results
- [x] `OHLCVData::new()` validate lengths
- [x] Không còn division by zero unguarded
- [x] `partial_cmp` NaN-safe
- [x] `x86_64` là duy nhất build target
- [x] `cargo test --workspace` green
