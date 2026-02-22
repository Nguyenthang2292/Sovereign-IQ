# XGBoost Phase 3: XGBoost Model Integration (Section 4)

## Goal

Wire up real XGBoost inference: align error types, add feature-count validation, connect S3 download to ModelManager, add inference tests, and create the Python train+upload script.

## Current State

- `xgboost_inference.rs` — dual `#[cfg(feature="xgboost")]` stub exists; real path compiles but is untested
- `model_manager.rs` — loads from `/tmp` only; no S3 download logic (lives in lambda `s3_client.rs`)
- `error.rs` — has `PredictionError` but design doc requires `InferenceError` + `InvalidFeatureCount`
- `Cargo.toml` — `xgboost-rs = "0.3"` name may not match the `use xgboost::Booster` code path
- `tests/inference_tests.rs` — does not exist yet

---

## Tasks

- [x] **Task 1: Align error variants in `error.rs`**
  - Rename `PredictionError` → `InferenceError(String)`
  - Add `InvalidFeatureCount { expected: usize, got: usize }`
  - Verify: `cargo check` passes; grep confirms both new names exist

- [x] **Task 2: Add feature-count guard in `xgboost_inference.rs`**
  - At top of `predict()`, add: `if features.len() != 92 { return Err(InvalidFeatureCount { expected: 92, got: features.len() }) }`
  - Works in BOTH `cfg` branches (stub + real)
  - Verify: `cargo check` clean

- [x] **Task 3: Add `with_cache_dir()` + `load_from_file()` to `ModelManager`**
  - Add `pub fn with_cache_dir(dir: PathBuf) -> Self` constructor
  - Add `pub fn load_into_cache(&self, key: &str, path: &Path) -> Result<Arc<XGBoostModel>>` — used by lambda after S3 download
  - Verify: `cargo check` clean

- [x] **Task 4: Connect S3 download in lambda `handler.rs`**
  - In handler: if `model_manager.get_or_load()` returns `ModelNotFoundError`, call `s3_client.download_model()` → then `model_manager.load_into_cache()`
  - Retry `get_or_load()` after successful download
  - Verify: `cargo check --manifest-path lambda/Cargo.toml` clean

- [x] **Task 5: Fix `xgboost-rs` crate name in `Cargo.toml`**
  - Check which crate name exposes `xgboost::Booster` and `xgboost::DMatrix`
  - Update `[features]` optional dep name to match
  - Add comment: `# requires libxgboost installed; build with --features xgboost`
  - Verify: `cargo check` (default, no feature) passes clean

- [x] **Task 6: Create `tests/inference_tests.rs`**
  - `test_prediction_output_format` — features vec of 92 values → label is UP/DOWN/NEUTRAL, probabilities len==3, confidence in [0,1]
  - `test_invalid_feature_count_returns_error` — 50 features → `InvalidFeatureCount` error
  - `test_probabilities_sum_to_one` (stub path only) — stub [0.1, 0.8, 0.1] sums to 1.0
  - `test_prediction_label_matches_max_probability` — verify label index matches argmax of probabilities
  - Verify: `cargo test` → all tests pass

- [x] **Task 7: Create `scripts/train_and_upload.py`**
  - Implement `train_and_upload(symbol, timeframe, version, bucket)` using existing `DataFetcher` + `train_model_with_cv`
  - Export model with `model.save_model(path)` → upload to S3 with metadata
  - Add `argparse` CLI entrypoint
  - Verify: `python scripts/train_and_upload.py --help` runs without import errors

---

## Done When

- [x] `cargo test` — all tests pass (existing 6 + new 4 inference tests = 10 total)
- [x] `cargo build --workspace` — zero warnings, zero errors
- [x] Lambda `handler.rs` has S3-download-then-cache flow wired up
- [x] `scripts/train_and_upload.py` imports cleanly

## Notes

- The `#[cfg(feature = "xgboost")]` guard is intentional — Tasks 1–4 and 6 target the **stub path** (default build, no libxgboost needed)
- Task 5 only affects the real path; keep guard intact
- `model_manager.rs` stays sync (`RwLock`, not `tokio::RwLock`) — the lambda handler is async but model loading is fast enough to block
- Task order: 1 → 2 → 3 → 4 (dependency chain); Task 5 and 6 are independent; Task 7 is pure Python
