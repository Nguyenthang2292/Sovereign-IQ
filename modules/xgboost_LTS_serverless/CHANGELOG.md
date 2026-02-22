# Changelog

All notable changes to this project will be documented in this file.

## [0.1.4] - 2026-02-22

### Deployed

- **AWS Lambda `us-east-1`** — `xgboost-serverless-predict` live
  - ARN: `arn:aws:lambda:us-east-1:081338828929:function:xgboost-serverless-predict`
  - Runtime: `provided.al2023` · Memory: 3008 MB · Timeout: 30s
  - Env vars: `RUST_LOG=info`, `MODEL_BUCKET=xgboost-models-store`, `PREDICTION_QUEUE_URL=...`
  - Binary: 15.2 MB compiled with AVX2+Haswell optimizations (`x86_64-unknown-linux-gnu`)

### Fixed (deploy_lambda.py)

- Added `--binary-path target/lambda/bootstrap/bootstrap` so `cargo lambda deploy` resolves the built binary correctly from the workspace root
- Removed `AWS_REGION` from Lambda env-vars (AWS reserved key — causes `InvalidParameterValueException`)
- Fixed IAM: replaced `AmazonSQSFullAccess` managed policy with scoped `sqs:SendMessage` inline policy on the `xgboost-predictions` queue ARN
- Added `simd-json = { version = "0.13", features = ["serde_impl"] }` to `lambda/Cargo.toml`
- Added `#[allow(dead_code)]` + doc comment to `parse_request_simd()` (reserved for future direct invocation path)

## [0.1.3] - 2026-02-22

### Changed

- **OPT-07 Lambda runtime tuning**: Switched Lambda entrypoint runtime from default multi-thread Tokio runtime to `#[tokio::main(flavor = "current_thread")]` in `lambda/src/main.rs` to reduce scheduler overhead for constrained Lambda vCPU environments while keeping `spawn_blocking` behavior intact.

### Verification

- `cargo test -p xgboost_lambda` — ✅ all tests pass

## [0.1.2] - 2026-02-22

### Fixed

- **R-01 `candlestick.rs` refactor**: Collapsed `to_feature_vec()` from 242 lines of repetitive `if/else` blocks into a compact 10-line array + `.map()` expression. File reduced from 1087 LOC → 854 LOC with identical output and zero behaviour change.
- **R-02 Dead code removed**: Deleted `FeatureCache` struct/impl, `pub cache` field on `FeatureEngine`, and unused `assemble_feature_vector()` static method from `feature_engine.rs`. Removed unused `HashMap` import.
- **R-03 SMA/WMA/ROC/rolling_std/rolling_skewness padding**: Changed warmup-period padding from `0.0` → `f64::NAN` in `sma()`, `wma()` (`moving_averages.rs`), `roc()`, `rolling_std()`, `rolling_skewness()` (`advanced.rs`). This ensures XGBoost sees proper missing-value markers instead of misleading zeros.
- **R-04 `log_returns()` zero-guard**: Added `if close[i-1] == 0.0 { NaN }` in `price_derived.rs` to prevent `-Infinity` output. Added same guard to `volatility()` and `roc()` full-series functions in `advanced.rs`.
- **R-05 Stochastic RSI filter corrected**: Removed erroneous `&& v != 50.0 && v != 0.0` conditions from both `stochastic_rsi()` and `stochastic_rsi_last()` in `indicators.rs`. RSI readings of exactly 50.0 or 0.0 are now correctly included rather than silently dropped.

### Verification

- `cargo build --workspace` — ✅ clean compile
- `cargo test --workspace` — ✅ all tests pass

## [0.1.1] - 2026-02-22 (Final Review: 8.1/10 — Production-ready)

### Fixed

- Removed deprecated `generate_candlesticks.py` script from module root.
- Added request batch guard in Lambda validation (`max 50` items) with test coverage.
- Hardened IAM in `template.yaml` to explicit inline `sqs:SendMessage` permission scoped to `PredictionQueue` ARN.
- Marked legacy lag feature helper module as intentionally retained (`#![allow(dead_code)]`) to keep dead-code lint clean.

### Verification

- `cargo test --workspace` passes.
- `cargo clippy --workspace -- -W clippy::all` reports no warnings.
- `cargo build --workspace` completes successfully.
- `python scripts/validate_feature_parity.py` passes using normalized overlap comparison and explicit exclusions for known convention-different features (`obv`, `bbp_5_2_0`).

## [0.1.0] - 2026-02-21

### Added

- Initial project setup
- Rust workspace with library and Lambda binary
- Core library structure:
  - OHLCV data structures
  - Feature engineering modules
  - Error handling
- Lambda handler with basic prediction endpoint
- AWS SAM template for deployment
- Build and deploy scripts
- Documentation (README, QUICK_START)

### Features

- Price-derived features (returns, log returns)
- Technical indicators (RSI, MACD, ATR)
- Moving averages (SMA, EMA, WMA)
- Candlestick pattern detection
- Advanced features (Bollinger Bands, volatility)
- Lag feature generation

### Infrastructure

- AWS Lambda function configuration (3008MB, 30s timeout)
- S3 bucket for model storage
- SQS queue for predictions
- API Gateway integration
