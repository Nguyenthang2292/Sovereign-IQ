# XGBoost Serverless - Phase 1: Infrastructure & Module Setup

## Goal

Set up the complete Rust workspace structure and AWS infrastructure foundation for the XGBoost serverless module.

## Tasks

### 1. Create Module Directory Structure

- [x] Create `modules/xgboost_LTS_serverless/` directory
- [x] Create subdirectories: `src/`, `lambda/`, `scripts/`, `tests/`, `docs/`
- [x] Create `src/features/` subdirectory
- [x] Create `tests/test_data/` subdirectory
- [x] Create `lambda/src/` subdirectory

→ **Verify:** `tree modules/xgboost_LTS_serverless` shows all directories

### 2. Initialize Cargo Workspace

- [x] Create root `Cargo.toml` with workspace config (`[workspace]`, `members = [".", "lambda"]`)
- [x] Create root `[package]` section: name="xgboost_serverless", version="0.1.0", edition="2021"
- [x] Add profile.release optimization: `opt-level=3, lto="thin", strip=true, codegen-units=1`
- [x] Create `rust-toolchain.toml` with channel="1.70" (or stable)

→ **Verify:** `cargo check` in root runs without errors

### 3. Add Core Dependencies

- [x] Add to root Cargo.toml: `xgboost-rs = "0.3"`, `serde = { version = "1.0", features = ["derive"] }`
- [x] Add: `serde_json = "1.0"`, `ndarray = "0.15"`, `rayon = "1.8"`
- [x] Add: `thiserror = "1.0"`, `anyhow = "1.0"`, `once_cell = "1.19"`
- [x] Add ta lib: `ta = "0.5"` (or plan to implement from scratch)
- [x] Add dev-dependencies: `proptest = "1.4"`, `approx = "0.5"`

→ **Verify:** `cargo fetch` downloads all dependencies successfully

### 4. Create Core Library Files

- [x] Create `src/lib.rs` with module declarations and basic exports
- [x] Create `src/ohlcv.rs` with `OHLCVData` struct (timestamp, open, high, low, close, volume as `Vec<f64>`)
- [x] Create `src/error.rs` with `XGBoostError` enum (ValidationError, ModelNotFoundError, etc.)
- [x] Create `src/feature_engine.rs` with `FeatureEngine` struct skeleton
- [x] Create `src/features/mod.rs` with submodule declarations

→ **Verify:** `cargo build` compiles without errors

### 5. Create Feature Module Skeletons

- [x] Create `src/features/price_derived.rs` with function signatures (returns_1, returns_5, etc.)
- [x] Create `src/features/indicators.rs` with RSI, MACD, ATR signatures
- [x] Create `src/features/moving_averages.rs` with SMA signature
- [x] Create `src/features/candlestick.rs` with `CandlestickPatterns` struct
- [x] Create `src/features/advanced.rs` with ROC, rolling stats signatures
- [x] Create `src/features/lag_features.rs` with lag creation signature

→ **Verify:** `cargo build` compiles, all modules accessible

### 6. Initialize Lambda Package

- [x] Create `lambda/Cargo.toml` with package config and dependencies
- [x] Add dependencies: `xgboost_serverless = { path = "../" }`, `lambda_runtime = "0.8"`
- [x] Add: `tokio = { version = "1", features = ["macros", "rt-multi-thread"] }`
- [x] Add AWS SDK: `aws-config = "1.0"`, `aws-sdk-s3 = "1.0"`, `aws-sdk-sqs = "1.0"`
- [x] Add: `tracing = "0.1"`, `tracing-subscriber = { version = "0.3", features = ["json"] }`

→ **Verify:** `cargo check -p xgboost_lambda` runs without errors

### 7. Create Lambda Handler Files

- [x] Create `lambda/src/main.rs` with tokio main and lambda_runtime::run
- [x] Create `lambda/src/handler.rs` with request/response structs and handler skeleton
- [x] Create `lambda/src/s3_client.rs` with S3Client struct for model downloads
- [x] Add basic request validation function

→ **Verify:** `cargo build -p xgboost_lambda` compiles successfully

### 8. Set Up AWS Infrastructure Files

- [x] Create `template.yaml` with SAM configuration (Function, S3 bucket, SQS queue definitions)
- [x] Create `lambda-trust-policy.json` with Lambda execution role policy
- [x] Add S3ReadPolicy and SQSSendMessagePolicy to template
- [x] Set Lambda config: MemorySize=3008, Timeout=30, Runtime=provided.al2

→ **Verify:** `sam validate` passes without errors

### 9. Create Build and Deploy Scripts

- [x] Create `scripts/deploy.sh` with cargo lambda build and sam deploy commands
- [x] Create `scripts/build.sh` with cargo lambda build --release --arm64
- [x] Make scripts executable: `chmod +x scripts/*.sh`
- [x] Add error handling (set -e) to scripts

→ **Verify:** `bash scripts/build.sh` runs (may fail on cargo lambda if not installed, that's OK)

### 10. Create Initial Documentation

- [x] Create `README.md` with project overview, quick start, architecture diagram
- [x] Create `docs/QUICK_START.md` with installation and first-run instructions
- [x] Create `CHANGELOG.md` with v0.1.0 initial commit entry
- [x] Create `.gitignore` with Rust patterns (target/, Cargo.lock for lib, *.json models)

→ **Verify:** Files exist and are readable

## Done When

- [x] All 10 tasks completed
- [x] `cargo build --workspace` compiles successfully
- [x] `tree modules/xgboost_LTS_serverless` shows complete structure
- [ ] `sam validate --template template.yaml` passes (SAM not installed)
- [x] Git status shows all new files ready to commit

## Notes

- **Cargo Lambda**: If not installed, run `cargo install cargo-lambda` (requires Docker for cross-compilation)
- **AWS SAM CLI**: Required for deployment - install from <https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html>
- **Feature Implementation**: This phase only creates structure - actual feature calculations come in Phase 2
- **Dependencies**: Some dependencies (like xgboost-rs) may require system libraries - document in README if needed
