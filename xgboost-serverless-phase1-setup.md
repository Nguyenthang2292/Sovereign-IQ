# XGBoost Serverless - Phase 1: Infrastructure & Module Setup

## Goal
Set up the complete Rust workspace structure and AWS infrastructure foundation for the XGBoost serverless module.

## Tasks

### 1. Create Module Directory Structure
- [ ] Create `modules/xgboost_LTS_serverless/` directory
- [ ] Create subdirectories: `src/`, `lambda/`, `scripts/`, `tests/`, `docs/`
- [ ] Create `src/features/` subdirectory
- [ ] Create `tests/test_data/` subdirectory
- [ ] Create `lambda/src/` subdirectory
→ **Verify:** `tree modules/xgboost_LTS_serverless` shows all directories

### 2. Initialize Cargo Workspace
- [ ] Create root `Cargo.toml` with workspace config (`[workspace]`, `members = [".", "lambda"]`)
- [ ] Create root `[package]` section: name="xgboost_serverless", version="0.1.0", edition="2021"
- [ ] Add profile.release optimization: `opt-level=3, lto="thin", strip=true, codegen-units=1`
- [ ] Create `rust-toolchain.toml` with channel="1.70" (or stable)
→ **Verify:** `cargo check` in root runs without errors

### 3. Add Core Dependencies
- [ ] Add to root Cargo.toml: `xgboost = "0.3"`, `serde = { version = "1.0", features = ["derive"] }`
- [ ] Add: `serde_json = "1.0"`, `ndarray = "0.15"`, `rayon = "1.8"`
- [ ] Add: `thiserror = "1.0"`, `anyhow = "1.0"`, `once_cell = "1.19"`
- [ ] Add ta lib: `ta = "0.5"` (or plan to implement from scratch)
- [ ] Add dev-dependencies: `proptest = "1.4"`, `approx = "0.5"`
→ **Verify:** `cargo fetch` downloads all dependencies successfully

### 4. Create Core Library Files
- [ ] Create `src/lib.rs` with module declarations and basic exports
- [ ] Create `src/ohlcv.rs` with `OHLCVData` struct (timestamp, open, high, low, close, volume as Vec<f64>)
- [ ] Create `src/error.rs` with `XGBoostError` enum (ValidationError, ModelNotFoundError, etc.)
- [ ] Create `src/feature_engine.rs` with `FeatureEngine` struct skeleton
- [ ] Create `src/features/mod.rs` with submodule declarations
→ **Verify:** `cargo build` compiles without errors

### 5. Create Feature Module Skeletons
- [ ] Create `src/features/price_derived.rs` with function signatures (returns_1, returns_5, etc.)
- [ ] Create `src/features/indicators.rs` with RSI, MACD, ATR signatures
- [ ] Create `src/features/moving_averages.rs` with SMA signature
- [ ] Create `src/features/candlestick.rs` with `CandlestickPatterns` struct
- [ ] Create `src/features/advanced.rs` with ROC, rolling stats signatures
- [ ] Create `src/features/lag_features.rs` with lag creation signature
→ **Verify:** `cargo build` compiles, all modules accessible

### 6. Initialize Lambda Package
- [ ] Create `lambda/Cargo.toml` with package config and dependencies
- [ ] Add dependencies: `xgboost_serverless = { path = "../" }`, `lambda_runtime = "0.8"`
- [ ] Add: `tokio = { version = "1", features = ["macros", "rt-multi-thread"] }`
- [ ] Add AWS SDK: `aws-config = "1.0"`, `aws-sdk-s3 = "1.0"`, `aws-sdk-sqs = "1.0"`
- [ ] Add: `tracing = "0.1"`, `tracing-subscriber = { version = "0.3", features = ["json"] }`
→ **Verify:** `cargo check -p xgboost_lambda` runs without errors

### 7. Create Lambda Handler Files
- [ ] Create `lambda/src/main.rs` with tokio main and lambda_runtime::run
- [ ] Create `lambda/src/handler.rs` with request/response structs and handler skeleton
- [ ] Create `lambda/src/s3_client.rs` with S3Client struct for model downloads
- [ ] Add basic request validation function
→ **Verify:** `cargo build -p xgboost_lambda` compiles successfully

### 8. Set Up AWS Infrastructure Files
- [ ] Create `template.yaml` with SAM configuration (Function, S3 bucket, SQS queue definitions)
- [ ] Create `lambda-trust-policy.json` with Lambda execution role policy
- [ ] Add S3ReadPolicy and SQSSendMessagePolicy to template
- [ ] Set Lambda config: MemorySize=3008, Timeout=30, Runtime=provided.al2
→ **Verify:** `sam validate` passes without errors

### 9. Create Build and Deploy Scripts
- [ ] Create `scripts/deploy.sh` with cargo lambda build and sam deploy commands
- [ ] Create `scripts/build.sh` with cargo lambda build --release --arm64
- [ ] Make scripts executable: `chmod +x scripts/*.sh`
- [ ] Add error handling (set -e) to scripts
→ **Verify:** `bash scripts/build.sh` runs (may fail on cargo lambda if not installed, that's OK)

### 10. Create Initial Documentation
- [ ] Create `README.md` with project overview, quick start, architecture diagram
- [ ] Create `docs/QUICK_START.md` with installation and first-run instructions
- [ ] Create `CHANGELOG.md` with v0.1.0 initial commit entry
- [ ] Create `.gitignore` with Rust patterns (target/, Cargo.lock for lib, *.json models)
→ **Verify:** Files exist and are readable

## Done When
- [x] All 10 tasks completed
- [ ] `cargo build --workspace` compiles successfully
- [ ] `tree modules/xgboost_LTS_serverless` shows complete structure
- [ ] `sam validate --template template.yaml` passes
- [ ] Git status shows all new files ready to commit

## Notes
- **Cargo Lambda**: If not installed, run `cargo install cargo-lambda` (requires Docker for cross-compilation)
- **AWS SAM CLI**: Required for deployment - install from https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html
- **Feature Implementation**: This phase only creates structure - actual feature calculations come in Phase 2
- **Dependencies**: Some dependencies (like xgboost-rs) may require system libraries - document in README if needed
