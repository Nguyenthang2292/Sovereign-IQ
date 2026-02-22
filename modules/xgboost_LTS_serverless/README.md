# XGBoost LTS Serverless

Rust-based XGBoost inference module for AWS Lambda, designed for low-latency crypto prediction.

## Architecture

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   API Gateway   │───▶│  Lambda Function  │───▶│      S3         │
│   (REST API)    │     │  (XGBoost)       │     │  (Model Store)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │       SQS        │
                        │  (Predictions)   │
                        └──────────────────┘
```

## Project Structure

```text
modules/xgboost_LTS_serverless/
├── src/                      # Core library
│   ├── features/              # Feature engineering
│   │   ├── price_derived.rs
│   │   ├── indicators.rs
│   │   ├── moving_averages.rs
│   │   ├── candlestick.rs
│   │   ├── advanced.rs
│   │   └── lag_features.rs
│   ├── lib.rs
│   ├── ohlcv.rs
│   ├── error.rs
│   └── feature_engine.rs
├── lambda/                   # Lambda handler
│   └── src/
│       ├── main.rs
│       ├── handler.rs
│       └── s3_client.rs
├── scripts/                  # Build & deploy
├── template.yaml             # SAM template
└── docs/                     # Documentation
```

## Quick Start

From module root:

```bash
cargo fetch
cargo test --workspace
cargo build --workspace
```

Optional feature-path check:

```bash
cargo check --features xgboost
```

Generate feature vector from OHLCV JSON:

```bash
cargo run --bin calculate_features -- tests/test_data/btc_usdt_1h.json
```

Prepare Lambda deployment:

```bash
bash scripts/build.sh
bash scripts/deploy.sh staging us-east-1
```

Run integration smoke test against deployed API:

```bash
python scripts/lambda_demo.py --endpoint https://<api-id>.execute-api.us-east-1.amazonaws.com/staging/predict
```

More docs:

- `docs/QUICK_START.md`
- `docs/AWS_SETUP.md`
- `docs/MODEL_EXPORT.md`
- `docs/DEPLOYMENT_RUNBOOK.md`

## Features

- OHLCV data processing
- Technical indicators (RSI, MACD, ATR)
- Moving averages (SMA, EMA, WMA)
- Candlestick pattern recognition
- Lag feature generation
- Bollinger Bands
- Rolling statistics

## Requirements

- Rust 1.70+
- cargo-lambda
- AWS SAM CLI
- CMake (for XGBoost native libraries)

## License

MIT
