# Quick Start Guide

## Prerequisites

1. **Rust**: Install from <https://rustup.rs/>
2. **cargo-lambda**: `cargo install cargo-lambda`
3. **AWS SAM CLI**: Install from AWS documentation
4. **Docker**: Required for cross-compilation

## Installation

```bash
# Clone the repository
cd modules/xgboost_LTS_serverless

# Fetch dependencies
cargo fetch
```

## First Run

### Local Development

```bash
# Build the project
cargo build

# Run tests
cargo test
```

### Build for Lambda

```bash
# Build release binary
bash scripts/build.sh
```

### Deploy to AWS

```bash
# Deploy
bash scripts/deploy.sh
```

## Configuration

Set environment variables in `template.yaml` or via AWS Console:

- `MODEL_BUCKET`: S3 bucket containing XGBoost models
- `PREDICTION_QUEUE_URL`: SQS queue for results

## API Usage

```bash
# Example prediction request
curl -X POST https://<api-endage>/Prod/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timestamp": 1700000000,
    "data": {
      "timestamp": [...],
      "open": [...],
      "high": [...],
      "low": [...],
      "close": [...],
      "volume": [...]
    }
  }'
```

## Troubleshooting

- **Build fails**: Ensure CMake and XGBoost native libraries are installed
- **SAM not found**: Install AWS SAM CLI
- **Permission errors**: Check IAM roles in template.yaml
