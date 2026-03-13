# ATC Serverless - Adaptive Trend Classification for AWS Lambda

High-performance Rust implementation of Adaptive Trend Classification (ATC),
optimized for AWS Lambda batch scanning.

## Overview

ATC Serverless:

- calculates 6 MA types (EMA, HMA, WMA, DEMA, LSMA, KAMA)
- applies diflen robustness around each base length
- computes Layer 1 signals + Layer 2 equity weights
- aggregates multi-timeframe results
- returns LONG/SHORT/NEUTRAL with score and diagnostics

## Task 3 - Execution Shift Boundary

Nguyen tac dong bo voi adaptive_trend va LTS mini:

- core signal engine tra ve raw causal signal
- strategy/non-repainting shift 1 bar chi nam o execution layer

Ke hoach refactor chi tiet:

- [modules/adaptive_trend_LTS_serverless/docs/2026-03-14-task3-execution-shift-refactor-plan.md](docs/2026-03-14-task3-execution-shift-refactor-plan.md)

## Current Invocation Model (v0.2.x)

The deployed model is **direct AWS SDK invoke**:

```
Python client --[boto3 Lambda Invoke: RequestResponse]--> Lambda --> ScanResult JSON
```

Important:

- no SQS polling for result retrieval
- no Function URL required
- no API Gateway required for normal operation

## Quick Start

```bash
# Python deps for demo tooling
pip install -r scripts/requirements.txt

# Local mock run (no AWS)
python scripts/binance_lambda_demo.py --mock --symbols 10

# Real Lambda run (AWS SDK invoke)
python scripts/binance_lambda_demo.py \
  --function-name atc-serverless \
  --region us-east-1 \
  --symbols 20
```

Reference docs:

- `docs/aws/quick_start.md`
- `docs/aws/aws_setup_deployment_guide.md`
- `docs/aws/binance_lambda_demo_overview.md`

## Project Structure

```
modules/adaptive_trend_LTS_serverless/
|-- Cargo.toml
|-- src/
|   |-- lib.rs
|   |-- ma_calculations.rs
|   |-- signal_detection.rs
|   |-- equity.rs
|   |-- multi_tf_voting.rs
|   |-- validation.rs
|   |-- parallelism.rs
|   `-- buffer_pool.rs
|-- lambda/
|   |-- Cargo.toml
|   `-- src/
|       |-- main.rs
|       `-- handler.rs
|-- scripts/
|   |-- binance_lambda_demo.py
|   `-- deploy_lambda.py
`-- tests/
    `-- atc_tests.rs
```

## Build

```bash
cd modules/adaptive_trend_LTS_serverless
cargo build
cargo test

cd lambda
cargo lambda build --release
```

## Lambda Request / Response

The Lambda handler accepts a `BatchRequest` payload and returns `ScanResult`.

Request shape:

```json
{
  "batch_id": "batch-001",
  "symbols": [
    {
      "symbol": "BTCUSDT",
      "timeframes": {
        "1h": {
          "timestamp": [1704067200],
          "open": [42000.0],
          "high": [42200.0],
          "low": [41900.0],
          "close": [42100.0],
          "volume": [100.0]
        },
        "4h": {
          "timestamp": [1704067200],
          "open": [42000.0],
          "high": [42300.0],
          "low": [41800.0],
          "close": [42150.0],
          "volume": [350.0]
        }
      }
    }
  ],
  "apply_strategy_shift": false,
  "config": {
    "weights": { "1h": 0.6, "4h": 0.4 },
    "threshold": 0.3,
    "min_signal": 0.0,
    "use_signal_strength": true,
    "lambda_param": 0.02,
    "decay": 0.03,
    "cutout": 0,
    "equity_floor": 0.25,
    "robustness": "Medium",
    "ma_configs": [
      { "ma_type": "EMA", "length": 12, "weight": 1.0 }
    ]
  }
}
```

Note: every timeframe key in `config.weights` must exist under each symbol's
`timeframes` map.

Response notes:

- `results[*].score` is raw causal snapshot score.
- `results[*].average_signal_raw` (optional) mirrors raw score contract.
- `results[*].average_signal_exec` is optional and may be omitted for snapshot API.
- `apply_strategy_shift` is an adapter hint only; core Rust signal math remains raw.

## Deploy

```bash
cd modules/adaptive_trend_LTS_serverless/lambda

# Build binary for Lambda
cargo lambda build --release --target x86_64-unknown-linux-gnu

# Deploy function (replace role ARN)
cargo lambda deploy atc-serverless \
  --iam-role arn:aws:iam::YOUR_ACCOUNT:role/YOUR_LAMBDA_ROLE \
  --runtime provided.al2

# Optional runtime tuning
aws lambda update-function-configuration \
  --function-name atc-serverless \
  --memory-size 1769 \
  --timeout 60
```

Smoke test:

```bash
python modules/adaptive_trend_LTS_serverless/scripts/binance_lambda_demo.py \
  --function-name atc-serverless \
  --region us-east-1 \
  --symbols 20
```

## Tests

```bash
cd modules/adaptive_trend_LTS_serverless
cargo test
cargo test -- --nocapture
```

## Notes on Algorithm Stability

`compute_symbol_score` in `src/signal_detection.rs` intentionally follows legacy behavior:

1. Layer 1 output is thresholded to discrete vote `{-1, 0, 1}`
2. Layer 2 equity contributes the weight
3. final score is weighted average of discrete votes

Do not switch back to raw continuous Layer 1 weighting without parity validation
against the original adaptive trend implementation.
