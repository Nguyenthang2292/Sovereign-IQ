# Binance Lambda Demo Overview

This document explains `scripts/binance_lambda_demo.py` for the current
serverless model (v0.2.x).

## Invocation Model

The demo uses direct AWS SDK invocation:

```
binance_lambda_demo.py
  -> ATCLambdaClient (boto3)
  -> Lambda Invoke (RequestResponse)
  -> ScanResult JSON response
```

No Function URL, API Gateway, or SQS result polling is required.

## What the Script Does

1. Fetches symbols and OHLCV from Binance.
2. Builds `BatchRequest` payload (`batch_id`, `symbols`, `config`).
3. Invokes Lambda via `ATCLambdaClient`.
4. Prints ranked LONG/SHORT/NEUTRAL results.

## Requirements

```bash
pip install -r modules/adaptive_trend_LTS_serverless/scripts/requirements.txt
```

AWS credentials must be available in your environment.

## Basic Usage

```bash
# Mock mode (no AWS)
python modules/adaptive_trend_LTS_serverless/scripts/binance_lambda_demo.py \
  --mock \
  --symbols 10

# Real invocation
python modules/adaptive_trend_LTS_serverless/scripts/binance_lambda_demo.py \
  --function-name atc-serverless \
  --region us-east-1 \
  --symbols 10
```

## Common Commands

```bash
# More symbols
python modules/adaptive_trend_LTS_serverless/scripts/binance_lambda_demo.py \
  --function-name atc-serverless \
  --region us-east-1 \
  --symbols 50

# All symbols (expensive / slow)
python modules/adaptive_trend_LTS_serverless/scripts/binance_lambda_demo.py \
  --function-name atc-serverless \
  --region us-east-1 \
  --all-symbols \
  --timeframes 1h

# Show per-timeframe strengths
python modules/adaptive_trend_LTS_serverless/scripts/binance_lambda_demo.py \
  --function-name atc-serverless \
  --region us-east-1 \
  --symbols 20 \
  --details

# Custom config
python modules/adaptive_trend_LTS_serverless/scripts/binance_lambda_demo.py \
  --function-name atc-serverless \
  --region us-east-1 \
  --symbols 20 \
  --config modules/adaptive_trend_LTS_serverless/scripts/custom_config.json
```

## CLI Arguments

- `--symbols <int>`: number of symbols (default `10`)
- `--all-symbols`: process all available USDT symbols
- `--timeframes <list>`: default `1h 4h`
- `--limit <int>`: candles per timeframe (default `100`)
- `--details`: print strength breakdown
- `--config <path>`: custom config JSON
- `--mock`: skip AWS and return synthetic output
- `--function-name <name>`: Lambda function name (default `atc-serverless`)
- `--region <region>`: AWS region (default `us-east-1`)

## Expected Output

The script prints:

- fetch status
- invocation duration
- sorted result table by absolute score
- summary counts for LONG/SHORT/NEUTRAL

## Troubleshooting

### `NoCredentialsError` or `PartialCredentialsError`

Configure AWS credentials:

```bash
aws configure
aws sts get-caller-identity
```

### Invocation fails with function error

- Check Lambda logs in CloudWatch.
- Verify payload structure and timeframe coverage.
- Verify function name and region.

### Slow run

- Reduce `--symbols`.
- Reduce number of `--timeframes`.
- Increase Lambda memory if needed.

## Related Files

- `scripts/binance_lambda_demo.py`
- `lambda_client.py`
- `docs/aws/quick_start.md`
- `docs/aws/aws_setup_deployment_guide.md`

