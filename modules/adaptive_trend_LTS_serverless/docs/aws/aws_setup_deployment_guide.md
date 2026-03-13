# AWS Setup and Deployment Guide (ATC Serverless)

This guide documents the **current production invocation model** for
`adaptive_trend_LTS_serverless` (v0.2.x):

```
Client -> boto3 Lambda Invoke (RequestResponse) -> Lambda -> direct ScanResult JSON
```

No Function URL or API Gateway is required for standard operation.

## 1. Prerequisites

- AWS account with permissions for Lambda, IAM, CloudWatch, SNS
- AWS CLI v2 configured (`aws configure`)
- Rust toolchain + `cargo lambda`
- Python 3.8+ for demo scripts

```bash
cargo install cargo-lambda
pip install -r modules/adaptive_trend_LTS_serverless/scripts/requirements.txt
```

## 2. Create IAM Role

```bash
aws iam create-role \
  --role-name ATC-Lambda-ExecutionRole \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[
      {
        "Effect":"Allow",
        "Principal":{"Service":"lambda.amazonaws.com"},
        "Action":"sts:AssumeRole"
      }
    ]
  }'

aws iam attach-role-policy \
  --role-name ATC-Lambda-ExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

Get role ARN:

```bash
aws iam get-role \
  --role-name ATC-Lambda-ExecutionRole \
  --query 'Role.Arn' \
  --output text
```

## 3. Build Lambda Binary

```bash
cd modules/adaptive_trend_LTS_serverless/lambda

# x86_64
cargo lambda build --release --target x86_64-unknown-linux-gnu

# or arm64
# cargo lambda build --release --target aarch64-unknown-linux-gnu
```

## 4. Deploy Lambda

```bash
cd modules/adaptive_trend_LTS_serverless/lambda

cargo lambda deploy atc-serverless \
  --iam-role arn:aws:iam::YOUR_ACCOUNT_ID:role/ATC-Lambda-ExecutionRole \
  --runtime provided.al2 \
  --region us-east-1
```

Recommended runtime settings:

```bash
aws lambda update-function-configuration \
  --function-name atc-serverless \
  --memory-size 1769 \
  --timeout 60 \
  --environment "Variables={RUST_LOG=info}"
```

## 5. Invoke and Verify

### Option A: Demo script (recommended)

```bash
python modules/adaptive_trend_LTS_serverless/scripts/binance_lambda_demo.py \
  --function-name atc-serverless \
  --region us-east-1 \
  --symbols 20
```

### Option B: AWS CLI invoke

Prepare `request.json` with `BatchRequest` payload.

```bash
aws lambda invoke \
  --function-name atc-serverless \
  --region us-east-1 \
  --cli-binary-format raw-in-base64-out \
  --payload file://request.json \
  response.json
```

Inspect:

```bash
cat response.json
```

## 6. CloudWatch Monitoring

The Lambda emits EMF metrics in namespace `ATC/Serverless` with dimension
`FunctionName`.

Primary metrics:

- `MemoryUsageMB`
- `MemoryDeltaMB`
- `SymbolsPerSecond`
- `ThreadCount`
- `ErrorRate`

### Create alarms

Use the helper script:

```powershell
pwsh modules/adaptive_trend_LTS_serverless/scripts/setup_cloudwatch_alarms.ps1 `
  -FunctionName atc-serverless `
  -Region us-east-1
```

### SNS subscriptions

Alarm topics must have subscriptions, otherwise alarms fire without notifications.

Example email subscription:

```bash
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:YOUR_ACCOUNT_ID:atc-serverless-alarms \
  --protocol email \
  --notification-endpoint you@example.com
```

Confirm subscription from your email inbox.

## 7. Operational Notes

- Keep batch size moderate (for example 20-100 symbols per invoke).
- Use Provisioned Concurrency if you need stable low-latency starts.
- Input validation is fail-closed for missing configured timeframes.
- Invocation path is synchronous `RequestResponse`; clients should parse returned
  `ScanResult` payload directly.

## 8. Troubleshooting

### AccessDenied

- Verify IAM role trust policy and attached permissions.
- Verify caller credentials (`aws sts get-caller-identity`).

### NoCredentials / PartialCredentials

- Configure AWS CLI credentials in the runtime environment.
- For local script runs, verify profile and region.

### Slow or timeout

- Increase Lambda memory and timeout.
- Reduce symbol count per batch.

### High error rate

- Check CloudWatch logs for per-symbol validation errors.
- Verify OHLCV completeness for configured timeframes.

## 9. What Changed from Legacy Flow

Removed from primary path:

- Function URL public invocation
- API Gateway bridge for normal scan requests
- SQS result-polling contract

Current contract:

- `boto3.client("lambda").invoke(..., InvocationType="RequestResponse")`
- Lambda returns final `ScanResult` in response payload.

