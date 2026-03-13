# Quick Start - ATC Serverless

Run the Binance demo against deployed Lambda in about 15 minutes.

## Current Runtime Contract

```
Demo script -> boto3 Lambda Invoke (RequestResponse) -> direct ScanResult
```

No Function URL or API Gateway setup is required.

## 1) Prerequisites

- Rust 1.70+
- AWS CLI v2
- Python 3.8+
- AWS credentials with Lambda + IAM permissions

```bash
cargo install cargo-lambda
pip install -r modules/adaptive_trend_LTS_serverless/scripts/requirements.txt
```

## 2) Create IAM Role

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

## 3) Build + Deploy

```bash
cd modules/adaptive_trend_LTS_serverless/lambda

cargo lambda build --release --target x86_64-unknown-linux-gnu

cargo lambda deploy atc-serverless \
  --iam-role arn:aws:iam::YOUR_ACCOUNT:role/ATC-Lambda-ExecutionRole \
  --runtime provided.al2 \
  --region us-east-1

aws lambda update-function-configuration \
  --function-name atc-serverless \
  --memory-size 1769 \
  --timeout 60 \
  --environment "Variables={RUST_LOG=info}"
```

## 4) Run Demo

```bash
# Mock (local only)
python modules/adaptive_trend_LTS_serverless/scripts/binance_lambda_demo.py \
  --mock \
  --symbols 5

# Real Lambda invoke
python modules/adaptive_trend_LTS_serverless/scripts/binance_lambda_demo.py \
  --function-name atc-serverless \
  --region us-east-1 \
  --symbols 10 \
  --timeframes 1h 4h
```

## 5) Useful Variants

```bash
# 50 symbols
python modules/adaptive_trend_LTS_serverless/scripts/binance_lambda_demo.py \
  --function-name atc-serverless \
  --region us-east-1 \
  --symbols 50

# Detailed strengths
python modules/adaptive_trend_LTS_serverless/scripts/binance_lambda_demo.py \
  --function-name atc-serverless \
  --region us-east-1 \
  --symbols 20 \
  --details

# All symbols (costly)
python modules/adaptive_trend_LTS_serverless/scripts/binance_lambda_demo.py \
  --function-name atc-serverless \
  --region us-east-1 \
  --all-symbols \
  --timeframes 1h
```

## 6) Troubleshooting

### Access denied

```bash
aws sts get-caller-identity
```

Check caller permissions and role setup.

### Timeout

Increase timeout and/or reduce symbols per batch.

### Credentials error

```bash
aws configure
```

## Cleanup

```bash
aws lambda delete-function --function-name atc-serverless
aws iam detach-role-policy \
  --role-name ATC-Lambda-ExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role --role-name ATC-Lambda-ExecutionRole
```

