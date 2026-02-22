# AWS Setup Guide

## Required Services

- Lambda (Rust custom runtime)
- API Gateway (REST)
- S3 (model registry)
- SQS (optional async delivery)
- CloudWatch Logs

## Prerequisites

- AWS CLI configured (`aws configure`)
- SAM CLI installed
- IAM permissions for CloudFormation, Lambda, S3, API Gateway, SQS, CloudWatch

## Environment Variables

- `MODEL_BUCKET`: S3 bucket storing model JSON files
- `RUST_LOG`: recommended `info`
- `RUST_BACKTRACE`: recommended `1`

## Deploy Stack

From module root:

```bash
sam build
sam deploy \
  --stack-name xgboost-serverless-staging \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides Environment=staging ModelBucketName=xgboost-models-staging
```

## Verify Deployment

```bash
sam list endpoints --stack-name xgboost-serverless-staging
aws lambda get-function --function-name xgboost-inference-staging
```

## IAM Notes

Lambda role must allow:

- `s3:GetObject` on model bucket path
- `sqs:SendMessage` when queue integration is enabled
- CloudWatch log write permissions
