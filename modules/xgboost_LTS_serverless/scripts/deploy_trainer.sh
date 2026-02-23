#!/bin/bash
set -eo pipefail

STAGE=${1:-staging}
REGION=${2:-us-east-1}
MODEL_BUCKET=${3:-xgboost-models-store}

echo "Deploying XGBoost Trainer to Stage: $STAGE, Region: $REGION, Bucket: $MODEL_BUCKET"

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/xgboost-trainer:latest"

# Thư mục gốc của project (trở về 3 level từ thư mục scripts)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "${PROJECT_ROOT}"

sam deploy \
    --template-file modules/xgboost_LTS_serverless/template.yaml \
    --stack-name xgboost-lts-serverless-${STAGE} \
    --resolve-s3 \
    --capabilities CAPABILITY_IAM \
    --region ${REGION} \
    --parameter-overrides \
        ModelBucketName=${MODEL_BUCKET} \
    --image-repositories XGBoostTrainerFunction=${ECR_URI}

echo "Deployment complete."
