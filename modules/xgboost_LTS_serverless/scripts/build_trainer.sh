#!/bin/bash
set -eo pipefail

echo "Building XGBoost Trainer Docker Image..."

# Thư mục gốc của project (trở về 3 level từ thư mục scripts)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "${PROJECT_ROOT}"

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-us-east-1}
REPO_NAME="xgboost-trainer"
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:latest"

echo "Logging into ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

echo "Checking if ECR repository exists..."
if ! aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${REGION} > /dev/null 2>&1; then
    echo "Creating ECR repository ${REPO_NAME}..."
    aws ecr create-repository --repository-name ${REPO_NAME} --region ${REGION}
else
    echo "ECR repository ${REPO_NAME} exists."
fi

echo "Building image (from project root)..."
docker build --provenance=false -f modules/xgboost_LTS_serverless/lambda/trainer/Dockerfile -t ${REPO_NAME} .

echo "Tagging image..."
docker tag ${REPO_NAME}:latest ${IMAGE_URI}

echo "Pushing image to ECR..."
docker push ${IMAGE_URI}

echo "Done! Image pushed to: ${IMAGE_URI}"
