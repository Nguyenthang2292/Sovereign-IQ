#!/bin/bash
set -e

echo "Deploying XGBoost Serverless..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Building..."
bash scripts/build.sh

echo "Deploying to AWS..."
sam deploy \
  --template-file template.yaml \
  --stack-name xgboost-serverless \
  --capabilities CAPABILITY_IAM \
  --region us-east-1

echo "Deployment complete!"
echo "Use 'sam logs' to view function logs"
